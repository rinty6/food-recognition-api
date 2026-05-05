"""
Phase 3 dataset preparation — extends Food-101 (101 classes) with
Fruits-360 and a Vegetable Image Dataset to produce ~190 classes
that include raw fruits and vegetables.

INPUTS (download from Kaggle before running):
  data/fruits-360/Training/<Class Name>/*.jpg
  data/fruits-360/Test/<Class Name>/*.jpg
  data/vegetables/train/<Class Name>/*.jpg
  data/vegetables/validation/<Class Name>/*.jpg
  data/vegetables/test/<Class Name>/*.jpg

OUTPUTS (written to data/splits/):
  phase3_classes.json   — {label_index: class_name} for all N classes
  phase3_train.json     — [{path, label, class_name}, ...]
  phase3_val.json
  phase3_test.json

DESIGN DECISIONS:
  • Food-101 labels 0–100 are preserved exactly — the Phase 2 ResNet-50
    checkpoint weights remain valid for those classes when fine-tuning.
  • New classes are assigned labels 101, 102, ... in alphabetical order.
  • Class names are normalised to lower_snake_case to match Food-101 style.
  • Fruits-360 has variety-specific class names (apple_braeburn, apple_10,
    cherry_wax_red, etc.). These are consolidated into their canonical base
    class (apple, cherry, etc.) so the model learns generic fruit recognition,
    not fruit-variety classification. See consolidate_name() for the rules.
  • If the same consolidated name appears in both Fruits-360 and the
    Vegetable dataset, their image pools are merged.
  • Food-101 classes are loaded from the existing phase2 splits.

Usage:
  python data/prepare_extended_dataset.py             # writes splits
  python data/prepare_extended_dataset.py --dry-run   # prints summary only
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR    = Path(__file__).parent.parent
SPLITS_DIR  = BASE_DIR / "data" / "splits"
FRUITS_ROOT = BASE_DIR / "data" / "fruits-360"
VEG_ROOT    = BASE_DIR / "data" / "vegetables"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED  = 42
VAL_FRACTION = 0.20   # share of Fruits-360 Training/ used for validation

# ── Class-name consolidation ──────────────────────────────────────────────────

# Multi-word fruits/vegetables whose full name must be kept as-is.
# Without this list, "passion_fruit" would consolidate to "passion".
KNOWN_COMPOUND_NAMES = {
    "passion_fruit", "dragon_fruit", "star_fruit", "prickly_pear",
    "bitter_gourd", "bottle_gourd",
}

# Hard remap applied BEFORE the general consolidation rule.
# Handles Fruits-360 naming quirks that differ from common English names.
#   "cocos"    is what Fruits-360 calls coconut
#   "pitahaya" is what Fruits-360 calls dragon fruit
#   "carambola" is what Fruits-360 calls star fruit
#   "potato_sweet" should not merge into "potato"
HARD_REMAP: dict[str, str] = {
    "cocos":        "coconut",
    "pitahaya":     "dragon_fruit",
    "carambola":    "star_fruit",
    "potato_sweet": "sweet_potato",
    "passion":      "passion_fruit",   # Fruits-360 has both "Passion" and "Passion Fruit"
}


def normalize_name(raw: str) -> str:
    """'Apple Braeburn' → 'apple_braeburn', 'Bitter_Gourd' → 'bitter_gourd'"""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def consolidate_name(name: str) -> str:
    """
    Collapses Fruits-360 variety names into a canonical base class.

    Rules (applied in order):
    1. Hard remap table (e.g. 'cocos' → 'coconut').
    2. If the name is a known compound (e.g. 'passion_fruit'), keep it.
    3. Strip trailing purely-numeric tokens ('apple_10' → 'apple').
    4. If multiple tokens remain after step 3, take only the first token
       ('apple_braeburn' → 'apple', 'cherry_wax_red' → 'cherry').
       This treats all variety names as belonging to the primary fruit.

    Examples:
      apple_10        → apple
      apple_braeburn  → apple
      cherry_wax_red  → cherry
      banana          → banana          (unchanged, single word)
      passion_fruit   → passion_fruit   (preserved by rule 2)
      cocos           → coconut         (rule 1)
      potato_sweet    → sweet_potato    (rule 1)
    """
    # Rule 1 — hard remap (check prefix match too, e.g. 'pitahaya_red')
    for prefix, target in HARD_REMAP.items():
        if name == prefix or name.startswith(prefix + "_"):
            return target

    # Rule 2 — known compound names
    if name in KNOWN_COMPOUND_NAMES:
        return name

    parts = name.split("_")

    # Rule 3 — strip trailing numeric tokens
    while parts and parts[-1].isdigit():
        parts.pop()

    if not parts:
        return name

    # Rule 4 — if multiple tokens remain, it's a variety → keep only first
    return parts[0]


# ── Folder scanner ────────────────────────────────────────────────────────────

def scan_folder(folder: Path) -> dict[str, list[Path]]:
    """Scans one level of class subdirectories, applying name consolidation.
    Returns {consolidated_class_name: [image_path, ...]}.
    """
    result: dict[str, list[Path]] = defaultdict(list)
    if not folder.exists():
        return result
    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        raw   = normalize_name(class_dir.name)
        canon = consolidate_name(raw)
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            result[canon].extend(images)
    return result


def make_records(images: list[Path], label: int, class_name: str) -> list[dict]:
    return [{"path": str(p), "label": label, "class_name": class_name} for p in images]


# ── Load Food-101 existing splits ─────────────────────────────────────────────

def load_food101_splits() -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    train = json.loads((SPLITS_DIR / "phase2_train.json").read_text())
    val   = json.loads((SPLITS_DIR / "phase2_val.json").read_text())
    test  = json.loads((SPLITS_DIR / "phase2_test.json").read_text())

    class_to_label: dict[str, int] = {}
    for r in train + val + test:
        class_to_label[r["class_name"]] = r["label"]

    return train, val, test, class_to_label


# ── Scan new datasets ─────────────────────────────────────────────────────────

def scan_fruits360() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Returns (train_pool, test_pool) keyed by consolidated class name."""
    training_dir = FRUITS_ROOT / "Training"
    test_dir     = FRUITS_ROOT / "Test"

    # Kaggle zip sometimes nests an extra folder
    if not training_dir.exists():
        for sub in FRUITS_ROOT.iterdir():
            candidate = sub / "Training"
            if candidate.exists():
                training_dir = candidate
                test_dir     = sub / "Test"
                break

    return scan_folder(training_dir), scan_folder(test_dir)


def scan_vegetables() -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]]:
    """Returns (train, val, test) keyed by consolidated class name."""
    root = VEG_ROOT
    for candidate_name in ["Vegetable Images", "vegetable-image-dataset", "."]:
        candidate = VEG_ROOT / candidate_name
        if (candidate / "train").exists():
            root = candidate
            break

    return scan_folder(root / "train"), scan_folder(root / "validation"), scan_folder(root / "test")


# ── Merge new classes ─────────────────────────────────────────────────────────

def build_new_class_pool(
    fruits_train: dict[str, list[Path]],
    fruits_test:  dict[str, list[Path]],
    veg_train:    dict[str, list[Path]],
    veg_val:      dict[str, list[Path]],
    veg_test:     dict[str, list[Path]],
    food101_classes: set[str],
) -> dict[str, dict]:
    """
    Merges Fruits-360 and Vegetable datasets into a single class pool,
    skipping any class whose name collides with a Food-101 class.
    """
    all_new_names = (
        set(fruits_train) | set(fruits_test)
        | set(veg_train) | set(veg_val) | set(veg_test)
    )

    collisions = all_new_names & food101_classes
    if collisions:
        print(f"\n  [INFO] Skipping {len(collisions)} class(es) already in Food-101: {sorted(collisions)}")

    pool: dict[str, dict] = {}
    for name in sorted(all_new_names - collisions):
        pool[name] = {
            "fruits_train": fruits_train.get(name, []),
            "fruits_test":  fruits_test.get(name, []),
            "veg_train":    veg_train.get(name, []),
            "veg_val":      veg_val.get(name, []),
            "veg_test":     veg_test.get(name, []),
        }
    return pool


# ── Split new classes ─────────────────────────────────────────────────────────

def split_new_class(
    class_name: str,
    label: int,
    pool: dict,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Produces train/val/test record lists for one new class.

    Vegetable dataset already has train/val/test splits — use them directly.
    Fruits-360 has Training/ and Test/:
      • Test/ → test set
      • Training/ → shuffle, then 80% train / 20% val
    Images from both datasets are merged per split.
    """
    fruit_train_imgs = list(pool["fruits_train"])
    rng.shuffle(fruit_train_imgs)
    cut = int(len(fruit_train_imgs) * (1 - VAL_FRACTION))

    train_imgs = fruit_train_imgs[:cut]           + list(pool["veg_train"])
    val_imgs   = fruit_train_imgs[cut:]           + list(pool["veg_val"])
    test_imgs  = list(pool["fruits_test"])        + list(pool["veg_test"])

    return (
        make_records(train_imgs, label, class_name),
        make_records(val_imgs,   label, class_name),
        make_records(test_imgs,  label, class_name),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool):
    rng = random.Random(RANDOM_SEED)

    print("=" * 60)
    print("Phase 3 Dataset Preparation")
    print("=" * 60)

    # 1. Load Food-101
    print("\n[1/4] Loading existing Food-101 splits...")
    f101_train, f101_val, f101_test, food101_cls_to_label = load_food101_splits()
    food101_classes = set(food101_cls_to_label.keys())
    print(f"      Food-101 classes  : {len(food101_classes)}")
    print(f"      Train records     : {len(f101_train):,}")
    print(f"      Val   records     : {len(f101_val):,}")
    print(f"      Test  records     : {len(f101_test):,}")

    # 2. Scan Fruits-360
    print("\n[2/4] Scanning Fruits-360 (with variety consolidation)...")
    if not FRUITS_ROOT.exists():
        print(f"  ✗  NOT FOUND at {FRUITS_ROOT}")
        fruits_train_pool, fruits_test_pool = {}, {}
    else:
        fruits_train_pool, fruits_test_pool = scan_fruits360()
        raw_count = sum(
            1 for d in (FRUITS_ROOT / "Training").iterdir() if d.is_dir()
            if (FRUITS_ROOT / "Training").exists()
        )
        print(f"      Raw variety folders : {raw_count}")
        print(f"      After consolidation : {len(fruits_train_pool)} canonical classes")
        print(f"      Training images     : {sum(len(v) for v in fruits_train_pool.values()):,}")
        print(f"      Test images         : {sum(len(v) for v in fruits_test_pool.values()):,}")

    # 3. Scan Vegetable dataset
    print("\n[3/4] Scanning Vegetable Image Dataset...")
    if not VEG_ROOT.exists():
        print(f"  ✗  NOT FOUND at {VEG_ROOT}")
        veg_train_pool, veg_val_pool, veg_test_pool = {}, {}, {}
    else:
        veg_train_pool, veg_val_pool, veg_test_pool = scan_vegetables()
        print(f"      Classes found   : {len(veg_train_pool)}")
        print(f"      Train images    : {sum(len(v) for v in veg_train_pool.values()):,}")
        print(f"      Val   images    : {sum(len(v) for v in veg_val_pool.values()):,}")
        print(f"      Test  images    : {sum(len(v) for v in veg_test_pool.values()):,}")

    if not fruits_train_pool and not veg_train_pool:
        print("\n  Both new datasets are missing. Download them first, then re-run.")
        return

    # 4. Build combined class pool
    print("\n[4/4] Building combined class index...")
    new_class_pool = build_new_class_pool(
        fruits_train_pool, fruits_test_pool,
        veg_train_pool, veg_val_pool, veg_test_pool,
        food101_classes,
    )

    new_class_names = sorted(new_class_pool.keys())
    next_label = max(food101_cls_to_label.values()) + 1   # 101

    full_label_map: dict[int, str] = {v: k for k, v in food101_cls_to_label.items()}
    for i, name in enumerate(new_class_names):
        full_label_map[next_label + i] = name

    total_classes = len(full_label_map)

    print(f"\n  Food-101 classes     : {len(food101_classes)}")
    print(f"  New classes added    : {len(new_class_names)}")
    print(f"  Total classes        : {total_classes}")
    print(f"\n  New classes          : {sorted(new_class_names)}")

    # Build splits for new classes
    new_train, new_val, new_test = [], [], []
    skipped = []
    for i, name in enumerate(new_class_names):
        label = next_label + i
        tr, va, te = split_new_class(name, label, new_class_pool[name], rng)
        if not tr:
            skipped.append(name)
            continue
        new_train.extend(tr)
        new_val.extend(va)
        new_test.extend(te)

    if skipped:
        print(f"\n  [WARN] {len(skipped)} class(es) had no training images: {skipped}")

    combined_train = f101_train + new_train
    combined_val   = f101_val   + new_val
    combined_test  = f101_test  + new_test

    print(f"\n  Combined train records: {len(combined_train):,}")
    print(f"  Combined val   records: {len(combined_val):,}")
    print(f"  Combined test  records: {len(combined_test):,}")

    train_classes = {r["class_name"] for r in combined_train}
    val_classes   = {r["class_name"] for r in combined_val}
    missing_val   = train_classes - val_classes
    if missing_val:
        print(f"\n  [WARN] {len(missing_val)} class(es) in train have no val images: {sorted(missing_val)}")

    if dry_run:
        print("\n  --dry-run: no files written. Remove the flag to write splits.")
        return

    # Write files
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    (SPLITS_DIR / "phase3_classes.json").write_text(
        json.dumps(full_label_map, indent=2, sort_keys=False)
    )
    print(f"\n  Written: data/splits/phase3_classes.json  ({total_classes} classes)")

    for name, records, path in [
        ("train", combined_train, SPLITS_DIR / "phase3_train.json"),
        ("val",   combined_val,   SPLITS_DIR / "phase3_val.json"),
        ("test",  combined_test,  SPLITS_DIR / "phase3_test.json"),
    ]:
        path.write_text(json.dumps(records, indent=2))
        print(f"  Written: data/splits/phase3_{name}.json  ({len(records):,} records)")

    print("\nDone. Phase 1 complete — proceed to Phase 2 (model + training script).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing any files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
