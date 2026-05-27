"""
Phase 3.5 dataset preparation — merges FruitVision (real-world fruit photos)
into the existing Phase 3 splits.

WHY THIS EXISTS
---------------
Phase 3 trained on Fruits-360, which is a studio dataset (white backgrounds,
controlled lighting, perfectly ripe fruit). The model fails on real-world
photos where the same fruit appears against a natural background, is overripe,
held in a hand, or has uneven lighting.

FruitVision (Daffodil International University) contains real-world photos of
5 fruits in 3 condition states:
  - Fresh    : typical ripe fruit
  - Rotten   : overripe, brown/black, mouldy
  - Formalin : preserved (adds visual diversity)

All 3 states are merged into ONE canonical class label (rotten_apple → apple).
A user logging food does not care whether their banana is fresh or overripe —
they just want "banana." Training on all states makes the model robust to them.

SUPPORTED FOLDER LAYOUTS
-------------------------
The script auto-detects several naming conventions used by different
FruitVision releases on Kaggle / Roboflow / GitHub:

  Layout A (flat):
    fruitvision/
      fresh_apple/     rotten_apple/     formalin_apple/
      fresh_banana/    rotten_banana/    formalin_banana/
      ...

  Layout B (split):
    fruitvision/
      train/ fresh_apple/  rotten_apple/ ...
      test/  fresh_apple/  rotten_apple/ ...

  Layout C (nested):
    fruitvision/
      Apple/ Fresh/   Rotten/   Formalin/
      Banana/ Fresh/  Rotten/   Formalin/

EFFECT ON SPLITS
----------------
New image records are APPENDED to phase3_train.json and phase3_val.json
using an 80/20 split of available images (reproducible via SEED=42).
phase3_test.json is NEVER touched — it stays clean for fair evaluation.
phase3_classes.json is NOT changed — same 187 labels.

USAGE
-----
  # Show what would change without writing anything
  python data/prepare_fruitvision.py --dry-run

  # Merge (default path: data/fruitvision/)
  python data/prepare_fruitvision.py

  # Custom dataset location
  python data/prepare_fruitvision.py --dataset path/to/fruitvision

AFTER THIS SCRIPT
-----------------
  python train/train_phase3.py
  (fresh start from Phase 2 backbone — all data including FruitVision)
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path(__file__).parent.parent
SPLITS_DIR = BASE_DIR / "data" / "splits"
DEFAULT_DATASET_DIR = BASE_DIR / "data" / "fruitvision"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VAL_RATIO = 0.20
SEED      = 42

# ── Condition prefixes/suffixes we strip to get the fruit name ────────────────
# Handles: fresh_apple, rotten_apple, formalin_apple,
#          apple_fresh, apple_rotten, apple_overripe, etc.
CONDITION_TOKENS = {
    "fresh", "rotten", "formalin", "overripe", "mixed",
    "good", "bad", "spoiled", "unripe", "raw",
}

# ── Canonical name map: resolves naming quirks across dataset versions ─────────
# Keys are lower_snake_case folder names AFTER stripping condition tokens.
CANONICAL: dict[str, str] = {
    "apple":    "apple",
    "banana":   "banana",
    "mango":    "mango",
    "orange":   "orange",
    "grape":    "grape",
    "grapes":   "grape",
    "kiwi":     "kiwi",       # some releases include kiwi
    "peach":    "peach",
    "pear":     "pear",
    "lemon":    "lemon",
    "lime":     "limes",      # match our phase3 class name "limes"
}


# ── Name normalisation ────────────────────────────────────────────────────────

def _normalise(raw: str) -> str:
    """'Fresh Apple' → 'fresh_apple'"""
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def _strip_condition(name: str) -> str:
    """
    Removes condition prefix or suffix from a folder name.
      fresh_apple  → apple
      rotten_apple → apple
      apple_fresh  → apple
      apple        → apple   (no condition token present)
    """
    parts = name.split("_")
    filtered = [p for p in parts if p not in CONDITION_TOKENS]
    return "_".join(filtered) if filtered else name


def folder_to_canonical(folder_name: str) -> str | None:
    """
    Converts a folder name to its canonical class name, or None if not handled.

    Examples:
      'fresh_apple'   → 'apple'
      'Rotten_Banana' → 'banana'
      'Apple'         → 'apple'
      'mango_formalin'→ 'mango'
      'broccoli'      → None   (not a FruitVision class)
    """
    norm    = _normalise(folder_name)
    stripped = _strip_condition(norm)
    return CANONICAL.get(stripped)


# ── Dataset scanner ───────────────────────────────────────────────────────────

def scan_dataset(dataset_dir: Path) -> dict[str, list[Path]]:
    """
    Recursively scans dataset_dir for image folders.
    Returns {canonical_class: [image_path, ...]} across all condition variants.

    Handles three layouts:
      Layout A (flat):    fresh_apple/  rotten_apple/   → fruit name in folder itself
      Layout B (split):   train/fresh_apple/            → fruit name in folder itself
      Layout C (nested):  Apple/Fresh/  Apple/Rotten/   → fruit name in PARENT folder

    For Layout C (used by FruitVision Original Image release), the leaf folders are
    named 'Fresh', 'Rotten', 'Formalin-mixed' — these are pure condition words with
    no fruit name.  We fall back to checking the parent folder ('Apple', 'Banana',
    etc.) to resolve the canonical class.
    """
    result: dict[str, list[Path]] = defaultdict(list)

    for folder in dataset_dir.rglob("*"):
        if not folder.is_dir():
            continue

        images = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            continue  # skip directories that contain no images directly

        # Try the folder name first (Layout A / B: "fresh_apple", "Apple")
        canonical = folder_to_canonical(folder.name)

        # Fall back to parent folder name (Layout C: "Apple/Fresh" → parent is "Apple")
        if canonical is None:
            canonical = folder_to_canonical(folder.parent.name)

        if canonical is None:
            continue

        result[canonical].extend(images)

    return dict(result)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ── Download hint ─────────────────────────────────────────────────────────────

DOWNLOAD_HINT = """
FruitVision dataset not found at: {path}

Download it and place it at that location. Expected structure:
  fruitvision/
    fresh_apple/    rotten_apple/    formalin_apple/
    fresh_banana/   rotten_banana/   formalin_banana/
    fresh_mango/    rotten_mango/    formalin_mango/
    fresh_orange/   rotten_orange/   formalin_orange/
    fresh_grapes/   rotten_grapes/   formalin_grapes/

Suggested sources:
  Kaggle: search "fruitvision fresh rotten" or
           "fresh and rotten fruit detection daffodil"
  GitHub: Daffodil International University fruit detection repo

The script also handles nested layouts (Apple/Fresh/, Banana/Rotten/, etc.)
and split layouts (train/fresh_apple/, test/fresh_apple/, etc.).
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dataset_dir: Path, dry_run: bool) -> None:
    # 1. Check dataset exists
    if not dataset_dir.exists():
        print(DOWNLOAD_HINT.format(path=dataset_dir))
        sys.exit(1)

    # 2. Load existing splits and class map
    train_split = _load_json(SPLITS_DIR / "phase3_train.json")
    val_split   = _load_json(SPLITS_DIR / "phase3_val.json")
    classes     = _load_json(SPLITS_DIR / "phase3_classes.json")
    name_to_label: dict[str, int] = {v: int(k) for k, v in classes.items()}

    # 3. Scan FruitVision
    print(f"\nScanning FruitVision dataset: {dataset_dir}")
    images_by_class = scan_dataset(dataset_dir)

    if not images_by_class:
        print("\nERROR: No recognisable fruit folders found.")
        print("Folder names must contain a fruit name (apple/banana/mango/orange/grape/grapes).")
        print("Current CANONICAL map:", list(CANONICAL.keys()))
        sys.exit(1)

    # 4. Build new records
    rng = random.Random(SEED)
    new_train: list[dict] = []
    new_val:   list[dict] = []

    print()
    print(f"{'Class':<14} {'Label':>5}  {'Total':>6}  {'->Train':>7}  {'->Val':>6}  Folders found")
    print("-" * 72)

    skipped = []
    for class_name in sorted(images_by_class):
        label = name_to_label.get(class_name)
        if label is None:
            skipped.append(class_name)
            continue

        all_images = list(images_by_class[class_name])
        rng.shuffle(all_images)
        cut        = int(len(all_images) * (1 - VAL_RATIO))
        train_imgs = all_images[:cut]
        val_imgs   = all_images[cut:]

        for p in train_imgs:
            new_train.append({
                "path":       str(p),
                "label":      label,
                "class_name": class_name,
                "source":     "fruitvision",
            })
        for p in val_imgs:
            new_val.append({
                "path":       str(p),
                "label":      label,
                "class_name": class_name,
                "source":     "fruitvision",
            })

        # Count unique condition folders for this class
        condition_dirs = {p.parent.name for p in all_images}
        print(
            f"{class_name:<14} {label:>5}  {len(all_images):>6,}  "
            f"{len(train_imgs):>7,}  {len(val_imgs):>6,}  {sorted(condition_dirs)}"
        )

    print("-" * 72)

    if skipped:
        print(f"\nSKIPPED (not in phase3_classes.json): {skipped}")

    total_new = len(new_train) + len(new_val)
    print(f"\nNew records  : {total_new:,}  ({len(new_train):,} train / {len(new_val):,} val)")
    print(f"After merge  : {len(train_split) + len(new_train):,} train  /  "
          f"{len(val_split) + len(new_val):,} val")

    if dry_run:
        print("\n[DRY RUN] Nothing written. Remove --dry-run to apply.")
        return

    if total_new == 0:
        print("\nNothing to write — no new records generated.")
        return

    # 5. Append to splits
    train_split.extend(new_train)
    val_split.extend(new_val)

    _save_json(SPLITS_DIR / "phase3_train.json", train_split)
    _save_json(SPLITS_DIR / "phase3_val.json",   val_split)

    print(f"\nUpdated:")
    print(f"  {SPLITS_DIR / 'phase3_train.json'}")
    print(f"  {SPLITS_DIR / 'phase3_val.json'}")
    print(f"\nphase3_classes.json unchanged — same 187 labels.")
    print(f"\nNext step:")
    print(f"  python train/train_phase3.py")
    print(f"  (fresh start from Phase 2 backbone — now includes FruitVision images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge FruitVision real-world images into Phase 3 splits."
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_DIR),
        help=f"Path to FruitVision folder (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing any files",
    )
    args = parser.parse_args()
    main(Path(args.dataset), args.dry_run)
