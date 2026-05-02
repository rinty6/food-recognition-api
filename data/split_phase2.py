"""
Phase 2 — Stratified train/val split for all 101 Food-101 classes.

Splits the official training pool (750/class × 101 = 75,750 images) into:
  train : 600/class (80%) = 60,600 images
  val   : 150/class (20%) = 15,150 images
Keeps the official test set (250/class × 101 = 25,250 images) untouched.

Also writes:
  data/splits/food101_classes.json — index→class_name mapping for API serving

Usage:
  python data/split_phase2.py
"""

from pathlib import Path
import json
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR     = Path(__file__).parent.parent
FOOD101_ROOT = BASE_DIR / "data" / "food-101" / "food-101"
META_DIR     = FOOD101_ROOT / "meta"
IMAGES_DIR   = FOOD101_ROOT / "images"
SPLITS_DIR   = BASE_DIR / "data" / "splits"


def load_classes() -> list[str]:
    return sorted([
        line.strip()
        for line in (META_DIR / "classes.txt").read_text().splitlines()
        if line.strip()
    ])


def parse_meta_file(filename: str, class_to_idx: dict) -> list[dict]:
    lines = (META_DIR / filename).read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        class_name = line.split("/")[0]
        if class_name not in class_to_idx:
            continue
        records.append({
            "path": str(IMAGES_DIR / f"{line}.jpg"),
            "label": class_to_idx[class_name],
            "class_name": class_name,
        })
    return records


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    classes = load_classes()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"Found {len(classes)} classes")

    (SPLITS_DIR / "food101_classes.json").write_text(
        json.dumps(idx_to_class, indent=2)
    )
    print(f"Saved class map → data/splits/food101_classes.json")

    print("\nReading Food-101 metadata...")
    train_pool   = parse_meta_file("train.txt", class_to_idx)
    test_records = parse_meta_file("test.txt",  class_to_idx)
    print(f"  Train pool : {len(train_pool):,} images")
    print(f"  Test set   : {len(test_records):,} images (kept isolated)")

    labels = [r["label"] for r in train_pool]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(train_pool, labels))

    train_records = [train_pool[i] for i in train_idx]
    val_records   = [train_pool[i] for i in val_idx]

    for split_name, records, path in [
        ("train", train_records, SPLITS_DIR / "phase2_train.json"),
        ("val",   val_records,   SPLITS_DIR / "phase2_val.json"),
        ("test",  test_records,  SPLITS_DIR / "phase2_test.json"),
    ]:
        path.write_text(json.dumps(records, indent=2))
        print(f"\n  {split_name.upper()} → {path.name}: {len(records):,} images")

    print("\nSplit complete.")


if __name__ == "__main__":
    main()
