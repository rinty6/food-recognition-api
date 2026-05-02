"""
Phase 1 — Stratified train/val split for the 5-class subset.

Reads Food-101 official train.txt and test.txt.
Splits the official training pool (750/class) into:
  train : 600/class (80%)
  val   : 150/class (20%)
Keeps the official test set (250/class) completely untouched.

Writes three JSON files to data/splits/:
  phase1_train.json
  phase1_val.json
  phase1_test.json

Each file is a list of {"path": "...", "label": int, "class_name": str} dicts.
"""

from pathlib import Path
import json
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = Path(__file__).parent.parent
FOOD101_ROOT = BASE_DIR / "data" / "food-101" / "food-101"
META_DIR = FOOD101_ROOT / "meta"
IMAGES_DIR = FOOD101_ROOT / "images"
SPLITS_DIR = BASE_DIR / "data" / "splits"

PHASE1_CLASSES = ["ice_cream", "pancakes", "pizza", "sushi", "waffles"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(PHASE1_CLASSES)}


def parse_meta_file(filename: str) -> list[dict]:
    lines = (META_DIR / filename).read_text().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        class_name = line.split("/")[0]
        if class_name not in PHASE1_CLASSES:
            continue
        image_path = str(IMAGES_DIR / f"{line}.jpg")
        records.append({
            "path": image_path,
            "label": CLASS_TO_IDX[class_name],
            "class_name": class_name,
        })
    return records


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading Food-101 metadata...")
    train_pool = parse_meta_file("train.txt")
    test_records = parse_meta_file("test.txt")

    print(f"  Official train pool : {len(train_pool)} images across {len(PHASE1_CLASSES)} classes")
    print(f"  Official test set   : {len(test_records)} images (kept isolated)")

    labels = [r["label"] for r in train_pool]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(train_pool, labels))

    train_records = [train_pool[i] for i in train_idx]
    val_records = [train_pool[i] for i in val_idx]

    for split_name, records, path in [
        ("train", train_records, SPLITS_DIR / "phase1_train.json"),
        ("val",   val_records,   SPLITS_DIR / "phase1_val.json"),
        ("test",  test_records,  SPLITS_DIR / "phase1_test.json"),
    ]:
        path.write_text(json.dumps(records, indent=2))
        per_class = {}
        for r in records:
            per_class[r["class_name"]] = per_class.get(r["class_name"], 0) + 1
        print(f"\n  {split_name.upper()} split → {path.name} ({len(records)} images)")
        for cls in PHASE1_CLASSES:
            print(f"    {cls:<15} : {per_class.get(cls, 0)}")

    print("\nSplit complete. Files saved to data/splits/")


if __name__ == "__main__":
    main()
