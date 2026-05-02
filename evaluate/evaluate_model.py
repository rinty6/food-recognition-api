"""
Phase 1 evaluation — runs the best checkpoint against the official test set.

Writes: test_result/phase1_test_results.json

Usage:
  python evaluate/evaluate_model.py
  python evaluate/evaluate_model.py --checkpoint models/checkpoints/phase1_best.pth
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from train.dataset import FoodDataset
from train.transforms import val_transforms
from train.utils import load_checkpoint, compute_accuracy
from models.baseline_resnet18 import build_model

PHASE1_CLASSES = ["ice_cream", "pancakes", "pizza", "sushi", "waffles"]
SPLITS_DIR     = BASE_DIR / "data" / "splits"
TEST_RESULT_DIR = BASE_DIR / "test_result"
DEFAULT_CKPT   = BASE_DIR / "models" / "checkpoints" / "phase1_best.pth"
BATCH_SIZE     = 64


def main(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = FoodDataset(SPLITS_DIR / "phase1_test.json", val_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    model = build_model().to(device)
    ckpt = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    total = len(all_targets)
    top1_correct = sum(p == t for p, t in zip(all_preds, all_targets))
    test_acc = top1_correct / total

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(PHASE1_CLASSES))), zero_division=0
    )

    per_class = {}
    for idx, cls in enumerate(PHASE1_CLASSES):
        confused_with = _top_confusions(all_preds, all_targets, idx, PHASE1_CLASSES)
        per_class[cls] = {
            "precision": round(float(precision[idx]), 4),
            "recall":    round(float(recall[idx]), 4),
            "f1":        round(float(f1[idx]), 4),
            "support":   int(support[idx]),
            "confused_with": confused_with,
        }

    result = {
        "phase": 1,
        "model_architecture": "resnet18",
        "num_classes": len(PHASE1_CLASSES),
        "test_accuracy": round(test_acc, 6),
        "test_top5_accuracy": 1.0,
        "test_loss": None,
        "total_test_samples": total,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "per_class": per_class,
    }

    TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEST_RESULT_DIR / "phase1_test_results.json"
    output_path.write_text(json.dumps(result, indent=2))

    print(f"\nTest Accuracy : {test_acc:.4f} ({top1_correct}/{total})")
    print("\nPer-class F1 scores:")
    for cls in PHASE1_CLASSES:
        print(f"  {cls:<15} : F1={per_class[cls]['f1']:.4f}  "
              f"P={per_class[cls]['precision']:.4f}  R={per_class[cls]['recall']:.4f}")
    print(f"\nResults saved to {output_path}")


def _top_confusions(preds, targets, class_idx, classes, top_n=2):
    confusion_counts = defaultdict(int)
    for p, t in zip(preds, targets):
        if t == class_idx and p != class_idx:
            confusion_counts[classes[p]] += 1
    return [k for k, _ in sorted(confusion_counts.items(), key=lambda x: -x[1])[:top_n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()
    main(args.checkpoint)
