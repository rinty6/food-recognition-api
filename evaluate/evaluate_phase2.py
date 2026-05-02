"""
Phase 2 evaluation — runs the best ResNet-50 checkpoint against the official test set.

Writes: test_result/phase2_test_results.json

Usage:
  python evaluate/evaluate_phase2.py
  python evaluate/evaluate_phase2.py --checkpoint models/checkpoints/phase2_resnet50_best.pth
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from train.dataset import FoodDataset
from train.transforms import val_transforms
from train.utils import load_checkpoint
from train.utils import compute_accuracy
from models.resnet50_food101 import build_model

SPLITS_DIR      = BASE_DIR / "data" / "splits"
TEST_RESULT_DIR = BASE_DIR / "test_result"
DEFAULT_CKPT    = BASE_DIR / "models" / "checkpoints" / "phase2_resnet50_best.pth"
BATCH_SIZE      = 64


def main(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes_path = SPLITS_DIR / "food101_classes.json"
    idx_to_class = json.loads(classes_path.read_text())
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes  = len(idx_to_class)

    test_dataset = FoodDataset(SPLITS_DIR / "phase2_test.json", val_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Test samples : {len(test_dataset):,}")
    print(f"Classes      : {num_classes}")

    model = build_model().to(device)
    ckpt = load_checkpoint(checkpoint_path, model)
    print(f"Checkpoint   : epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_accuracy', 0):.4f}")
    model.eval()

    all_preds, all_targets = [], []
    total_top1, total_top5, n = 0.0, 0.0, 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            with autocast():
                logits = model(images)

            top1, top5 = compute_accuracy(logits, targets.to(device), topk=(1, 5))
            bs = images.size(0)
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            n += bs

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    test_acc  = total_top1 / n
    test_top5 = total_top5 / n

    label_list = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=label_list, zero_division=0
    )

    per_class = {}
    for idx in range(num_classes):
        cls = idx_to_class[idx]
        confused_with = _top_confusions(all_preds, all_targets, idx, idx_to_class)
        per_class[cls] = {
            "precision": round(float(precision[idx]), 4),
            "recall":    round(float(recall[idx]), 4),
            "f1":        round(float(f1[idx]), 4),
            "support":   int(support[idx]),
            "confused_with": confused_with,
        }

    worst = sorted(per_class.items(), key=lambda x: x[1]["f1"])[:10]
    best  = sorted(per_class.items(), key=lambda x: x[1]["f1"], reverse=True)[:10]

    result = {
        "phase": 2,
        "model_architecture": "resnet50",
        "num_classes": num_classes,
        "test_accuracy":      round(test_acc, 6),
        "test_top5_accuracy": round(test_top5, 6),
        "total_test_samples": n,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "per_class": per_class,
    }

    TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEST_RESULT_DIR / "phase2_test_results.json"
    output_path.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*55}")
    print(f"  Test Accuracy (top-1) : {test_acc*100:.2f}%")
    print(f"  Test Accuracy (top-5) : {test_top5*100:.2f}%")
    print(f"{'='*55}")

    print("\nTop 10 BEST classes (F1):")
    for cls, m in best:
        print(f"  {cls:<30} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    print("\nTop 10 WORST classes (F1):")
    for cls, m in worst:
        confused = ", ".join(m["confused_with"]) if m["confused_with"] else "—"
        print(f"  {cls:<30} F1={m['f1']:.4f}  confused_with=[{confused}]")

    print(f"\nResults saved → {output_path}")


def _top_confusions(preds, targets, class_idx, idx_to_class, top_n=3):
    counts = defaultdict(int)
    for p, t in zip(preds, targets):
        if t == class_idx and p != class_idx:
            counts[idx_to_class[p]] += 1
    return [k for k, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()
    main(args.checkpoint)
