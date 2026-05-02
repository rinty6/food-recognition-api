"""
Phase 1 training script — ResNet-18 on 5-class Food-101 subset.

Writes epoch results to:
  training_result/epoch_NNN.json
  validatation_result/epoch_NNN.json

Saves best checkpoint to:
  models/checkpoints/phase1_best.pth

Usage:
  python train/train_phase1.py
  python train/train_phase1.py --resume models/checkpoints/phase1_best.pth
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from train.dataset import FoodDataset
from train.transforms import train_transforms, val_transforms
from train.utils import (
    save_checkpoint, load_checkpoint, compute_accuracy,
    write_training_epoch, write_validation_epoch, EarlyStopping,
)
from models.baseline_resnet18 import build_model, activate_stage_b, get_param_groups

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_WORKERS   = 4
TOTAL_EPOCHS  = 20
HEAD_EPOCHS   = 5       # Stage A: head only
BASE_LR       = 1e-3
WEIGHT_DECAY  = 1e-4
ES_PATIENCE   = 5       # Early stopping patience
PHASE         = 1

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR       = BASE_DIR / "data" / "splits"
TRAIN_RESULT_DIR = BASE_DIR / "training_result"
VAL_RESULT_DIR   = BASE_DIR / "validatation_result"
CHECKPOINT_DIR   = BASE_DIR / "models" / "checkpoints"
BEST_CKPT        = CHECKPOINT_DIR / "phase1_best.pth"


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for pg in optimizer.param_groups for p in pg["params"]], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        top1, top5 = compute_accuracy(logits.detach(), targets, topk=(1, 5))

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_top1, total_top5, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def main(resume_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = FoodDataset(SPLITS_DIR / "phase1_train.json", train_transforms)
    val_dataset   = FoodDataset(SPLITS_DIR / "phase1_val.json",   val_transforms)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(get_param_groups(model, BASE_LR),
                                 lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True
    )
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=ES_PATIENCE)

    start_epoch = 1
    best_val_acc = 0.0

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_accuracy", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best_val_acc={best_val_acc:.4f}")

    print(f"\nStarting training — {TOTAL_EPOCHS} epochs, Stage A for {HEAD_EPOCHS} epochs\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        if epoch == HEAD_EPOCHS + 1:
            activate_stage_b(model)
            optimizer = torch.optim.Adam(
                get_param_groups(model, BASE_LR),
                lr=BASE_LR, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5, verbose=True
            )

        current_lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()

        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)

        epoch_secs = time.time() - t0
        scheduler.step(val_loss)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "config": {"architecture": "resnet18", "num_classes": 5},
            }, BEST_CKPT)

        write_training_epoch(
            TRAIN_RESULT_DIR, epoch, PHASE,
            train_loss, train_acc, train_top5,
            current_lr, BATCH_SIZE, len(train_dataset), epoch_secs
        )
        write_validation_epoch(
            VAL_RESULT_DIR, epoch, PHASE,
            val_loss, val_acc, val_top5, is_best
        )

        marker = " ← best" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"LR {current_lr:.6f} | {epoch_secs:.0f}s{marker}"
        )

        if early_stopper.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {ES_PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    main(resume_path=args.resume)
