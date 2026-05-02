"""
Phase 2 training script — ResNet-50 on all 101 Food-101 classes.

Three-stage discriminative fine-tuning:
  Stage A (epochs  1–10) : frozen conv base, head only, LR=1e-3
  Stage B (epochs 11–25) : layer4 unfrozen, LR_layer4=1e-4
  Stage C (epochs 26–50) : layer3+layer4 unfrozen, label_smoothing=0.1

Results written to:
  training_result/phase2/resnet50/epoch_NNN.json
  validatation_result/phase2/resnet50/epoch_NNN.json

Checkpoints saved to:
  models/checkpoints/phase2_resnet50_best.pth
  models/checkpoints/phase2_resnet50_epoch_NNN.pth  (every 5 epochs)

Usage:
  python train/train_phase2.py
  python train/train_phase2.py --resume models/checkpoints/phase2_resnet50_epoch_025.pth
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
from models.resnet50_food101 import build_model, activate_stage_b, activate_stage_c, get_param_groups

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
NUM_WORKERS       = 4
TOTAL_EPOCHS      = 50
STAGE_A_END       = 10    # epochs 1–10:  head only
STAGE_B_END       = 25    # epochs 11–25: + layer4
                          # epochs 26–50: + layer3, label smoothing
BASE_LR           = 1e-3
WEIGHT_DECAY      = 1e-4
CHECKPOINT_EVERY  = 5     # save a checkpoint every N epochs
ES_PATIENCE       = 8     # early stopping patience (longer than phase 1 — 101 classes converge slower)
PHASE             = 2

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR       = BASE_DIR / "data" / "splits"
TRAIN_RESULT_DIR = BASE_DIR / "training_result"   / "phase2" / "resnet50"
VAL_RESULT_DIR   = BASE_DIR / "validatation_result" / "phase2" / "resnet50"
CHECKPOINT_DIR   = BASE_DIR / "models" / "checkpoints"
BEST_CKPT        = CHECKPOINT_DIR / "phase2_resnet50_best.pth"

TRAIN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
VAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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
            [p for pg in optimizer.param_groups for p in pg["params"] if p.requires_grad],
            max_norm=1.0,
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
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def build_optimizer_and_scheduler(model, base_lr, total_epochs):
    optimizer = torch.optim.Adam(
        get_param_groups(model, base_lr),
        lr=base_lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6
    )
    return optimizer, scheduler


def main(resume_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset = FoodDataset(SPLITS_DIR / "phase2_train.json", train_transforms)
    val_dataset   = FoodDataset(SPLITS_DIR / "phase2_val.json",   val_transforms)
    print(f"\nTrain  : {len(train_dataset):,} images")
    print(f"Val    : {len(val_dataset):,} images")
    print(f"Classes: {len(train_dataset.classes)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = build_model().to(device)
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_hard   = nn.CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(model, BASE_LR, TOTAL_EPOCHS)
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=ES_PATIENCE)

    start_epoch  = 1
    best_val_acc = 0.0
    current_stage = "A"

    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer)
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_accuracy", 0.0)
        print(f"\nResumed from epoch {start_epoch - 1}, best_val_acc={best_val_acc:.4f}")
        # Re-apply correct stage activations
        if start_epoch > STAGE_A_END + 1:
            activate_stage_b(model)
        if start_epoch > STAGE_B_END + 1:
            activate_stage_c(model)
            criterion_hard = criterion_smooth
        optimizer, scheduler = build_optimizer_and_scheduler(model, BASE_LR, TOTAL_EPOCHS)
        for _ in range(start_epoch - 1):
            scheduler.step()

    print(f"\nStarting Phase 2 — ResNet-50, {TOTAL_EPOCHS} epochs")
    print(f"  Stage A (1–{STAGE_A_END})  : head only")
    print(f"  Stage B ({STAGE_A_END+1}–{STAGE_B_END}) : + layer4")
    print(f"  Stage C ({STAGE_B_END+1}–{TOTAL_EPOCHS}) : + layer3, label smoothing\n")

    criterion = criterion_hard

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        # ── Stage transitions ──────────────────────────────────────────────────
        if epoch == STAGE_A_END + 1 and current_stage == "A":
            activate_stage_b(model)
            optimizer, scheduler = build_optimizer_and_scheduler(model, BASE_LR, TOTAL_EPOCHS - epoch + 1)
            current_stage = "B"

        if epoch == STAGE_B_END + 1 and current_stage == "B":
            activate_stage_c(model)
            optimizer, scheduler = build_optimizer_and_scheduler(model, BASE_LR, TOTAL_EPOCHS - epoch + 1)
            criterion = criterion_smooth
            current_stage = "C"

        current_lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()

        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, device)

        epoch_secs = time.time() - t0
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "stage": current_stage,
                "config": {"architecture": "resnet50", "num_classes": 101},
            }, BEST_CKPT)

        if epoch % CHECKPOINT_EVERY == 0:
            periodic_path = CHECKPOINT_DIR / f"phase2_resnet50_epoch_{epoch:03d}.pth"
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "stage": current_stage,
                "config": {"architecture": "resnet50", "num_classes": 101},
            }, periodic_path)

        write_training_epoch(
            TRAIN_RESULT_DIR, epoch, PHASE,
            train_loss, train_acc, train_top5,
            current_lr, BATCH_SIZE, len(train_dataset), epoch_secs,
        )
        write_validation_epoch(
            VAL_RESULT_DIR, epoch, PHASE,
            val_loss, val_acc, val_top5, is_best,
        )

        marker = " ← best" if is_best else ""
        mins, secs = divmod(int(epoch_secs), 60)
        print(
            f"[{current_stage}] Ep {epoch:02d}/{TOTAL_EPOCHS} | "
            f"Train {train_acc:.4f} (top5 {train_top5:.4f}) loss {train_loss:.4f} | "
            f"Val {val_acc:.4f} (top5 {val_top5:.4f}) loss {val_loss:.4f} | "
            f"LR {current_lr:.2e} | {mins}m{secs:02d}s{marker}"
        )

        if early_stopper.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} ({ES_PATIENCE} epochs without improvement)")
            break

    print(f"\nPhase 2 complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(resume_path=args.resume)
