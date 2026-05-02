"""
Stage C fine-tuning — resumes from the Phase 2 Stage B best checkpoint and
runs the layer3 + layer4 + label-smoothing stage that was killed by the LR
reset in train_phase2.py.

What was wrong in train_phase2.py:
  - Stage C reset the LR back to 1e-3 (same as Stage A start)
  - The early stopping counter had already accumulated 7 counts during the
    Stage B plateau (epochs 19-25), so the val_loss spike from the LR reset
    triggered early stopping on the very first Stage C epoch (epoch 26)
  - Result: Stage C never ran, model stopped at epoch 25

What this script does differently:
  - Starts from the Stage B best checkpoint (epoch 25, val_acc ~77.8%)
  - Uses a much lower LR appropriate for a nearly-converged model:
      head    : 1e-4
      layer4  : 5e-5
      layer3  : 2e-5
  - Resets early stopping with a fresh counter
  - Runs CosineAnnealingLR from the lower starting point
  - Adds label smoothing (0.1) as planned for Stage C
  - Runs for 25 epochs (equivalent to the original epochs 26-50)

Expected outcome: test accuracy 84-87% (up from 82.2%)

Usage:
  python train/train_stage_c.py
  python train/train_stage_c.py --resume models/checkpoints/phase2_resnet50_stagec_epoch_010.pth
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
from models.resnet50_food101 import build_model, activate_stage_b, activate_stage_c

# ── Stage C hyperparameters ────────────────────────────────────────────────────
# Much lower than Stage B start (1e-3) — model is already well-converged
LR_HEAD    = 1e-4
LR_LAYER4  = 5e-5
LR_LAYER3  = 2e-5

BATCH_SIZE       = 32
NUM_WORKERS      = 4
STAGE_C_EPOCHS   = 25
CHECKPOINT_EVERY = 5
ES_PATIENCE      = 10     # more generous — Stage C loss changes slowly
PHASE            = 2
LABEL_SMOOTHING  = 0.1

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLITS_DIR       = BASE_DIR / "data" / "splits"
TRAIN_RESULT_DIR = BASE_DIR / "training_result"    / "phase2" / "resnet50"
VAL_RESULT_DIR   = BASE_DIR / "validatation_result" / "phase2" / "resnet50"
CHECKPOINT_DIR   = BASE_DIR / "models" / "checkpoints"
STAGE_B_CKPT     = CHECKPOINT_DIR / "phase2_resnet50_best.pth"
BEST_CKPT        = CHECKPOINT_DIR / "phase2_resnet50_stagec_best.pth"


def build_stagec_optimizer(model) -> torch.optim.Optimizer:
    """Separate param groups with differentiated LRs for each block."""
    groups = [
        {"params": list(model.fc.parameters()),     "lr": LR_HEAD,   "name": "head"},
        {"params": list(model.layer4.parameters()), "lr": LR_LAYER4, "name": "layer4"},
        {"params": list(model.layer3.parameters()), "lr": LR_LAYER3, "name": "layer3"},
    ]
    return torch.optim.Adam(groups, weight_decay=1e-4)


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


def main(resume_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_dataset = FoodDataset(SPLITS_DIR / "phase2_train.json", train_transforms)
    val_dataset   = FoodDataset(SPLITS_DIR / "phase2_val.json",   val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # ── Load model and apply stage activations ─────────────────────────────────
    model = build_model().to(device)

    source_ckpt = resume_path or str(STAGE_B_CKPT)
    print(f"\nLoading checkpoint: {source_ckpt}")
    ckpt = load_checkpoint(source_ckpt, model)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_accuracy', 0):.4f}")

    # Unfreeze both blocks — Stage C requires layer3 + layer4
    activate_stage_b(model)   # layer4
    activate_stage_c(model)   # layer3

    optimizer = build_stagec_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE_C_EPOCHS, eta_min=1e-7
    )
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Epoch offset: continue epoch numbers from where Stage B left off
    epoch_offset = ckpt.get("epoch", 25)
    best_val_acc = ckpt.get("val_accuracy", 0.0)
    early_stopper = EarlyStopping(patience=ES_PATIENCE)

    print(f"\nStage C — {STAGE_C_EPOCHS} epochs")
    print(f"  LR  head={LR_HEAD:.0e}  layer4={LR_LAYER4:.0e}  layer3={LR_LAYER3:.0e}")
    print(f"  Label smoothing={LABEL_SMOOTHING}")
    print(f"  Starting from epoch {epoch_offset + 1}\n")

    for step in range(1, STAGE_C_EPOCHS + 1):
        epoch = epoch_offset + step
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
                "stage": "C",
                "config": {"architecture": "resnet50", "num_classes": 101},
            }, BEST_CKPT)

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "stage": "C",
                "config": {"architecture": "resnet50", "num_classes": 101},
            }, CHECKPOINT_DIR / f"phase2_resnet50_stagec_epoch_{step:03d}.pth")

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
            f"[C] Ep {epoch:02d} (step {step:02d}/{STAGE_C_EPOCHS}) | "
            f"Train {train_acc:.4f} (top5 {train_top5:.4f}) loss {train_loss:.4f} | "
            f"Val {val_acc:.4f} (top5 {val_top5:.4f}) loss {val_loss:.4f} | "
            f"LR {current_lr:.2e} | {mins}m{secs:02d}s{marker}"
        )

        if early_stopper.step(val_loss):
            print(f"\nEarly stopping at step {step} (epoch {epoch})")
            break

    print(f"\nStage C complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")
    print(f"\nNext: python evaluate/evaluate_phase2.py "
          f"--checkpoint {BEST_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a Stage C checkpoint (not Stage B — use default for that)")
    args = parser.parse_args()
    main(resume_path=args.resume)
