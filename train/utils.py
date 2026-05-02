"""
Shared utilities: checkpoint save/load, accuracy computation, JSON result writers.
"""

import json
import torch
from datetime import datetime
from pathlib import Path


def save_checkpoint(state: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_accuracy(outputs, targets, topk=(1,)):
    """Returns top-k accuracy as a list of floats in [0, 1]."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / batch_size).item())
        return results


def write_training_epoch(result_dir: str | Path, epoch: int, phase: int,
                         loss: float, accuracy: float, top5: float,
                         lr: float, batch_size: int, samples: int, duration: float):
    result = {
        "phase": phase,
        "epoch": epoch,
        "loss": round(loss, 6),
        "accuracy": round(accuracy, 6),
        "top5_accuracy": round(top5, 6),
        "lr": lr,
        "batch_size": batch_size,
        "samples_seen": samples,
        "epoch_duration_seconds": round(duration, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(result_dir) / f"epoch_{epoch:03d}.json"
    path.write_text(json.dumps(result, indent=2))


def write_validation_epoch(result_dir: str | Path, epoch: int, phase: int,
                           val_loss: float, val_accuracy: float,
                           val_top5: float, is_best: bool):
    result = {
        "phase": phase,
        "epoch": epoch,
        "val_loss": round(val_loss, 6),
        "val_accuracy": round(val_accuracy, 6),
        "val_top5_accuracy": round(val_top5, 6),
        "is_best": is_best,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(result_dir) / f"epoch_{epoch:03d}.json"
    path.write_text(json.dumps(result, indent=2))


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
