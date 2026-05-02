"""
Reads epoch JSON files from training_result/ and validatation_result/,
and test_result/ to generate training charts.

Phase 1 results are in the root result directories.
Phase 2 results are in phase2/resnet50/ subdirectories.

Produces (saved to charts/output/):
  phase1_loss_curve.png, phase1_accuracy_curve.png, phase1_lr_curve.png, phase1_per_class_f1.png
  phase2_loss_curve.png, phase2_accuracy_curve.png, phase2_lr_curve.png,
  phase2_top5_accuracy_curve.png, phase2_per_class_f1_heatmap.png, phase2_confusion_top20.png

Usage:
  python charts/plot_results.py --phase 1
  python charts/plot_results.py --phase 2
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR   = Path(__file__).parent.parent
TEST_DIR   = BASE_DIR / "test_result"
OUTPUT_DIR = BASE_DIR / "charts" / "output"
STYLE      = "seaborn-v0_8-darkgrid"

RESULT_DIRS = {
    1: {
        "train": BASE_DIR / "training_result",
        "val":   BASE_DIR / "validatation_result",
    },
    2: {
        "train": BASE_DIR / "training_result"    / "phase2" / "resnet50",
        "val":   BASE_DIR / "validatation_result" / "phase2" / "resnet50",
    },
}


def load_epoch_files(directory: Path, phase: int) -> list[dict]:
    records = []
    for f in sorted(directory.glob("epoch_*.json")):
        data = json.loads(f.read_text())
        records.append(data)
    records.sort(key=lambda x: x["epoch"])
    return records


def plot_loss_curve(train_records, val_records, phase: int):
    epochs     = [r["epoch"] for r in train_records]
    train_loss = [r["loss"] for r in train_records]
    val_loss   = [r["val_loss"] for r in val_records]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, label="Training Loss",   color="#2196F3", linewidth=2)
    ax.plot(epochs, val_loss,   label="Validation Loss", color="#FF9800", linewidth=2)

    best_val_records = [r for r in val_records if r.get("is_best")]
    if best_val_records:
        best = min(best_val_records, key=lambda x: x["val_loss"])
        ax.annotate(
            f"Best val loss\nEpoch {best['epoch']}: {best['val_loss']:.4f}",
            xy=(best["epoch"], best["val_loss"]),
            xytext=(best["epoch"] + 1, best["val_loss"] + 0.05),
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green", fontsize=9,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Phase {phase} — Loss Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(epochs)
    fig.tight_layout()

    out = OUTPUT_DIR / f"phase{phase}_loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_accuracy_curve(train_records, val_records, phase: int):
    epochs     = [r["epoch"] for r in train_records]
    train_acc  = [r["accuracy"] * 100 for r in train_records]
    val_acc    = [r["val_accuracy"] * 100 for r in val_records]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_acc, label="Training Accuracy",   color="#2196F3", linewidth=2)
    ax.plot(epochs, val_acc,   label="Validation Accuracy", color="#FF9800", linewidth=2)

    best_epoch = max(range(len(val_acc)), key=lambda i: val_acc[i])
    ax.scatter(
        epochs[best_epoch], val_acc[best_epoch],
        color="green", s=120, zorder=5, label=f"Best val: {val_acc[best_epoch]:.1f}%"
    )

    ax.axhline(y=85, color="red", linestyle="--", alpha=0.6, label="Target: 85%")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Phase {phase} — Accuracy Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)
    fig.tight_layout()

    out = OUTPUT_DIR / f"phase{phase}_accuracy_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_lr_curve(train_records, phase: int):
    epochs = [r["epoch"] for r in train_records]
    lrs    = [r["lr"] for r in train_records]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, lrs, color="#9C27B0", linewidth=2, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate (log scale)", fontsize=12)
    ax.set_title(f"Phase {phase} — Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.set_xticks(epochs)
    fig.tight_layout()

    out = OUTPUT_DIR / f"phase{phase}_lr_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_per_class_f1(phase: int):
    test_file = TEST_DIR / f"phase{phase}_test_results.json"
    if not test_file.exists():
        print(f"  Skipping per-class F1: {test_file.name} not found (run evaluate_model.py first)")
        return

    data = json.loads(test_file.read_text())
    per_class = data["per_class"]
    classes = list(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]

    colors = ["#4CAF50" if f >= 0.80 else "#FF5722" for f in f1_scores]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(classes, [f * 100 for f in f1_scores], color=colors, edgecolor="white", linewidth=0.5)

    for bar, f in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{f * 100:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="Target: 80% F1")
    ax.set_xlabel("Food Class", fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title(
        f"Phase {phase} — Per-Class F1 Score  (Overall Accuracy: {data['test_accuracy']*100:.1f}%)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    fig.tight_layout()

    out = OUTPUT_DIR / f"phase{phase}_per_class_f1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_top5_curve(train_records, val_records, phase: int):
    """Phase 2 extra: top-5 accuracy curve."""
    epochs       = [r["epoch"] for r in train_records]
    train_top5   = [r.get("top5_accuracy", 0) * 100 for r in train_records]
    val_top5     = [r.get("val_top5_accuracy", 0) * 100 for r in val_records]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_top5, label="Train Top-5",      color="#2196F3", linewidth=2)
    ax.plot(epochs, val_top5,   label="Validation Top-5", color="#FF9800", linewidth=2)
    ax.axhline(y=95, color="red", linestyle="--", alpha=0.6, label="Target: 95%")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Top-5 Accuracy (%)", fontsize=12)
    ax.set_title(f"Phase {phase} — Top-5 Accuracy Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(80, 101)
    fig.tight_layout()
    out = OUTPUT_DIR / f"phase{phase}_top5_accuracy_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_phase2_f1_heatmap():
    """Phase 2: per-class F1 as a sorted horizontal bar chart (101 classes)."""
    test_file = TEST_DIR / "phase2_test_results.json"
    if not test_file.exists():
        print("  Skipping F1 heatmap: phase2_test_results.json not found")
        return

    data = json.loads(test_file.read_text())
    per_class = data["per_class"]
    sorted_items = sorted(per_class.items(), key=lambda x: x[1]["f1"])
    classes  = [c for c, _ in sorted_items]
    f1_vals  = [m["f1"] for _, m in sorted_items]
    colors   = ["#4CAF50" if f >= 0.80 else "#FF9800" if f >= 0.65 else "#F44336" for f in f1_vals]

    fig, ax = plt.subplots(figsize=(10, 22))
    bars = ax.barh(classes, [f * 100 for f in f1_vals], color=colors, edgecolor="none")
    ax.axvline(x=80, color="red", linestyle="--", alpha=0.7, label="Target 80%")
    ax.set_xlabel("F1 Score (%)", fontsize=12)
    ax.set_title(
        f"Phase 2 — Per-Class F1 Score\n"
        f"Overall Accuracy: {data['test_accuracy']*100:.1f}%  |  Top-5: {data['test_top5_accuracy']*100:.1f}%",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    out = OUTPUT_DIR / "phase2_per_class_f1_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_phase2_confusion_top20():
    """Phase 2: top-20 most-confused class pairs as a bar chart."""
    test_file = TEST_DIR / "phase2_test_results.json"
    if not test_file.exists():
        print("  Skipping confusion chart: phase2_test_results.json not found")
        return

    data = json.loads(test_file.read_text())
    per_class = data["per_class"]

    pair_counts = {}
    for cls, info in per_class.items():
        for confused in info.get("confused_with", []):
            pair = tuple(sorted([cls, confused]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    top20 = sorted(pair_counts.items(), key=lambda x: -x[1])[:20]
    if not top20:
        print("  No confusion data to plot yet.")
        return

    labels = [f"{a} ↔ {b}" for (a, b), _ in top20]
    counts = [c for _, c in top20]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels[::-1], counts[::-1], color="#FF5722", edgecolor="none")
    ax.set_xlabel("Number of mutual confusions", fontsize=12)
    ax.set_title("Phase 2 — Top 20 Most Confused Class Pairs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / "phase2_confusion_top20.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def main(phase: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts for Phase {phase}...")

    dirs = RESULT_DIRS[phase]
    train_records = load_epoch_files(dirs["train"], phase)
    val_records   = load_epoch_files(dirs["val"],   phase)

    if not train_records:
        print(f"  No training records found. Run the Phase {phase} training script first.")
        return
    if not val_records:
        print(f"  No validation records found.")
        return

    print(f"  Found {len(train_records)} training epochs, {len(val_records)} val epochs")

    plot_loss_curve(train_records, val_records, phase)
    plot_accuracy_curve(train_records, val_records, phase)
    plot_lr_curve(train_records, phase)
    plot_per_class_f1(phase)

    if phase == 2:
        plot_top5_curve(train_records, val_records, phase)
        plot_phase2_f1_heatmap()
        plot_phase2_confusion_top20()

    print(f"\nAll charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    main(args.phase)
