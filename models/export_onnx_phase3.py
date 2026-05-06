"""
Exports the best Phase 3 ResNet-50 checkpoint to ONNX format.

Run AFTER train_phase3.py has finished and you are happy with val accuracy.

Usage:
  python models/export_onnx_phase3.py
  python models/export_onnx_phase3.py --checkpoint models/checkpoints/phase3_resnet50_epoch_025.pth
"""

import sys
import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.resnet50_extended import build_model
from train.utils import load_checkpoint

CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
DEFAULT_CKPT   = CHECKPOINT_DIR / "phase3_resnet50_best.pth"
OUTPUT_PATH    = CHECKPOINT_DIR / "resnet50_extended.onnx"


def main(checkpoint_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    num_classes = ckpt_raw.get("config", {}).get("num_classes", 187)
    print(f"  num_classes: {num_classes}")

    model = build_model(num_classes=num_classes)
    ckpt = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"  Epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_accuracy', 0):.4f}")

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"\nExporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        opset_version=18,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    print("Validating ONNX model...")
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid.")

    print("Checking output consistency (PyTorch vs ONNX)...")
    session = ort.InferenceSession(str(OUTPUT_PATH))
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    onnx_out = session.run(["logits"], {"image": dummy_input.numpy()})[0]
    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"  Max output difference: {max_diff:.2e}  (should be < 1e-4)")

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nONNX file: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print("\nNext step: update api/predictor.py to point to this file and phase3_classes.json,")
    print("then redeploy to Railway.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()
    main(args.checkpoint)
