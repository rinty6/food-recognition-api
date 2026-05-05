"""
Model inference for the food recognition API.

Loads the ONNX model on startup (preferred — 2-4x faster on CPU).
Falls back to PyTorch if ONNX is not available (e.g. during local dev
before export_onnx.py has been run).

Image preprocessing matches the validation transforms used during training:
  Resize(256) → CenterCrop(224) → Normalize(ImageNet stats)
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent.parent

ONNX_MODEL_PATH  = BASE_DIR / "models" / "checkpoints" / "resnet50_extended.onnx"
PYTORCH_CKPT     = BASE_DIR / "models" / "checkpoints" / "phase3_resnet50_best.pth"
CLASSES_JSON     = BASE_DIR / "data" / "splits" / "phase3_classes.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOP_K = 5


def _load_class_names() -> list[str]:
    with open(CLASSES_JSON) as f:
        mapping = json.load(f)
    # mapping is {"0": "apple_pie", "1": "baby_back_ribs", ...}
    return [mapping[str(i)] for i in range(len(mapping))]


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize(256) → CenterCrop(224) → float32 → Normalize → NCHW."""
    image = image.convert("RGB")

    # Resize shortest side to 256
    w, h = image.size
    scale = 256 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # CenterCrop 224
    left = (new_w - 224) // 2
    top  = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    arr = np.array(image, dtype=np.float32) / 255.0          # (224, 224, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]             # (1, 3, 224, 224)
    return arr


class Predictor:
    def __init__(self):
        self.class_names = _load_class_names()
        self.onnx_session = None
        self.torch_model = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self):
        if ONNX_MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.inter_op_num_threads = 4
                sess_opts.intra_op_num_threads = 4
                self.onnx_session = ort.InferenceSession(
                    str(ONNX_MODEL_PATH),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                print(f"[Predictor] ONNX model loaded: {ONNX_MODEL_PATH.name}")
                return
            except Exception as e:
                print(f"[Predictor] ONNX load failed ({e}), falling back to PyTorch")

        # PyTorch fallback
        import sys
        sys.path.insert(0, str(BASE_DIR))
        import torch
        from models.resnet50_extended import build_model
        from train.utils import load_checkpoint

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model().to(self.device)
        load_checkpoint(str(PYTORCH_CKPT), model)
        model.eval()
        self.torch_model = model
        print(f"[Predictor] PyTorch model loaded on {self.device}")

    @property
    def is_onnx(self) -> bool:
        return self.onnx_session is not None

    def predict(self, image: Image.Image) -> list[tuple[str, float]]:
        """
        Returns list of (class_name, confidence) sorted by confidence desc.
        Length = TOP_K (5).
        """
        arr = _preprocess(image)

        if self.onnx_session is not None:
            logits = self.onnx_session.run(["logits"], {"image": arr})[0][0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(arr).to(self.device)
                logits = self.torch_model(tensor).cpu().numpy()[0]

        # Softmax
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        top_indices = np.argsort(probs)[::-1][:TOP_K]
        return [(self.class_names[i], float(probs[i])) for i in top_indices]


# Module-level singleton — instantiated once when FastAPI starts up
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
