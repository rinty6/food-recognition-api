"""
PyTorch Dataset for Phase 1 food recognition.

Loads images from the JSON split files produced by data/split_dataset.py.
Each record has {"path": str, "label": int, "class_name": str}.
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self, split_json_path: str | Path, transform=None):
        records = json.loads(Path(split_json_path).read_text())
        self.samples = [(r["path"], r["label"]) for r in records]
        self.classes = sorted({r["class_name"] for r in records})
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
