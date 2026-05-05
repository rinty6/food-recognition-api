"""
Phase 3 model: ResNet-50 extended to 187 classes (101 Food-101 + 86 fruits/vegetables).

Architecture is identical to resnet50_food101.py — only the output size of the
final FC layer changes.  The Phase 2 backbone weights (layer1-4) are reused
via load_backbone_from_phase2(), which preserves the 85%+ accuracy already
achieved on the 101 existing classes while letting the new head learn the 86
new classes from scratch.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

BASE_DIR = Path(__file__).parent.parent


def _num_classes_from_json() -> int:
    path = BASE_DIR / "data" / "splits" / "phase3_classes.json"
    if path.exists():
        return len(json.loads(path.read_text()))
    return 187   # fallback if splits not yet generated


def build_model(num_classes: int | None = None) -> nn.Module:
    if num_classes is None:
        num_classes = _num_classes_from_json()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    return model


def load_backbone_from_phase2(model: nn.Module, checkpoint_path: str | Path) -> None:
    """
    Copies backbone weights (conv1, bn1, layer1-4) from a Phase 2 checkpoint
    into the extended model.  The FC head is intentionally left with its random
    initialisation so it can learn all 187 classes from scratch.

    Why this works: the backbone already knows how to extract food-relevant
    visual features (textures, shapes, colours) from 85%+ accuracy training.
    Only the classification head needs re-learning for the new class set.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    phase2_state = ckpt["model_state_dict"]

    backbone_state = {k: v for k, v in phase2_state.items() if not k.startswith("fc.")}
    missing, _ = model.load_state_dict(backbone_state, strict=False)

    non_fc_missing = [k for k in missing if not k.startswith("fc.")]
    if non_fc_missing:
        raise RuntimeError(f"Unexpected missing backbone weights: {non_fc_missing}")

    fc_count = len([k for k in missing if k.startswith("fc.")])
    print(f"  Backbone loaded   : {len(backbone_state)} tensors (Phase 2 weights)")
    print(f"  FC head           : {fc_count} tensors (freshly initialised)")
    print(f"  Phase 2 val_acc   : {ckpt.get('val_accuracy', 0.0):.4f}")


def activate_stage_b(model: nn.Module):
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("Stage B activated: layer4 unfrozen.")


def activate_stage_c(model: nn.Module):
    for param in model.layer3.parameters():
        param.requires_grad = True
    print("Stage C activated: layer3 + layer4 unfrozen.")


def get_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    """Discriminative learning rates — lower LR for deeper layers."""
    groups = [{"params": list(model.fc.parameters()), "lr": base_lr, "name": "head"}]

    layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
    if layer4_params:
        groups.append({"params": layer4_params, "lr": base_lr * 0.1, "name": "layer4"})

    layer3_params = [p for p in model.layer3.parameters() if p.requires_grad]
    if layer3_params:
        groups.append({"params": layer3_params, "lr": base_lr * 0.05, "name": "layer3"})

    return groups
