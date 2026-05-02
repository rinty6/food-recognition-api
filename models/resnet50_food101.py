"""
Phase 2 model: ResNet-50 fine-tuned for 101-class Food-101 recognition.

Three training stages, activated by the training script:
  Stage A (epochs  1–10): entire conv base frozen, head only
  Stage B (epochs 11–25): layer4 unfrozen, discriminative LR
  Stage C (epochs 26–50): layer3+layer4 unfrozen, label smoothing added

Call activate_stage_b() and activate_stage_c() at the right epoch boundaries.
"""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 101


def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES),
    )

    return model


def activate_stage_b(model: nn.Module):
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("Stage B activated: layer4 unfrozen.")


def activate_stage_c(model: nn.Module):
    for param in model.layer3.parameters():
        param.requires_grad = True
    print("Stage C activated: layer3 + layer4 unfrozen.")


def get_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    """
    Returns optimizer param groups.
    layer3/layer4 use 10x lower LR than the head to avoid destroying
    the pretrained features during fine-tuning.
    """
    groups = [{"params": list(model.fc.parameters()), "lr": base_lr, "name": "head"}]

    layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
    if layer4_params:
        groups.append({"params": layer4_params, "lr": base_lr * 0.1, "name": "layer4"})

    layer3_params = [p for p in model.layer3.parameters() if p.requires_grad]
    if layer3_params:
        groups.append({"params": layer3_params, "lr": base_lr * 0.05, "name": "layer3"})

    return groups
