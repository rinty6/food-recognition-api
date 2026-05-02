"""
Phase 1 model: ResNet-18 fine-tuned for 5-class food recognition.

Training stages:
  Stage A (epochs 1 – head_epochs):
    Entire conv base frozen. Only the new classification head is trained.

  Stage B (epochs head_epochs+1 – total_epochs):
    layer4 (last residual block) unfrozen with a 10x lower learning rate.
    The classification head continues training at the base learning rate.

Use activate_stage_b() to trigger the unfreeze between stages.
"""

import torch.nn as nn
from torchvision import models


NUM_CLASSES = 5


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )

    return model


def activate_stage_b(model: nn.Module):
    """Unfreeze layer4 for discriminative fine-tuning in Stage B."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("Stage B: layer4 unfrozen.")


def get_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    """Returns optimizer param groups with 10x lower LR for layer4."""
    head_params = list(model.fc.parameters())
    layer4_params = list(model.layer4.parameters())

    groups = [{"params": head_params, "lr": base_lr}]
    if any(p.requires_grad for p in layer4_params):
        groups.append({"params": layer4_params, "lr": base_lr * 0.1})
    return groups
