"""
Image transform pipelines for food recognition training.

train_transforms  : augmented pipeline — tuned for real-world robustness
val_transforms    : deterministic pipeline for validation and test (unchanged)

Key changes vs the original Phase 1/2 transforms:
  - saturation range raised 0.2 → 0.7  (yellow banana → overripe brown/black)
  - hue jitter added (±0.1)            (natural colour variation between shots)
  - RandomGrayscale(p=0.1)             (forces shape learning, not just colour)
  - RandomErasing(p=0.25)              (simulates occlusion, e.g. hand holding fruit)
  - crop scale lower bound 0.7 → 0.5   (more aggressive framing variation)
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.7,   # was 0.2 — now simulates overripe (yellow → brown → black)
        hue=0.1,
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.20)),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
