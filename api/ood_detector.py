"""
Out-of-Distribution (OOD) detection via softmax confidence thresholding.

If the model's top-1 confidence is below the threshold, the image is likely
not a recognised food class. The threshold was calibrated on Phase 2/3 results
where well-predicted classes consistently score above 0.75.
"""

import numpy as np

CONFIDENCE_THRESHOLD = 0.75

# Classes that are visually ambiguous with each other — lower the threshold
# so alternatives are surfaced rather than forcing a single answer.
AMBIGUOUS_CLUSTERS = [
    # Food-101 meat cuts
    {"steak", "pork_chop", "filet_mignon", "prime_rib"},
    # Citrus fruits look nearly identical
    {"orange", "mandarine", "clementine", "tangerine", "grapefruit", "pomelo"},
    # Lemon and lime confusion
    {"lemon", "limes"},
    # Apple and pear can look similar when whole
    {"apple", "pear"},
    # Eggplant / brinjal are the same vegetable under two names
    {"eggplant", "brinjal"},
    # Stone fruits
    {"peach", "apricot", "nectarine"},
    # Dark berries
    {"blackberry", "mulberry", "blueberry"},
]
AMBIGUOUS_THRESHOLD = 0.60


def _class_ambiguous(class_name: str) -> bool:
    return any(class_name in cluster for cluster in AMBIGUOUS_CLUSTERS)


def detect_ood(class_name: str, confidence: float) -> tuple[bool, str | None]:
    """
    Returns (is_ood, message).

    is_ood=True means the image should be flagged as uncertain.
    The caller still returns the prediction but includes the warning.
    """
    threshold = AMBIGUOUS_THRESHOLD if _class_ambiguous(class_name) else CONFIDENCE_THRESHOLD

    if confidence < threshold:
        msg = (
            f"Low confidence ({confidence:.0%}). "
            "The image may not be a recognized food item, or the food "
            "could not be clearly identified. Please try a clearer photo."
        )
        return True, msg

    return False, None


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()
