"""
Out-of-Distribution (OOD) detection via softmax confidence thresholding.

If the model's top-1 confidence is below the threshold, the image is likely
not a Food-101 class (e.g., a car, a landscape, or an abstract food photo).
The threshold was chosen based on Phase 2 test results where well-predicted
classes consistently score above 0.75.
"""

import numpy as np

CONFIDENCE_THRESHOLD = 0.75

# Classes that are visually ambiguous with each other — lower the bar
# slightly so we surface alternatives rather than forcing a single answer.
AMBIGUOUS_CLUSTERS = [
    {"steak", "pork_chop", "filet_mignon", "prime_rib"},
]
AMBIGUOUS_THRESHOLD = 0.60


def _class_ambiguous(class_name: str) -> bool:
    return any(class_name in cluster for cluster in AMBIGUOUS_CLUSTERS)


def detect_ood(class_name: str, confidence: float) -> tuple[bool, str | None]:
    """
    Returns (is_ood, message).

    is_ood=True means the image should be flagged as uncertain / out-of-distribution.
    The caller should still return the prediction but include the warning message.
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
