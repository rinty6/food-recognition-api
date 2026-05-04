"""
Persists user feedback (wrong-prediction corrections) to disk.

Storage layout under FEEDBACK_DIR (default /data/feedback, fallback ./feedback):
  images/  — one JPEG per submission, named by timestamp + predicted class
  log.jsonl — one JSON record per line: timestamp, predicted, correct, image_path

When deployed on Railway, mount a volume at /data so images survive redeploys.
Without a volume, /data is writable but ephemeral (lost on redeploy).
"""

import json
import os
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", "/data/feedback"))
_IMAGES_DIR = _FEEDBACK_DIR / "images"
_LOG_FILE = _FEEDBACK_DIR / "log.jsonl"


def _ensure_dirs() -> bool:
    try:
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.warning(f"Could not create feedback directory {_IMAGES_DIR}: {e}")
        return False


def save_feedback(predicted_class: str, correct_class: str, image_bytes: bytes | None) -> None:
    if not _ensure_dirs():
        return

    ts = int(time.time() * 1000)
    image_path: str | None = None

    if image_bytes:
        filename = f"{ts}_{predicted_class}.jpg"
        dest = _IMAGES_DIR / filename
        try:
            dest.write_bytes(image_bytes)
            image_path = str(dest)
        except OSError as e:
            logger.warning(f"Could not save feedback image: {e}")

    record = {
        "timestamp": ts,
        "predicted_class": predicted_class,
        "correct_class": correct_class,
        "image_path": image_path,
    }

    try:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Feedback saved: {predicted_class!r} → {correct_class!r}")
    except OSError as e:
        logger.warning(f"Could not write feedback log: {e}")
