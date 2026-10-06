"""Capture quality checks and keyframe selection. Same code runs live (server) and offline (rerun)."""
import json

import cv2
import numpy as np

from pipeline import settings


def decode(jpeg: bytes):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def quality(img, prev_small=None):
    """Blur, glare, motion for one BGR frame. Deterministic: the LLM is never asked 'is this blurry'."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(cv2.resize(img, (640, int(640 * h / w))), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    q = {
        "blur": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "glare": round(float((gray >= 250).mean()), 4),
        "motion": round(float(np.abs(small - prev_small).mean()), 2) if prev_small is not None else 0.0,
    }
    q["issue"] = ("blur" if q["blur"] < settings.BLUR_MIN else
                  "glare" if q["glare"] > settings.GLARE_MAX else
                  "fast" if q["motion"] > settings.MOTION_MAX else "")
    return q, small


class KeyframeSelector:
    """Sharp, glare-free frames that show something new, at most 2/s, at least one per 2 s of usable video."""

    def __init__(self):
        self.last_small = None
        self.last_t = -1e9

    def offer(self, t, q, small):
        if q["issue"] or t - self.last_t < 0.5:
            return False
        novelty = 1e9 if self.last_small is None else float(np.abs(small - self.last_small).mean())
        if novelty < settings.NOVELTY_MIN and t - self.last_t < 2.0:
            return False
        self.last_small, self.last_t = small, t
        return True


def select_keyframes(sweep_dir):
    """Offline rerun: recompute quality + keyframes from saved frames."""
    rows = [json.loads(line) for line in (sweep_dir / "frames.jsonl").read_text().splitlines() if line]
    sel, prev, keys = KeyframeSelector(), None, []
    for r in rows:
        img = cv2.imread(str(sweep_dir / "frames" / r["file"]))
        if img is None:
            continue
        q, small = quality(img, prev)
        prev = small
        if sel.offer(r["t"], q, small):
            keys.append({"frame": r["file"], "t": r["t"], **q})
    return keys
