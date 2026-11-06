"""Face attributes measured on one frame (`detection faces --attributes`, ADR-005). Heuristics, documented as such.

All of them are measured from the face box and YuNet's five landmarks (right eye, left eye, nose tip, right and left
mouth corner, in that order; the subject's right eye is on the image's left for a face looking at the camera):

- `size`: box height / frame height (0-1). Scale-free, so the same face in a 720p and a 4K frame has the same size.
- `yaw_deg`, `roll_deg`, `frontal`: head pose from the landmarks only (no 3-D model). Roll is the angle of the eye
  line. Yaw is estimated from how far the nose tip sits from the eyes' midpoint along the eye line, in half eye
  distances: r = offset / (eye distance / 2), yaw = asin(clamp(r)). `frontal` = cos(yaw), 1 for a face looking
  straight at the camera, 0 for a profile. Pitch (nodding) is not estimated.
- `sharpness`: variance of the Laplacian of the face region resized to 96x96 gray. Higher is sharper; a blurred or
  out-of-focus face scores low. Comparable between faces, not an absolute focus measure.
- `mouth_contrast`: standard deviation of the gray levels in the mouth region (around the two mouth corners) / 128,
  clamped to 0-1. An open mouth shows the dark inside and the teeth, a closed one is mostly lips, so the value
  changes while someone talks; `detection speakers` reads its variation over time as mouth motion. One frame's value
  says nothing about speech.
- `eyes_open`: edge directions in the two eye regions (around the eye landmarks, 0.56 x 0.28 eye distances, resized
  to 40x20 gray): an open eye shows the iris and the eye corners, edges across the eye (horizontal gradients), a
  closed one mostly the lid line, an edge along it (vertical gradients). Per eye r = mean|dI/dx| / (mean|dI/dx| +
  mean|dI/dy|) and eyes_open = mean over the eyes of clamp((r - 0.25) / 0.2, 0, 1). A measured value, not a
  classifier; null when the head is rolled more than 25 degrees (the eye line is no longer horizontal) or an eye
  region leaves the frame. Glasses, heavy make-up, strong side light and small faces (eye distance under about 12 px)
  disturb it.

`pose` is plain Python (unit-testable without OpenCV); `measure` needs numpy and OpenCV and runs inside the engine.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

ATTRIBUTE_KEYS = ("size", "yaw_deg", "roll_deg", "frontal", "sharpness", "mouth_contrast", "eyes_open")
EYES_MAX_ROLL = 25.0
EYE_LOW, EYE_SPAN = 0.25, 0.2
FACE_SIDE = 96
DIGITS = 4


def pose(landmarks: Optional[Sequence[Sequence[float]]]) -> Dict[str, Optional[float]]:
    """yaw_deg, roll_deg, frontal from five landmarks; all None without them."""
    if not landmarks or len(landmarks) < 3:
        return {"yaw_deg": None, "roll_deg": None, "frontal": None}
    (rx, ry), (lx, ly), (nx, ny) = landmarks[0][:2], landmarks[1][:2], landmarks[2][:2]
    ex, ey = lx - rx, ly - ry
    d = math.hypot(ex, ey)
    if d < 1e-6:
        return {"yaw_deg": None, "roll_deg": None, "frontal": None}
    ux, uy = ex / d, ey / d
    mx, my = (rx + lx) / 2.0, (ry + ly) / 2.0
    r = ((nx - mx) * ux + (ny - my) * uy) / (d / 2.0)
    yaw = math.asin(max(-1.0, min(1.0, r)))
    return {"yaw_deg": round(math.degrees(yaw), 2), "roll_deg": round(math.degrees(math.atan2(ey, ex)), 2),
            "frontal": round(math.cos(yaw), DIGITS)}


def _clip_rect(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> Optional[List[int]]:
    a, b = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    c, d = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if c - a < 2 or d - b < 2:
        return None
    return [a, b, c, d]


def measure(gray: Any, box: Sequence[float], landmarks: Optional[Sequence[Sequence[float]]]) -> Dict[str, Optional[float]]:
    """Attributes of one face in a gray (uint8) frame; box [x, y, w, h] and landmarks in that frame's pixels."""
    import cv2
    import numpy as np
    h, w = gray.shape[:2]
    out: Dict[str, Optional[float]] = {k: None for k in ATTRIBUTE_KEYS}
    out["size"] = round(min(1.0, max(0.0, float(box[3]) / float(h))), DIGITS)
    out.update(pose(landmarks))
    r = _clip_rect(box[0], box[1], box[0] + box[2], box[1] + box[3], w, h)
    if r is not None:
        face = cv2.resize(gray[r[1]:r[3], r[0]:r[2]], (FACE_SIDE, FACE_SIDE), interpolation=cv2.INTER_AREA)
        out["sharpness"] = round(float(cv2.Laplacian(face, cv2.CV_64F).var()), 2)
    if landmarks and len(landmarks) >= 5:
        (ax, ay), (bx, by) = landmarks[3][:2], landmarks[4][:2]
        span = max(math.hypot(bx - ax, by - ay), 4.0)
        cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
        m = _clip_rect(cx - 0.75 * span, cy - 0.45 * span, cx + 0.75 * span, cy + 0.45 * span, w, h)
        if m is not None:
            patch = gray[m[1]:m[3], m[0]:m[2]].astype(np.float64)
            out["mouth_contrast"] = round(min(1.0, float(patch.std()) / 128.0), DIGITS)
    if landmarks and len(landmarks) >= 2 and out["roll_deg"] is not None and abs(out["roll_deg"]) <= EYES_MAX_ROLL:
        out["eyes_open"] = _eyes_open(cv2, np, gray, landmarks)
    return out


def eye_share(gx: float, gy: float) -> Optional[float]:
    """clamp((r - 0.25) / 0.2) for r = gx / (gx + gy), the share of edges across the eye; None without edges."""
    if gx + gy <= 1e-9:
        return None
    r = gx / (gx + gy)
    return min(1.0, max(0.0, (r - EYE_LOW) / EYE_SPAN))


def _eyes_open(cv2: Any, np: Any, gray: Any, landmarks: Sequence[Sequence[float]]) -> Optional[float]:
    h, w = gray.shape[:2]
    (rx, ry), (lx, ly) = landmarks[0][:2], landmarks[1][:2]
    d = math.hypot(lx - rx, ly - ry)
    vals = []
    for ex, ey in ((rx, ry), (lx, ly)):
        if ex - 0.28 * d < 0 or ey - 0.14 * d < 0 or ex + 0.28 * d > w or ey + 0.14 * d > h:
            return None
        r = _clip_rect(ex - 0.28 * d, ey - 0.14 * d, ex + 0.28 * d, ey + 0.14 * d, w, h)
        if r is None:
            return None
        p = cv2.resize(gray[r[1]:r[3], r[0]:r[2]], (40, 20), interpolation=cv2.INTER_AREA).astype(np.float64)
        v = eye_share(float(np.abs(cv2.Sobel(p, cv2.CV_64F, 1, 0)).mean()), float(np.abs(cv2.Sobel(p, cv2.CV_64F, 0, 1)).mean()))
        if v is None:
            return None
        vals.append(v)
    return round(sum(vals) / len(vals), DIGITS)


__all__ = ["ATTRIBUTE_KEYS", "pose", "measure", "eye_share"]
