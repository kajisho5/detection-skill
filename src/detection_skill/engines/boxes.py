"""Box arithmetic in plain Python ([x, y, w, h], top-left origin, pixels). Shared by engines, tracks and metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

Box = Sequence[float]


def iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(dets: List[Dict[str, Any]], threshold: float, by_label: bool = False) -> List[Dict[str, Any]]:
    """Greedy non-maximum suppression, highest score first; with by_label only boxes of one label suppress each other."""
    kept: List[Dict[str, Any]] = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["box"], k["box"]) <= threshold or (by_label and k.get("label") != d.get("label")) for k in kept):
            kept.append(d)
    return kept


def union(boxes: Sequence[Box]) -> List[float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def clip(box: Box, width: float, height: float) -> List[float]:
    x1, y1 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x2, y2 = min(float(width), float(box[0] + box[2])), min(float(height), float(box[1] + box[3]))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
