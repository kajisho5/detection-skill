"""Measurements used by the tests and by `detection score`: box recall/precision at an IoU threshold, and the
character error rate of OCR text."""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Sequence

from .engines.boxes import iou


def match_boxes(truth: Sequence[Sequence[float]], found: Sequence[Sequence[float]], threshold: float = 0.5) -> Dict[str, Any]:
    """Greedy one-to-one matching by IoU (highest first). recall = matched / truth, precision = matched / found."""
    pairs = sorted(((iou(t, f), ti, fi) for ti, t in enumerate(truth) for fi, f in enumerate(found)), reverse=True)
    used_t, used_f = set(), set()
    matched: List[Dict[str, Any]] = []
    for v, ti, fi in pairs:
        if v < threshold:
            break
        if ti in used_t or fi in used_f:
            continue
        used_t.add(ti)
        used_f.add(fi)
        matched.append({"truth": ti, "found": fi, "iou": round(v, 4)})
    return {"truth": len(truth), "found": len(found), "matched": len(matched),
            "recall": len(matched) / len(truth) if truth else 1.0,
            "precision": len(matched) / len(found) if found else 1.0,
            "missed": sorted(set(range(len(truth))) - used_t), "pairs": matched}


def normalize_text(s: str) -> str:
    """NFKC, and whitespace removed: CER here counts characters, not spacing."""
    return "".join(unicodedata.normalize("NFKC", s).split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
