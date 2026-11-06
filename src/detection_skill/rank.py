"""RANK: face detections with attributes -> frames ranked for a thumbnail or a representative take
(`detection rank`, `detection-skill/face-rank@1`, ADR-007). A heuristic, documented as one; it picks nothing on its own.

Each face made by `detection faces --attributes` carries `size`, `frontal`, `sharpness` and `eyes_open` (faceattrs.py).
A face's score is a weighted sum of four parts in 0-1:

- eyes open (0.35): `eyes_open` as measured (a heuristic from edge directions in the eye regions);
- sharpness (0.25): sharpness / (sharpness + 100), so a blurred frame scores low;
- frontal (0.20): `frontal`, cos of the yaw estimated from the landmarks;
- size (0.20): min(1, size / 0.4), a face filling 40 % of the frame height or more scores 1.

A frame's score is its best face's (or, with `--all-faces`, the mean over its faces, so a group shot where one person
blinks scores lower). A part that could not be measured counts 0 and is named in the frame's `missing`. Frames with no
face are listed after the ranked ones with a null score. Ties keep time order.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .detections import ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now, validate_detections
from .errors import DetectionError

RANK_SCHEMA = "detection-skill/face-rank@1"
TOOL = "detection/rank"
WEIGHTS = {"eyes_open": 0.35, "sharpness": 0.25, "frontal": 0.20, "size": 0.20}
SHARP_HALF = 100.0
SIZE_FULL = 0.4
REQUIRED = ("schema", "source", "input", "method", "parameters", "frames", "counts", "skill", "skill_version", "tool",
            "created_at", "processing_seconds", "warnings")
NOTE = ("heuristic: a face scores by eyes open (edge directions in the eye regions), sharpness (Laplacian variance), "
        "frontal angle (landmarks) and size; a face is never an identity; the ranking suggests, a person picks")


def parts(attrs: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Optional[float]], List[str]]:
    a = attrs or {}
    out: Dict[str, Optional[float]] = {
        "eyes_open": float(a["eyes_open"]) if _num(a.get("eyes_open")) else None,
        "sharpness": a["sharpness"] / (a["sharpness"] + SHARP_HALF) if _num(a.get("sharpness")) else None,
        "frontal": float(a["frontal"]) if _num(a.get("frontal")) else None,
        "size": min(1.0, float(a["size"]) / SIZE_FULL) if _num(a.get("size")) else None,
    }
    return out, [k for k, v in out.items() if v is None]


def face_score(attrs: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Optional[float]], List[str]]:
    p, missing = parts(attrs)
    return sum(WEIGHTS[k] * (v or 0.0) for k, v in p.items()), p, missing


def make_rank(faces: Dict[str, Any], *, all_faces: bool = False, top: Optional[int] = None,
              file: Optional[str] = None, fingerprint: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    rep = validate_detections(faces)
    if not rep.ok:
        raise DetectionError("INVALID_DETECTIONS", "not a valid detections document", rep.errors[:10])
    if faces["kind"] != "face":
        raise DetectionError("INVALID_ARGUMENT", "the input must be face detections (detection faces --attributes)")
    if top is not None and (not isinstance(top, int) or top < 1):
        raise DetectionError("INVALID_ARGUMENT", "--top must be >= 1")
    dets = [d for fr in faces["frames"] for d in fr["detections"]]
    if dets and not any(isinstance(d.get("attributes"), dict) for d in dets):
        raise DetectionError("INVALID_ARGUMENT", "the faces carry no attributes: run detection faces --attributes")
    warnings: List[str] = []
    ranked: List[Dict[str, Any]] = []
    empty: List[Dict[str, Any]] = []
    for fr in faces["frames"]:
        scored = []
        for j, d in enumerate(fr["detections"]):
            s, p, missing = face_score(d.get("attributes"))
            scored.append({"face": j, "box": list(d["box"]), "score": round(s, 4),
                           "parts": {k: (round(v, 4) if v is not None else None) for k, v in p.items()}, "missing": missing,
                           "eyes_open": (d.get("attributes") or {}).get("eyes_open")})
        if not scored:
            empty.append({"index": fr["index"], "t": fr["t"], "score": None, "faces": [], "best_face": None, "rank": None})
            continue
        best = max(scored, key=lambda x: (x["score"], -x["face"]))
        score = sum(x["score"] for x in scored) / len(scored) if all_faces else best["score"]
        ranked.append({"index": fr["index"], "t": fr["t"], "score": round(score, 4), "faces": scored, "best_face": best["face"],
                       "rank": None})
    ranked.sort(key=lambda f: (-f["score"], f["index"]))
    for i, f in enumerate(ranked):
        f["rank"] = i + 1
    if not ranked:
        warnings.append("no face in any frame: nothing to rank")
    if any(x["missing"] for f in ranked for x in f["faces"]):
        warnings.append("some attributes could not be measured (no landmarks or a face at the edge): they count 0")
    frames = ranked + empty
    if top is not None:
        frames = ranked[:top]
    return {
        "schema": RANK_SCHEMA,
        "source": dict(faces["source"]),
        "input": {"file": os.path.basename(file) if file else None, "fingerprint": fingerprint, "tool": faces.get("tool"),
                  "frames": faces["counts"]["frames"], "detections": faces["counts"]["detections"],
                  "sampling": (faces.get("sampling") or {}).get("method")},
        "method": {"name": "face-quality", "heuristic": True, "identity": False, "note": NOTE},
        "parameters": {"weights": dict(WEIGHTS), "sharpness_half": SHARP_HALF, "size_full": SIZE_FULL, "all_faces": bool(all_faces),
                       "top": top},
        "frames": frames,
        "counts": {"frames": len(frames), "ranked": sum(1 for f in frames if f["rank"] is not None),
                   "without_face": sum(1 for f in frames if f["rank"] is None)},
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }


def validate_rank(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("rank document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != RANK_SCHEMA:
        rep.errors.append(f"schema must be {RANK_SCHEMA}")
    _check_run_fields(doc, rep, TOOL)
    check_source(doc["source"], rep)
    m = doc["method"]
    if not isinstance(m, dict) or m.get("heuristic") is not True or m.get("identity") is not False:
        rep.errors.append("method must declare heuristic: true and identity: false")
    frames = doc["frames"]
    if not isinstance(frames, list):
        rep.errors.append("frames must be a list")
        return rep
    prev: Optional[float] = None
    seen_unranked = False
    for i, f in enumerate(frames):
        k = f"frames[{i}]"
        if not isinstance(f, dict) or any(x not in f for x in ("index", "t", "score", "faces", "best_face", "rank")):
            rep.errors.append(f"{k}: needs index, t, score, faces, best_face, rank")
            continue
        if f["rank"] is None:
            seen_unranked = True
            if f["score"] is not None or f["faces"]:
                rep.errors.append(f"{k}: a frame without a face has score null and no faces")
            continue
        if seen_unranked:
            rep.errors.append(f"{k}: ranked frames come before the frames without a face")
        if f["rank"] != i + 1:
            rep.errors.append(f"{k}: rank must be {i + 1}")
        if not _num(f["score"]) or not 0 <= f["score"] <= 1:
            rep.errors.append(f"{k}: score must be in [0, 1]")
        elif prev is not None and f["score"] > prev + 1e-9:
            rep.errors.append(f"{k}: frames must be in descending score order")
        prev = f["score"] if _num(f["score"]) else prev
        if not isinstance(f["faces"], list) or not f["faces"]:
            rep.errors.append(f"{k}: a ranked frame lists its faces")
            continue
        for j, x in enumerate(f["faces"]):
            if not isinstance(x, dict) or not _num(x.get("score")) or not isinstance(x.get("parts"), dict) \
                    or any(p not in x["parts"] for p in WEIGHTS):
                rep.errors.append(f"{k}.faces[{j}]: needs score and parts {sorted(WEIGHTS)}")
                continue
            want = sum(WEIGHTS[p] * (x["parts"][p] or 0.0) for p in WEIGHTS)
            if abs(want - x["score"]) > 1e-3:
                rep.errors.append(f"{k}.faces[{j}]: score must be the weighted sum of its parts")
    c = doc["counts"]
    ranked = sum(1 for f in frames if isinstance(f, dict) and f.get("rank") is not None)
    if not isinstance(c, dict) or c.get("frames") != len(frames) or c.get("ranked") != ranked or c.get("without_face") != len(frames) - ranked:
        rep.errors.append("counts do not match")
    scan_leakage(doc, "", rep)
    return rep


__all__ = ["make_rank", "validate_rank", "face_score", "RANK_SCHEMA", "WEIGHTS"]
