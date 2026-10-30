"""TRACKS: detections document -> tracks document (`detection-skill/tracks@1`) with redact-ready rectangles.

A heuristic, documented as one: boxes of consecutive sampled frames are linked greedily by IoU (highest first,
at least `min_iou`, same label for objects, no more than `max_gap` seconds apart); when consecutive boxes do not
overlap (motion faster than the sampling), a box whose centre moved at most `max_shift` box sizes still links.
A track id says "these boxes followed each other from one sample to the next", nothing more. It is not an identity: two people who swap places, or a face
that leaves and comes back, can share an id or get two. Between samples nothing was measured.

Each track carries `redact` rectangles for ffmpeg-skill redact: the union of the track's boxes (per `window`
seconds, or over the whole track), padded by `pad` of its size on each side, snapped outward to even pixels
(redact needs even width/height) and clipped to the frame, over [first sample - extend, last sample + extend].
`extend` defaults to one sampling step, because the object may already be there just before the first sample
that saw it and still there just after the last.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

from . import __version__
from .detections import (ISO_RE, KINDS, ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now,
                         validate_detections)
from .engines.boxes import iou, union
from .errors import DetectionError

TRACKS_SCHEMA = "detection-skill/tracks@1"
METHOD = "greedy-iou"
TRACK_ID_RE = re.compile(r"^track_\d{3,}$")
DEFAULTS = {"min_iou": 0.3, "max_shift": 1.0, "pad": 0.15, "min_hits": 1, "window": 0.0}
REQUIRED = ("schema", "kind", "source", "frame_size", "detections", "method", "parameters", "tracks", "counts", "skill",
            "skill_version", "tool", "created_at", "processing_seconds", "warnings")
NOTE = ("heuristic: boxes linked over time by IoU, or by centre distance when they do not overlap; a track id is not an identity, and nothing was measured between "
        "samples (redact spans are extended by `extend` on each side to cover that)")


def sampling_step(doc: Dict[str, Any]) -> float:
    sm = doc.get("sampling") or {}
    if sm.get("method") == "every" and _num(sm.get("every")):
        return float(sm["every"])
    times = [f["t"] for f in doc.get("frames", []) if _num(f.get("t"))]
    diffs = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    if not diffs:
        return 0.0
    return diffs[len(diffs) // 2]


def link(doc: Dict[str, Any], *, min_iou: float, max_gap: float, by_label: bool,
         max_shift: float = 0.0) -> List[Dict[str, Any]]:
    """Greedy linking, frame by frame in time order. A box may join a live track when it overlaps the track's last
    box by IoU >= min_iou, or (max_shift > 0) when its centre lies within max_shift x the last box's size
    (sqrt(w*h)) of the last box's centre and its area is within 4x of it -- the fallback for motion faster than
    the sampling, where consecutive boxes no longer overlap. Highest IoU first, then smallest shift.
    Returns raw tracks [{label, boxes: [...]}]."""
    tracks: List[Dict[str, Any]] = []
    for fr in doc["frames"]:
        t = fr["t"]
        live = [tr for tr in tracks if t - tr["boxes"][-1]["t"] <= max_gap + 1e-9]
        pairs = []
        for di, d in enumerate(fr["detections"]):
            for ti, tr in enumerate(live):
                if by_label and tr["label"] != d["label"]:
                    continue
                last = tr["boxes"][-1]["box"]
                v = iou(last, d["box"])
                shift = _shift(last, d["box"])
                if v >= min_iou or (max_shift > 0 and shift <= max_shift and _area_ratio(last, d["box"]) <= 4.0):
                    pairs.append((v, shift, -tr["boxes"][-1]["t"], ti, di))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2], p[3], p[4]))
        used_t, used_d = set(), set()
        for _v, _s, _neg, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            live[ti]["boxes"].append(_entry(fr["detections"][di], t))
        for di, d in enumerate(fr["detections"]):
            if di not in used_d:
                tracks.append({"label": d["label"], "boxes": [_entry(d, t)]})
    return tracks


def _shift(a: List[float], b: List[float]) -> float:
    """Distance between box centres in units of a's size (sqrt of its area)."""
    size = math.sqrt(max(a[2] * a[3], 1e-9))
    return math.hypot((b[0] + b[2] / 2) - (a[0] + a[2] / 2), (b[1] + b[3] / 2) - (a[1] + a[3] / 2)) / size


def _area_ratio(a: List[float], b: List[float]) -> float:
    aa, bb = max(a[2] * a[3], 1e-9), max(b[2] * b[3], 1e-9)
    return max(aa, bb) / min(aa, bb)


def _entry(d: Dict[str, Any], t: float) -> Dict[str, Any]:
    e = {"t": t, "box": list(d["box"]), "score": d["score"]}
    if d.get("label") is not None:
        e["label"] = d["label"]
    return e


def redact_rect(boxes: List[List[float]], pad: float, width: int, height: int) -> Dict[str, int]:
    x, y, w, h = union(boxes)
    px, py = pad * w, pad * h
    x1 = max(0, int(math.floor((x - px) / 2.0)) * 2)
    y1 = max(0, int(math.floor((y - py) / 2.0)) * 2)
    max_x, max_y = width - width % 2, height - height % 2
    x2 = min(max_x, int(math.ceil((x + w + px) / 2.0)) * 2)
    y2 = min(max_y, int(math.ceil((y + h + py) / 2.0)) * 2)
    if x2 - x1 < 2:
        x1 = max(0, min(x1, max_x - 2))
        x2 = x1 + 2
    if y2 - y1 < 2:
        y1 = max(0, min(y1, max_y - 2))
        y2 = y1 + 2
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def make_tracks(doc: Dict[str, Any], *, min_iou: Optional[float] = None, max_gap: Optional[float] = None,
                pad: Optional[float] = None, extend: Optional[float] = None, window: Optional[float] = None,
                min_hits: Optional[int] = None, max_shift: Optional[float] = None, detections_file: Optional[str] = None,
                detections_fingerprint: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    rep = validate_detections(doc)
    if not rep.ok:
        raise DetectionError("INVALID_DETECTIONS", "not a valid detections document", rep.errors[:10])
    if doc["source"]["media_type"] != "video":
        raise DetectionError("INVALID_ARGUMENT", "tracks need a video's detections (an image has no time)")
    step = sampling_step(doc)
    min_iou = DEFAULTS["min_iou"] if min_iou is None else float(min_iou)
    max_shift = DEFAULTS["max_shift"] if max_shift is None else float(max_shift)
    pad = DEFAULTS["pad"] if pad is None else float(pad)
    window = DEFAULTS["window"] if window is None else float(window)
    min_hits = int(DEFAULTS["min_hits"] if min_hits is None else min_hits)
    max_gap = max(2.5 * step, 0.5) if max_gap is None else float(max_gap)
    extend = step if extend is None else float(extend)
    if not 0.0 < min_iou <= 1.0:
        raise DetectionError("INVALID_ARGUMENT", "--min-iou must be in (0, 1]")
    for name, v in (("--max-shift", max_shift), ("--pad", pad), ("--window", window), ("--max-gap", max_gap), ("--extend", extend)):
        if not math.isfinite(v) or v < 0:
            raise DetectionError("INVALID_ARGUMENT", f"{name} must be >= 0")
    if min_hits < 1:
        raise DetectionError("INVALID_ARGUMENT", "--min-hits must be >= 1")
    kind = doc["kind"]
    width, height = doc["frame_size"]["width"], doc["frame_size"]["height"]
    duration = float(doc["source"]["duration"])
    raw = link(doc, min_iou=min_iou, max_gap=max_gap, by_label=(kind == "object"), max_shift=max_shift)
    out: List[Dict[str, Any]] = []
    dropped = 0
    for tr in raw:
        if len(tr["boxes"]) < min_hits:
            dropped += len(tr["boxes"])
            continue
        boxes = tr["boxes"]
        start, end = boxes[0]["t"], boxes[-1]["t"]
        groups: List[List[Dict[str, Any]]] = []
        for b in boxes:
            if window > 0 and groups and b["t"] - groups[-1][0]["t"] < window:
                groups[-1].append(b)
            elif window > 0 or not groups:
                groups.append([b])
            else:
                groups[-1].append(b)
        redact = []
        for g in groups:
            rect = redact_rect([b["box"] for b in g], pad, width, height)
            redact.append(dict(start=round(max(0.0, g[0]["t"] - extend), 3), end=round(min(duration, g[-1]["t"] + extend), 3),
                               **rect))
        label = tr["label"]
        if kind == "text":
            texts = [b.get("label") for b in boxes if b.get("label")]
            label = max(sorted(set(texts)), key=texts.count) if texts else None
        scores = [b["score"] for b in boxes]
        out.append({"id": f"track_{len(out) + 1:03d}", "label": None if kind == "face" else label, "start": start, "end": end,
                    "hits": len(boxes), "mean_score": round(sum(scores) / len(scores), 4), "max_score": max(scores),
                    "boxes": [{k: v for k, v in b.items() if k != "label"} for b in boxes], "redact": redact})
    warnings: List[str] = []
    if step and max_gap < step:
        warnings.append(f"max_gap {max_gap:g}s is shorter than the sampling step {step:g}s: no box can be linked")
    if not out:
        warnings.append("no tracks")
    return {
        "schema": TRACKS_SCHEMA,
        "kind": kind,
        "source": dict(doc["source"]),
        "frame_size": dict(doc["frame_size"]),
        "detections": {"file": os.path.basename(detections_file) if detections_file else None,
                       "fingerprint": detections_fingerprint, "tool": doc["tool"], "engine": doc["engine"]["id"],
                       "model": doc["engine"]["model"], "model_sha256": doc["engine"]["model_sha256"],
                       "sampling_method": doc["sampling"]["method"], "sampling_step": step},
        "method": {"name": METHOD, "heuristic": True, "identity": False, "note": NOTE},
        "parameters": {"min_iou": min_iou, "max_shift": max_shift, "max_gap": round(max_gap, 3), "pad": pad, "extend": round(extend, 3),
                       "window": window, "min_hits": min_hits, "same_label": kind == "object"},
        "tracks": out,
        "counts": {"tracks": len(out), "boxes_linked": sum(t["hits"] for t in out), "boxes_dropped": dropped,
                   "redact_rects": sum(len(t["redact"]) for t in out)},
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": "detection/tracks",
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }


def fingerprint_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def validate_tracks(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("tracks document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != TRACKS_SCHEMA:
        rep.errors.append(f"schema must be {TRACKS_SCHEMA}")
    if doc["kind"] not in KINDS:
        rep.errors.append(f"kind must be one of {KINDS}")
    _check_run_fields(doc, rep, "detection/tracks")
    check_source(doc["source"], rep)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    if src.get("media_type") != "video":
        rep.errors.append("tracks are made from a video's detections")
    fs = doc["frame_size"]
    if not isinstance(fs, dict) or (fs.get("width"), fs.get("height")) != (src.get("width"), src.get("height")):
        rep.errors.append("frame_size must equal source width/height")
        return rep
    width, height = fs["width"], fs["height"]
    duration = src.get("duration") if _num(src.get("duration")) else None
    m = doc["method"]
    if not isinstance(m, dict) or m.get("heuristic") is not True or m.get("identity") is not False:
        rep.errors.append("method must declare heuristic: true and identity: false")
    if not isinstance(doc["tracks"], list):
        rep.errors.append("tracks must be a list")
        return rep
    seen = set()
    linked = rects = 0
    for i, tr in enumerate(doc["tracks"]):
        w = f"tracks[{i}]"
        if not isinstance(tr, dict) or any(k not in tr for k in ("id", "label", "start", "end", "hits", "boxes", "redact")):
            rep.errors.append(f"{w}: needs id, label, start, end, hits, boxes, redact")
            continue
        if not isinstance(tr["id"], str) or not TRACK_ID_RE.match(tr["id"]) or tr["id"] in seen:
            rep.errors.append(f"{w}: id must be a unique track_nnn")
        seen.add(tr["id"])
        if doc["kind"] == "face" and tr["label"] is not None:
            rep.errors.append(f"{w}: a face track has label null (a track id is not an identity)")
        bx = tr["boxes"]
        if not isinstance(bx, list) or not bx or len(bx) != tr["hits"]:
            rep.errors.append(f"{w}: boxes must be a non-empty list of hits entries")
            continue
        linked += len(bx)
        ts = [b.get("t") for b in bx if isinstance(b, dict)]
        if len(ts) != len(bx) or not all(_num(t) for t in ts) or ts != sorted(set(ts)):
            rep.errors.append(f"{w}: boxes must have strictly increasing t")
        elif (tr["start"], tr["end"]) != (ts[0], ts[-1]):
            rep.errors.append(f"{w}: start/end must be the first/last box time")
        for j, b in enumerate(bx):
            box = b.get("box") if isinstance(b, dict) else None
            if not (isinstance(box, list) and len(box) == 4 and all(_num(v) for v in box)) or box[2] <= 0 or box[3] <= 0 \
                    or box[0] < 0 or box[1] < 0 or box[0] + box[2] > width + 0.5 or box[1] + box[3] > height + 0.5:
                rep.errors.append(f"{w}.boxes[{j}]: box must be [x, y, w, h] inside the frame")
        if not isinstance(tr["redact"], list) or not tr["redact"]:
            rep.errors.append(f"{w}: redact must be a non-empty list")
            continue
        for j, r in enumerate(tr["redact"]):
            rects += 1
            ok = isinstance(r, dict) and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool)
                                             for k in ("x", "y", "width", "height"))
            if not ok or r["width"] < 2 or r["height"] < 2 or r["width"] % 2 or r["height"] % 2 or r["x"] < 0 or r["y"] < 0 \
                    or r["x"] + r["width"] > width or r["y"] + r["height"] > height:
                rep.errors.append(f"{w}.redact[{j}]: needs even integer width/height >= 2 inside the frame")
            if not isinstance(r, dict) or not _num(r.get("start")) or not _num(r.get("end")) or r["end"] < r["start"] \
                    or r["start"] < 0 or (duration is not None and r["end"] > duration + 0.05):
                rep.errors.append(f"{w}.redact[{j}]: needs 0 <= start <= end <= duration")
    c = doc["counts"]
    if not isinstance(c, dict) or c.get("tracks") != len(doc["tracks"]) or c.get("boxes_linked") != linked \
            or c.get("redact_rects") != rects:
        rep.errors.append("counts do not match tracks")
    scan_leakage(doc, "", rep)
    return rep


__all__ = ["make_tracks", "validate_tracks", "TRACKS_SCHEMA", "ISO_RE", "fingerprint_bytes"]
