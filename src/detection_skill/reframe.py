"""REFRAME: face tracks (+ object tracks when asked, + shots) -> a subject path (`detection-skill/reframe@1`): one crop
window per sample time for a target aspect (9:16, 1:1, 4:5, ...), ADR-004. A heuristic, documented as one.

- **Window.** The largest window of the target aspect that fits the frame (full height from a wider source, full
  width from a taller one), even pixels, the same size all through. No zoom.
- **Subject.** Per shot (the whole video when no shots document is given), the face track with the most presence in
  the rest of the shot: the sum over its boxes of sqrt(box area), so a large face seen for long wins. With
  `objects`, an object track (default class person) is the fallback when no face is in view. The subject is kept
  until its track ends; then the next one present is taken. Every change is in `switches` with its reason.
  A track id is not an identity: "the subject" is a box track, not a person.
- **Smoothing.** Inside a shot the window centre follows the subject's box centre (interpolated between samples)
  with a dead zone (no move while the centre is within `dead_zone` x window width of the target) and a speed limit
  (at most `max_speed` x window width per second), so it never jumps inside a shot. At a shot boundary from
  `detection shots` the window is placed on the new subject directly (a cut is allowed there). The window is kept
  inside the frame.
- **One window per shot.** ffmpeg-skill has no time-varying crop yet, so `shots[]` also gives one fixed window per
  shot (centred on the middle of the subject's range of centres in that shot) with the fraction of samples whose
  subject box it contains. Applying a window is video-editing's and the engine's job, not this skill's.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .detections import ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now
from .errors import DetectionError
from .frames import plan_every
from .shots import shot_spans, validate_shots
from .tracks import TRACKS_SCHEMA, validate_tracks

REFRAME_SCHEMA = "detection-skill/reframe@1"
TOOL = "detection/reframe"
METHOD = "subject-path"
DEFAULTS = {"dead_zone": 0.1, "max_speed": 0.6}
DEFAULT_OBJECT_CLASSES = ("person",)
ASPECT_RE = re.compile(r"^\s*(\d{1,3})\s*[:x/]\s*(\d{1,3})\s*$")
SUBJECT_RE = re.compile(r"^(face|object)/track_\d{3,}$")
REQUIRED = ("schema", "source", "frame_size", "aspect", "window", "inputs", "method", "parameters", "windows", "switches",
            "shots", "counts", "skill", "skill_version", "tool", "created_at", "processing_seconds", "warnings")
NOTE = ("heuristic: the subject is the face (or object) box track with the most presence in the shot; the window follows "
        "its centre with a dead zone and a speed limit inside a shot and may jump only at a shot boundary; a track id is "
        "not an identity")
SPEED_SLACK_PX = 2  # even-pixel rounding


def parse_aspect(text: str) -> Tuple[str, float]:
    m = ASPECT_RE.match(str(text or ""))
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        raise DetectionError("INVALID_ARGUMENT", f"--aspect {text!r}: expected W:H such as 9:16, 1:1 or 4:5")
    w, h = int(m.group(1)), int(m.group(2))
    ratio = w / float(h)
    if not 0.1 <= ratio <= 10:
        raise DetectionError("INVALID_ARGUMENT", f"--aspect {text}: ratio out of range")
    return f"{w}:{h}", ratio


def window_size(width: int, height: int, ratio: float) -> Tuple[int, int]:
    """The largest even-pixel window of `ratio` (w / h) inside width x height."""
    ch = height - height % 2
    cw = int(math.floor(ch * ratio / 2.0)) * 2
    if cw > width:
        cw = width - width % 2
        ch = int(math.floor(cw / ratio / 2.0)) * 2
    return max(2, cw), max(2, ch)


def _interp(track: Dict[str, Any], t: float) -> Optional[List[float]]:
    """The track's box at time t, linear between its samples; None outside [start, end]."""
    bx = track["boxes"]
    if t < bx[0]["t"] - 1e-9 or t > bx[-1]["t"] + 1e-9:
        return None
    for a, b in zip(bx, bx[1:]):
        if a["t"] - 1e-9 <= t <= b["t"] + 1e-9:
            span = b["t"] - a["t"]
            u = 0.0 if span <= 0 else (t - a["t"]) / span
            return [a["box"][i] + u * (b["box"][i] - a["box"][i]) for i in range(4)]
    return list(bx[0]["box"])


def _presence(track: Dict[str, Any], t0: float, t1: float) -> float:
    return sum(math.sqrt(max(b["box"][2] * b["box"][3], 0.0)) for b in track["boxes"] if t0 - 1e-9 <= b["t"] < t1 - 1e-9)


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def _place(cx: float, cy: float, cw: int, ch: int, width: int, height: int) -> Tuple[int, int, float, float]:
    """Window top-left (even, inside the frame) for a centre; returns (x, y, clamped cx, clamped cy)."""
    max_x, max_y = width - cw, height - ch
    max_x -= max_x % 2
    max_y -= max_y % 2
    x = min(max(_even(cx - cw / 2.0), 0), max(0, max_x))
    y = min(max(_even(cy - ch / 2.0), 0), max(0, max_y))
    return x, y, x + cw / 2.0, y + ch / 2.0


def _inside(box: List[float], x: int, y: int, cw: int, ch: int) -> bool:
    return box[0] >= x - 0.5 and box[1] >= y - 0.5 and box[0] + box[2] <= x + cw + 0.5 and box[1] + box[3] <= y + ch + 0.5


def _input_ref(doc: Dict[str, Any], file: Optional[str], fingerprint: Optional[str]) -> Dict[str, Any]:
    ref: Dict[str, Any] = {"file": os.path.basename(file) if file else None, "fingerprint": fingerprint, "tool": doc.get("tool")}
    if doc.get("schema") == TRACKS_SCHEMA:
        ref.update(kind=doc["kind"], tracks=len(doc["tracks"]), sampling_step=doc["detections"]["sampling_step"])
    else:
        ref.update(engine=doc["engine"]["id"], shots=len(doc["shots"]))
    return ref


def make_reframe(faces: Dict[str, Any], *, aspect: str = "9:16", objects: Optional[Dict[str, Any]] = None,
                 shots: Optional[Dict[str, Any]] = None, object_classes: Optional[List[str]] = None,
                 dead_zone: Optional[float] = None, max_speed: Optional[float] = None,
                 files: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None) -> Dict[str, Any]:
    """faces: a face tracks document. objects: an object tracks document (optional). shots: a shots document
    (optional). files: {"faces"|"objects"|"shots": (file name, fingerprint)} for provenance."""
    t0 = time.time()
    files = files or {}
    rep = validate_tracks(faces)
    if not rep.ok:
        raise DetectionError("INVALID_DETECTIONS", "faces: not a valid tracks document", rep.errors[:10])
    if faces["kind"] != "face":
        raise DetectionError("INVALID_ARGUMENT", "the first input must be face tracks (detection faces -> detection tracks)")
    fp = faces["source"]["fingerprint"]
    if objects is not None:
        rep = validate_tracks(objects)
        if not rep.ok:
            raise DetectionError("INVALID_DETECTIONS", "--objects: not a valid tracks document", rep.errors[:10])
        if objects["kind"] != "object":
            raise DetectionError("INVALID_ARGUMENT", "--objects must be object tracks (detection objects -> detection tracks)")
        if objects["source"]["fingerprint"] != fp:
            raise DetectionError("INVALID_ARGUMENT", "--objects was measured on another file (source fingerprints differ)")
    if shots is not None:
        rep = validate_shots(shots)
        if not rep.ok:
            raise DetectionError("INVALID_DETECTIONS", "--shots: not a valid shots document", rep.errors[:10])
        if shots["source"]["fingerprint"] != fp:
            raise DetectionError("INVALID_ARGUMENT", "--shots was measured on another file (source fingerprints differ)")
    name, ratio = parse_aspect(aspect)
    dead_zone = DEFAULTS["dead_zone"] if dead_zone is None else float(dead_zone)
    max_speed = DEFAULTS["max_speed"] if max_speed is None else float(max_speed)
    if not math.isfinite(dead_zone) or not 0 <= dead_zone < 0.5:
        raise DetectionError("INVALID_ARGUMENT", "--dead-zone must be in [0, 0.5) (fraction of the window width)")
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise DetectionError("INVALID_ARGUMENT", "--max-speed must be > 0 (window widths per second)")
    classes = [c.strip().lower() for c in (object_classes or DEFAULT_OBJECT_CLASSES) if c and c.strip()]

    width, height = faces["frame_size"]["width"], faces["frame_size"]["height"]
    duration = float(faces["source"]["duration"])
    cw, ch = window_size(width, height, ratio)
    dz_px = dead_zone * cw
    v_px = max_speed * cw

    cands: List[Tuple[str, Dict[str, Any]]] = [("face/" + tr["id"], tr) for tr in faces["tracks"]]
    obj_cands: List[Tuple[str, Dict[str, Any]]] = []
    if objects is not None:
        obj_cands = [("object/" + tr["id"], tr) for tr in objects["tracks"] if (tr.get("label") or "").lower() in classes]

    times = _sample_times(faces, objects, duration)
    spans = shot_spans(shots) if shots is not None else [{"index": 0, "id": "shot_001", "start": 0.0, "end": duration,
                                                          "starts_with": "start"}]
    warnings: List[str] = []
    windows: List[Dict[str, Any]] = []
    switches: List[Dict[str, Any]] = []
    per_shot: List[Dict[str, Any]] = []
    carried: Optional[str] = None  # the previous shot's last subject, for the switch record at a boundary
    for sp in spans:
        st = [t for t in times if sp["start"] - 1e-9 <= t < sp["end"] - 1e-9]
        if not st:
            continue
        subject: Optional[str] = None
        cur: Optional[Tuple[float, float]] = None
        prev_t: Optional[float] = None
        shot_windows: List[Dict[str, Any]] = []
        for t in st:
            box = _box_of(subject, cands, obj_cands, t)
            reason = None
            if subject is None or box is None:
                new, new_box = _choose(cands, obj_cands, t, sp["end"])
                was = subject if prev_t is not None else carried
                if new != subject or (prev_t is None and new != carried):
                    if prev_t is None:
                        reason = ("first shot: " if sp["index"] == 0 else f"shot boundary ({sp['starts_with']}) at {sp['start']:.3f}s: ")
                    elif subject is not None:
                        reason = f"{subject} not seen after {prev_t:.3f}s: "
                    else:
                        reason = "a subject came into view: "
                    reason += (f"{new} has the most presence in the rest of the shot" if new else
                               "no face" + (" or object" if objects is not None else "") + " in view: the window holds")
                    if new and new.startswith("object/"):
                        reason += " (no face in view)"
                    switches.append({"t": t, "shot": sp["index"], "from": was, "to": new, "reason": reason})
                subject, box = new, new_box
            if box is not None:
                target = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
            else:
                target = cur if cur is not None else (width / 2.0, height / 2.0)
            if cur is None:
                cur = target
            else:
                dt = max(0.0, t - (prev_t if prev_t is not None else t))
                cur = (_follow(cur[0], target[0], dz_px, v_px * dt), _follow(cur[1], target[1], dz_px, v_px * dt))
            x, y, cx, cy = _place(cur[0], cur[1], cw, ch, width, height)
            cur = (cx, cy)
            w_doc = {"t": t, "shot": sp["index"], "x": x, "y": y, "width": cw, "height": ch, "subject": subject,
                     "subject_box": [round(v, 2) for v in box] if box is not None else None,
                     "subject_inside": _inside(box, x, y, cw, ch) if box is not None else None}
            windows.append(w_doc)
            shot_windows.append(w_doc)
            prev_t = t
        per_shot.append(_shot_window(sp, shot_windows, cw, ch, width, height))
        carried = subject
    outside = sum(1 for w in windows if w["subject_inside"] is False)
    if outside:
        warnings.append(f"the subject box reached outside the window in {outside} sample(s): raise --max-speed or lower --dead-zone")
    if not any(w["subject"] for w in windows):
        warnings.append("no subject in any sample: every window is centred")
    if shots is None:
        warnings.append("no --shots: the whole video is one shot, so the subject never changes at a cut")
    inputs = {"faces": _input_ref(faces, *files.get("faces", (None, None))),
              "objects": _input_ref(objects, *files.get("objects", (None, None))) if objects is not None else None,
              "shots": _input_ref(shots, *files.get("shots", (None, None))) if shots is not None else None}
    return {
        "schema": REFRAME_SCHEMA,
        "source": dict(faces["source"]),
        "frame_size": dict(faces["frame_size"]),
        "aspect": {"name": name, "ratio": round(ratio, 6)},
        "window": {"width": cw, "height": ch},
        "inputs": inputs,
        "method": {"name": METHOD, "heuristic": True, "identity": False, "note": NOTE},
        "parameters": {"dead_zone": dead_zone, "max_speed": max_speed, "dead_zone_px": round(dz_px, 2),
                       "max_speed_px_per_s": round(v_px, 2), "object_classes": classes if objects is not None else None},
        "windows": windows,
        "switches": switches,
        "shots": per_shot,
        "counts": {"windows": len(windows), "switches": len(switches), "shots": len(per_shot), "subject_outside": outside},
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }


def _sample_times(faces: Dict[str, Any], objects: Optional[Dict[str, Any]], duration: float) -> List[float]:
    det = faces["detections"]
    step = det.get("sampling_step") or 0.0
    if det.get("sampling_method") == "every" and step > 0:
        times = plan_every(float(step), duration, faces["source"].get("fps"))
    else:
        times = sorted({b["t"] for doc in (faces, objects) if doc is not None for tr in doc["tracks"] for b in tr["boxes"]})
    return times or [0.0]


def _box_of(subject: Optional[str], faces: List[Tuple[str, Dict[str, Any]]], objects: List[Tuple[str, Dict[str, Any]]],
            t: float) -> Optional[List[float]]:
    if subject is None:
        return None
    for sid, tr in faces + objects:
        if sid == subject:
            return _interp(tr, t)
    return None


def _choose(faces: List[Tuple[str, Dict[str, Any]]], objects: List[Tuple[str, Dict[str, Any]]], t: float,
            shot_end: float) -> Tuple[Optional[str], Optional[List[float]]]:
    for group in (faces, objects):
        best = None
        for sid, tr in group:
            box = _interp(tr, t)
            if box is None:
                continue
            p = _presence(tr, t, shot_end)
            key = (p, box[2] * box[3])
            if best is None or key > best[0]:
                best = (key, sid, box)
        if best is not None:
            return best[1], best[2]
    return None, None


def _follow(cur: float, target: float, dead: float, limit: float) -> float:
    d = target - cur
    if abs(d) <= dead:
        return cur
    move = min(abs(d) - dead, limit)
    return cur + math.copysign(move, d)


def _shot_window(sp: Dict[str, Any], ws: List[Dict[str, Any]], cw: int, ch: int, width: int, height: int) -> Dict[str, Any]:
    boxes = [w["subject_box"] for w in ws if w["subject_box"] is not None]
    subjects = [w["subject"] for w in ws if w["subject"]]
    if boxes:
        xs = [b[0] + b[2] / 2.0 for b in boxes]
        ys = [b[1] + b[3] / 2.0 for b in boxes]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    else:
        cx, cy = ws[0]["x"] + cw / 2.0, ws[0]["y"] + ch / 2.0
    x, y, _cx, _cy = _place(cx, cy, cw, ch, width, height)
    inside = [_inside(b, x, y, cw, ch) for b in boxes]
    main = max(sorted(set(subjects)), key=subjects.count) if subjects else None
    return {"index": sp["index"], "start": sp["start"], "end": sp["end"], "subject": main, "x": x, "y": y, "width": cw,
            "height": ch, "subject_inside_fraction": round(sum(inside) / len(inside), 4) if inside else None}


def fingerprint_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def validate_reframe(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("reframe document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != REFRAME_SCHEMA:
        rep.errors.append(f"schema must be {REFRAME_SCHEMA}")
    _check_run_fields(doc, rep, TOOL)
    check_source(doc["source"], rep)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    fs, win, par = doc["frame_size"], doc["window"], doc["parameters"]
    if not isinstance(fs, dict) or (fs.get("width"), fs.get("height")) != (src.get("width"), src.get("height")):
        rep.errors.append("frame_size must equal source width/height")
        return rep
    W, H = fs["width"], fs["height"]
    if not isinstance(win, dict) or not all(isinstance(win.get(k), int) and not isinstance(win.get(k), bool) and win[k] >= 2
                                             and win[k] % 2 == 0 for k in ("width", "height")) \
            or win["width"] > W or win["height"] > H:
        rep.errors.append("window must be even integer width/height inside the frame")
        return rep
    m = doc["method"]
    if not isinstance(m, dict) or m.get("heuristic") is not True or m.get("identity") is not False:
        rep.errors.append("method must declare heuristic: true and identity: false")
    v = par.get("max_speed_px_per_s") if isinstance(par, dict) else None
    if not _num(v) or v <= 0:
        rep.errors.append("parameters.max_speed_px_per_s must be > 0")
        v = None
    ws = doc["windows"]
    if not isinstance(ws, list) or not ws:
        rep.errors.append("windows must be a non-empty list")
        return rep
    prev: Optional[Dict[str, Any]] = None
    outside = 0
    for i, w in enumerate(ws):
        k = f"windows[{i}]"
        if not isinstance(w, dict) or any(f not in w for f in ("t", "shot", "x", "y", "width", "height", "subject", "subject_inside")):
            rep.errors.append(f"{k}: needs t, shot, x, y, width, height, subject, subject_inside")
            prev = None
            continue
        ints = all(isinstance(w[f], int) and not isinstance(w[f], bool) for f in ("x", "y", "width", "height", "shot"))
        if not ints or (w["width"], w["height"]) != (win["width"], win["height"]) or w["x"] < 0 or w["y"] < 0 \
                or w["x"] + w["width"] > W or w["y"] + w["height"] > H or w["x"] % 2 or w["y"] % 2:
            rep.errors.append(f"{k}: the window must be the document's size, at even x/y, inside the frame")
            prev = None
            continue
        if not _num(w["t"]) or w["t"] < 0 or (prev is not None and w["t"] <= prev["t"]):
            rep.errors.append(f"{k}: t must be increasing and >= 0")
        if w["subject"] is not None and (not isinstance(w["subject"], str) or not SUBJECT_RE.match(w["subject"])):
            rep.errors.append(f"{k}: subject must be face/track_nnn, object/track_nnn or null")
        if w["subject_inside"] is False:
            outside += 1
        if prev is not None and v is not None and w["shot"] == prev["shot"] and _num(w["t"]):
            limit = v * (w["t"] - prev["t"]) + SPEED_SLACK_PX
            if abs(w["x"] - prev["x"]) > limit or abs(w["y"] - prev["y"]) > limit:
                rep.errors.append(f"{k}: the window moved more than the speed limit inside a shot")
        if prev is not None and w["shot"] < prev["shot"]:
            rep.errors.append(f"{k}: shot indices must not go back")
        prev = w
    for i, s in enumerate(doc["switches"] if isinstance(doc["switches"], list) else []):
        if not isinstance(s, dict) or not isinstance(s.get("reason"), str) or not s["reason"] or not _num(s.get("t")):
            rep.errors.append(f"switches[{i}]: needs t and a reason")
    if not isinstance(doc["switches"], list):
        rep.errors.append("switches must be a list")
    shots = doc["shots"]
    if not isinstance(shots, list):
        rep.errors.append("shots must be a list")
        shots = []
    for i, s in enumerate(shots):
        if not isinstance(s, dict) or not all(isinstance(s.get(f), int) for f in ("x", "y", "width", "height")) \
                or s["x"] < 0 or s["y"] < 0 or s["x"] + s["width"] > W or s["y"] + s["height"] > H:
            rep.errors.append(f"shots[{i}]: needs a window x, y, width, height inside the frame")
    c = doc["counts"]
    if not isinstance(c, dict) or c.get("windows") != len(ws) or c.get("switches") != len(doc["switches"] or []) \
            or c.get("shots") != len(shots) or c.get("subject_outside") != outside:
        rep.errors.append("counts do not match")
    scan_leakage(doc, "", rep)
    return rep


__all__ = ["make_reframe", "validate_reframe", "parse_aspect", "window_size", "REFRAME_SCHEMA"]
