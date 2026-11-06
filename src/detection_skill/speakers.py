"""SPEAKERS: face tracks from two or more cameras (+ diarization-skill turns when given) -> for each time range, which
face is speaking and which camera shows it best, as a switch list for ffmpeg-skill `multicam`
(`detection-skill/speakers@1`, ADR-005). A heuristic, documented as one; it cuts nothing.

- **Input.** One face tracks document per camera, made with `detection faces --attributes` (each box carries its
  measured size, pose, sharpness and mouth contrast) -> `detection tracks`. Camera 0 is the reference, as in
  ffmpeg-skill `multicam`; `offsets` are multicam's `offsets_seconds` (reference time = camera time + offset).
- **Mouth motion.** A face's mouth activity at time t is the standard deviation of its `mouth_contrast` over the
  samples within `window` seconds around t (at least three samples): a talking mouth opens and closes, a silent one
  stays still. Sample faces densely (`--every 0.1` to `0.2`) for it to see the motion.
- **Camera quality.** Per box: 0.5 x min(1, size / 0.5) + 0.3 x frontal + 0.2 x sharpness / (sharpness + 100), from
  the attributes (size from the box when the attribute is missing; frontal and sharpness 0.5 when missing).
- **Without turns.** At each step the speaking faces are those whose activity is at least `min_activity` and at least
  half the highest activity at that time; among them the one with the best quality wins, so a close-up beats a wide
  shot of the same speaker. Nobody speaking: the camera holds.
- **With turns** (diarization-skill `turns@1`, on the reference timeline unless `turns_offset`): each anonymous
  speaker is matched, per camera, to the face track whose mouth activity is highest during that speaker's turns
  relative to the other speakers' turns. Each turn goes to the camera whose matched track is in view for at least
  half the turn with the best quality; a range starts at the turn's start and runs to the next turn's start.
- **Shortest shot.** Ranges shorter than `min_shot` seconds are folded into the previous one (the first into the
  next), then neighbours on the same camera are joined. `switch` is the ranges in `multicam --switch` form.
- A track id or a speaker id is not an identity. Mouth motion is not speech: chewing, laughing or a hand in front of
  the mouth moves it too, and a speaker turned away is not seen.
"""
from __future__ import annotations

import math
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .detections import ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now
from .errors import DetectionError
from .tracks import TRACK_ID_RE, validate_tracks

SPEAKERS_SCHEMA = "detection-skill/speakers@1"
TURNS_SCHEMA = "diarization-skill/turns@1"
TOOL = "detection/speakers"
METHODS = ("mouth-motion", "turns+mouth-motion")
DEFAULTS = {"window": 0.6, "min_activity": 0.02, "min_shot": 1.0}
SHARE = 0.5
MIN_POINTS = 3
SHARP_HALF = 100.0
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")
REQUIRED = ("schema", "source", "cameras", "turns", "method", "parameters", "matches", "ranges", "switch", "counts", "skill",
            "skill_version", "tool", "created_at", "processing_seconds", "warnings")
NOTE = ("heuristic: a face speaks when its mouth region changes over time (mouth contrast); the camera shown is the one "
        "with the best view (size, frontal angle, sharpness) of the speaking face; with diarization turns each anonymous "
        "speaker is matched to the face whose mouth moves most during their turns; neither a track id nor a speaker id "
        "is an identity")
DIGITS = 3


def quality(attrs: Optional[Dict[str, Any]], box: Sequence[float], frame_height: int) -> float:
    a = attrs or {}
    size = a.get("size") if _num(a.get("size")) else float(box[3]) / float(frame_height)
    frontal = a.get("frontal") if _num(a.get("frontal")) else 0.5
    sharp = a["sharpness"] / (a["sharpness"] + SHARP_HALF) if _num(a.get("sharpness")) else 0.5
    return 0.5 * min(1.0, size / 0.5) + 0.3 * frontal + 0.2 * sharp


class _Track:
    def __init__(self, camera: int, tr: Dict[str, Any], offset: float, frame_height: int) -> None:
        self.camera, self.id = camera, tr["id"]
        self.t = [b["t"] + offset for b in tr["boxes"]]
        self.mouth = [(b.get("attributes") or {}).get("mouth_contrast") for b in tr["boxes"]]
        self.q = [quality(b.get("attributes"), b["box"], frame_height) for b in tr["boxes"]]
        self.has_mouth = any(_num(m) for m in self.mouth)

    def activity(self, t: float, window: float) -> Optional[float]:
        vals = [m for tt, m in zip(self.t, self.mouth) if abs(tt - t) <= window / 2.0 + 1e-9 and _num(m)]
        return statistics.pstdev(vals) if len(vals) >= MIN_POINTS else None

    def visible(self, t: float, tol: float) -> bool:
        return self.t[0] - tol <= t <= self.t[-1] + tol

    def quality_in(self, t0: float, t1: float) -> Optional[float]:
        qs = [q for tt, q in zip(self.t, self.q) if t0 - 1e-9 <= tt < t1 - 1e-9]
        return sum(qs) / len(qs) if qs else None


def _parse_turns(turns: Dict[str, Any], offset: float, duration: float) -> List[Dict[str, Any]]:
    if not isinstance(turns, dict) or turns.get("schema") != TURNS_SCHEMA or not isinstance(turns.get("turns"), list):
        raise DetectionError("INVALID_DETECTIONS", f"--turns: not a {TURNS_SCHEMA} document")
    out = []
    for i, t in enumerate(turns["turns"]):
        if not isinstance(t, dict) or not _num(t.get("start")) or not _num(t.get("end")) or t["end"] <= t["start"] \
                or not isinstance(t.get("speaker"), str) or not SPEAKER_RE.match(t["speaker"]):
            raise DetectionError("INVALID_DETECTIONS", f"--turns: turns[{i}] needs start < end and a SPEAKER_nn speaker")
        s, e = max(0.0, t["start"] + offset), min(duration, t["end"] + offset)
        if e > s:
            out.append({"start": s, "end": e, "speaker": t["speaker"]})
    out.sort(key=lambda t: (t["start"], t["end"]))
    return out


def make_speakers(cameras: List[Dict[str, Any]], *, offsets: Optional[List[float]] = None, turns: Optional[Dict[str, Any]] = None,
                  turns_offset: float = 0.0, window: Optional[float] = None, min_activity: Optional[float] = None,
                  min_shot: Optional[float] = None,
                  files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """cameras: face tracks documents, camera 0 first (the reference). offsets: seconds per camera (reference time =
    camera time + offset; default 0). turns: a diarization-skill turns document (optional). files: {"cameras": [(name,
    fingerprint)], "turns": (name, fingerprint)} for provenance."""
    t0 = time.time()
    files = files or {}
    if not isinstance(cameras, list) or len(cameras) < 2:
        raise DetectionError("INVALID_ARGUMENT", "give two or more cameras (--camera tracks.json, camera 0 first)")
    for i, doc in enumerate(cameras):
        rep = validate_tracks(doc)
        if not rep.ok:
            raise DetectionError("INVALID_DETECTIONS", f"camera {i}: not a valid tracks document", rep.errors[:10])
        if doc["kind"] != "face":
            raise DetectionError("INVALID_ARGUMENT", f"camera {i}: must be face tracks (detection faces -> detection tracks)")
    fps = [doc["source"]["fingerprint"] for doc in cameras]
    if len(set(fps)) != len(fps):
        raise DetectionError("INVALID_ARGUMENT", "two cameras are the same file (source fingerprints are equal)")
    offsets = [0.0] * len(cameras) if offsets is None else [float(o) for o in offsets]
    if len(offsets) != len(cameras) or not all(math.isfinite(o) for o in offsets):
        raise DetectionError("INVALID_ARGUMENT", f"--offsets needs one number per camera ({len(cameras)})")
    window = DEFAULTS["window"] if window is None else float(window)
    min_activity = DEFAULTS["min_activity"] if min_activity is None else float(min_activity)
    min_shot = DEFAULTS["min_shot"] if min_shot is None else float(min_shot)
    if not math.isfinite(window) or window <= 0:
        raise DetectionError("INVALID_ARGUMENT", "--window must be > 0 seconds")
    if not math.isfinite(min_activity) or min_activity < 0:
        raise DetectionError("INVALID_ARGUMENT", "--min-activity must be >= 0")
    if not math.isfinite(min_shot) or min_shot < 0:
        raise DetectionError("INVALID_ARGUMENT", "--min-shot must be >= 0 seconds")
    if not math.isfinite(float(turns_offset)):
        raise DetectionError("INVALID_ARGUMENT", "--turns-offset must be a number")

    ref = cameras[0]
    duration = float(ref["source"]["duration"])
    steps = [doc["detections"].get("sampling_step") for doc in cameras]
    step = min([s for s in steps if _num(s) and s > 0] or [0.1])
    tracks = [_Track(c, tr, offsets[c], doc["frame_size"]["height"]) for c, doc in enumerate(cameras) for tr in doc["tracks"]]
    warnings: List[str] = []
    bare = [c for c, doc in enumerate(cameras) if doc["tracks"] and not any(t.has_mouth for t in tracks if t.camera == c)]
    if bare or not tracks:
        raise DetectionError("INVALID_ARGUMENT", f"camera(s) {bare or list(range(len(cameras)))}: no face box carries "
                                                 "mouth_contrast: run detection faces --attributes (then detection tracks) "
                                                 "on every camera")
    for c, doc in enumerate(cameras):
        if not doc["tracks"]:
            warnings.append(f"camera {c}: no face track")
        if steps[c] is not None and _num(steps[c]) and steps[c] * MIN_POINTS > window + 1e-9:
            warnings.append(f"camera {c}: sampled every {steps[c]:g}s, fewer than {MIN_POINTS} samples per {window:g}s "
                            "window: sample faces more densely (--every 0.1 to 0.2) or widen --window")
    grid = [round(k * step, 6) for k in range(int(math.floor(duration / step + 1e-9)) + 1) if k * step < duration - 1e-9]

    matches: List[Dict[str, Any]] = []
    turn_list: Optional[List[Dict[str, Any]]] = None
    if turns is not None:
        turn_list = _parse_turns(turns, float(turns_offset), duration)
        if not turn_list:
            warnings.append("--turns has no turn inside the reference camera's duration")
        acts = {id(tr): {t: tr.activity(t, window) for t in grid} for tr in tracks}
        matches = _match(turn_list, tracks, acts, grid, min_activity, len(cameras))
        raw = _by_turns(turn_list, tracks, matches, step, duration)
        method = METHODS[1]
    else:
        raw = _by_mouth(tracks, grid, step, window, min_activity, duration)
        method = METHODS[0]
    ranges = _fold(raw, min_shot, duration)
    for r in ranges:
        tr = next((t for t in tracks if t.camera == r["camera"] and t.id == r["track"]), None)
        q = tr.quality_in(r["start"], r["end"]) if tr is not None else None
        r["quality"] = round(q, 4) if q is not None else None
    switch = switch_list(ranges)
    n_switch = sum(1 for a, b in zip(ranges, ranges[1:]) if a["camera"] != b["camera"])
    if not any(r["track"] for r in ranges):
        warnings.append("no speaking face was found: the switch list holds camera 0")
    cam_refs = []
    for c, doc in enumerate(cameras):
        name, fp = (files.get("cameras") or [(None, None)] * len(cameras))[c]
        cam_refs.append({"index": c, "file": os.path.basename(name) if name else None, "fingerprint": fp,
                         "source": {"filename": doc["source"]["filename"], "fingerprint": doc["source"]["fingerprint"],
                                    "duration": doc["source"]["duration"]},
                         "offset": offsets[c], "tracks": len(doc["tracks"]), "sampling_step": steps[c],
                         "attributes": any(t.has_mouth for t in tracks if t.camera == c)})
    turns_ref = None
    if turns is not None:
        name, fp = files.get("turns") or (None, None)
        turns_ref = {"file": os.path.basename(name) if name else None, "fingerprint": fp, "tool": turns.get("tool"),
                     "offset": float(turns_offset), "speakers": sorted({t["speaker"] for t in turn_list or []}),
                     "turns": len(turn_list or [])}
    return {
        "schema": SPEAKERS_SCHEMA,
        "source": dict(ref["source"]),
        "cameras": cam_refs,
        "turns": turns_ref,
        "method": {"name": method, "heuristic": True, "identity": False, "note": NOTE},
        "parameters": {"window": window, "min_activity": min_activity, "min_shot": min_shot, "step": step, "share": SHARE},
        "matches": matches,
        "ranges": ranges,
        "switch": switch,
        "counts": {"cameras": len(cameras), "ranges": len(ranges), "switches": n_switch, "matches": len(matches)},
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }


def _by_mouth(tracks: List[_Track], grid: List[float], step: float, window: float, min_activity: float,
              duration: float) -> List[Dict[str, Any]]:
    picks: List[Optional[Tuple[_Track, float, float]]] = []
    for t in grid:
        cands = []
        for tr in tracks:
            if not tr.visible(t, step):
                continue
            a = tr.activity(t, window)
            if a is not None and a >= min_activity:
                cands.append((tr, a))
        if not cands:
            picks.append(None)
            continue
        top = max(a for _tr, a in cands)
        speaking = [(tr, a, _q_at(tr, t)) for tr, a in cands if a >= SHARE * top]
        picks.append(max(speaking, key=lambda x: (x[2], x[1], -x[0].camera)))
    segs: List[Tuple[float, Dict[str, Any]]] = []
    for t, p in zip(grid, picks):
        if p is None:
            if not segs:  # nobody has spoken yet; afterwards the camera holds
                segs.append((t, {"camera": 0, "track": None, "speaker": None,
                                 "reason": "no face speaking yet: the reference camera"}))
            continue
        tr, a, q = p
        if segs and (segs[-1][1]["camera"], segs[-1][1]["track"]) == (tr.camera, tr.id):
            continue
        segs.append((t, {"camera": tr.camera, "track": tr.id, "speaker": None,
                         "reason": f"camera {tr.camera} {tr.id}: mouth activity {a:.3f} at {t:.3f}s, the best view of a "
                                   f"speaking face (quality {q:.2f})"}))
    if not segs:
        segs.append((0.0, {"camera": 0, "track": None, "speaker": None, "reason": "no sample: the reference camera"}))
    out = []
    for i, (t, r) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else duration
        out.append(dict(r, start=0.0 if i == 0 else t, end=end))
    return out


def _q_at(tr: _Track, t: float) -> float:
    i = min(range(len(tr.t)), key=lambda k: abs(tr.t[k] - t))
    return tr.q[i]


def _match(turns: List[Dict[str, Any]], tracks: List[_Track], acts: Dict[int, Dict[float, Optional[float]]],
           grid: List[float], min_activity: float, n_cams: int) -> List[Dict[str, Any]]:
    speakers = sorted({t["speaker"] for t in turns})
    who: Dict[float, Optional[str]] = {}
    for t in grid:
        who[t] = next((u["speaker"] for u in turns if u["start"] - 1e-9 <= t < u["end"] - 1e-9), None)
    out = []
    for s in speakers:
        for c in range(n_cams):
            best = None
            for tr in (x for x in tracks if x.camera == c):
                during = [a for t, a in acts[id(tr)].items() if who[t] == s and a is not None]
                other = [a for t, a in acts[id(tr)].items() if who[t] not in (None, s) and a is not None]
                if len(during) < MIN_POINTS:
                    continue
                d = sum(during) / len(during)
                o = sum(other) / len(other) if other else 0.0
                score = d - o
                if d >= min_activity and score > 0 and (best is None or score > best[0]):
                    best = (score, tr, d, o)
            if best is not None:
                out.append({"speaker": s, "camera": c, "track": best[1].id, "score": round(best[0], 4),
                            "activity_in_turns": round(best[2], 4), "activity_elsewhere": round(best[3], 4)})
    return out


def _by_turns(turns: List[Dict[str, Any]], tracks: List[_Track], matches: List[Dict[str, Any]], step: float,
              duration: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, u in enumerate(turns):
        nxt = turns[i + 1]["start"] if i + 1 < len(turns) else duration
        best = None
        for m in (m for m in matches if m["speaker"] == u["speaker"]):
            tr = next(t for t in tracks if t.camera == m["camera"] and t.id == m["track"])
            ts = [u["start"] + k * step for k in range(max(1, int((u["end"] - u["start"]) / step)))]
            seen = sum(1 for t in ts if tr.visible(t, step)) / float(len(ts))
            q = tr.quality_in(u["start"], u["end"])
            if seen >= 0.5 and q is not None and (best is None or q > best[0]):
                best = (q, tr)
        if best is None:
            if not out:
                out.append({"start": 0.0, "end": nxt, "camera": 0, "track": None, "speaker": None,
                            "reason": f"{u['speaker']} is matched to no face in view: the reference camera"})
            else:
                out[-1]["end"] = nxt  # hold
            continue
        q, tr = best
        start = 0.0 if not out else u["start"]
        if out and out[-1]["camera"] == tr.camera and out[-1]["track"] == tr.id and out[-1]["speaker"] == u["speaker"]:
            out[-1]["end"] = nxt
            continue
        if not out and u["start"] > 0:
            out.append({"start": 0.0, "end": u["start"], "camera": 0, "track": None, "speaker": None,
                        "reason": "before the first turn: the reference camera"})
            start = u["start"]
        out.append({"start": start, "end": nxt, "camera": tr.camera, "track": tr.id, "speaker": u["speaker"],
                    "reason": f"turn of {u['speaker']} at {u['start']:.3f}s: matched to camera {tr.camera} {tr.id}, "
                              f"the best view of that face in view (quality {q:.2f})"})
    if not out:
        out.append({"start": 0.0, "end": duration, "camera": 0, "track": None, "speaker": None,
                    "reason": "no turn: the reference camera"})
    out[-1]["end"] = duration
    return out


def _fold(ranges: List[Dict[str, Any]], min_shot: float, duration: float) -> List[Dict[str, Any]]:
    rs = [dict(r) for r in ranges]
    while len(rs) > 1:
        short = [i for i, r in enumerate(rs) if r["end"] - r["start"] < min_shot - 1e-9]
        if not short:
            break
        i = min(short, key=lambda k: (rs[k]["end"] - rs[k]["start"], k))
        if i == 0:
            rs[1]["start"] = rs[0]["start"]
        else:
            rs[i - 1]["end"] = rs[i]["end"]
        rs.pop(i)
    joined: List[Dict[str, Any]] = []
    for r in rs:
        if joined and joined[-1]["camera"] == r["camera"] and joined[-1]["track"] == r["track"] \
                and joined[-1]["speaker"] == r["speaker"]:
            joined[-1]["end"] = r["end"]
        else:
            joined.append(r)
    for r in joined:
        r["start"], r["end"] = round(r["start"], DIGITS), round(r["end"], DIGITS)
    joined[0]["start"], joined[-1]["end"] = 0.0, round(duration, DIGITS)
    return joined


def switch_list(ranges: List[Dict[str, Any]]) -> str:
    """The ranges as ffmpeg-skill multicam's --switch: START-END:CAM, neighbours on one camera joined."""
    spans: List[List[Any]] = []
    for r in ranges:
        if spans and spans[-1][2] == r["camera"]:
            spans[-1][1] = r["end"]
        else:
            spans.append([r["start"], r["end"], r["camera"]])
    return ",".join(f"{s:.3f}-{e:.3f}:{c}" for s, e, c in spans)


def validate_speakers(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("speakers document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != SPEAKERS_SCHEMA:
        rep.errors.append(f"schema must be {SPEAKERS_SCHEMA}")
    _check_run_fields(doc, rep, TOOL)
    check_source(doc["source"], rep)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    duration = src.get("duration") if _num(src.get("duration")) else None
    m = doc["method"]
    if not isinstance(m, dict) or m.get("heuristic") is not True or m.get("identity") is not False or m.get("name") not in METHODS:
        rep.errors.append(f"method must be one of {METHODS} with heuristic: true and identity: false")
    cams = doc["cameras"]
    if not isinstance(cams, list) or len(cams) < 2 or any(not isinstance(c, dict) or c.get("index") != i or not _num(c.get("offset"))
                                                          for i, c in enumerate(cams)):
        rep.errors.append("cameras must list two or more cameras with index 0.. and an offset")
        return rep
    if (doc["turns"] is None) != (isinstance(m, dict) and m.get("name") == METHODS[0]):
        rep.errors.append("turns must be given exactly when the method is turns+mouth-motion")
    ranges = doc["ranges"]
    if not isinstance(ranges, list) or not ranges:
        rep.errors.append("ranges must be a non-empty list")
        return rep
    prev_end = 0.0
    for i, r in enumerate(ranges):
        k = f"ranges[{i}]"
        if not isinstance(r, dict) or any(f not in r for f in ("start", "end", "camera", "track", "speaker", "reason", "quality")):
            rep.errors.append(f"{k}: needs start, end, camera, track, speaker, reason, quality")
            return rep
        if not _num(r["start"]) or not _num(r["end"]) or r["end"] <= r["start"]:
            rep.errors.append(f"{k}: needs start < end")
        elif abs(r["start"] - prev_end) > 1e-6:
            rep.errors.append(f"{k}: ranges must be contiguous from 0")
        prev_end = r["end"] if _num(r["end"]) else prev_end
        if not isinstance(r["camera"], int) or isinstance(r["camera"], bool) or not 0 <= r["camera"] < len(cams):
            rep.errors.append(f"{k}: camera must be a camera index")
        if r["track"] is not None and (not isinstance(r["track"], str) or not TRACK_ID_RE.match(r["track"])):
            rep.errors.append(f"{k}: track must be track_nnn or null")
        if r["speaker"] is not None and (not isinstance(r["speaker"], str) or not SPEAKER_RE.match(r["speaker"])):
            rep.errors.append(f"{k}: speaker must be SPEAKER_nn or null (anonymous)")
        if not isinstance(r["reason"], str) or not r["reason"]:
            rep.errors.append(f"{k}: needs a reason")
    if duration is not None and abs(prev_end - duration) > 1e-3:
        rep.errors.append("ranges must end at the reference camera's duration")
    if doc["switch"] != switch_list(ranges):
        rep.errors.append("switch must be the ranges in multicam --switch form")
    if not isinstance(doc["matches"], list):
        rep.errors.append("matches must be a list")
    else:
        for i, x in enumerate(doc["matches"]):
            if not isinstance(x, dict) or not isinstance(x.get("speaker"), str) or not SPEAKER_RE.match(x["speaker"]) \
                    or not isinstance(x.get("track"), str) or not TRACK_ID_RE.match(x["track"]) \
                    or not isinstance(x.get("camera"), int) or not 0 <= x["camera"] < len(cams):
                rep.errors.append(f"matches[{i}]: needs a SPEAKER_nn speaker, a camera index and a track_nnn")
    c = doc["counts"]
    n_switch = sum(1 for a, b in zip(ranges, ranges[1:]) if a.get("camera") != b.get("camera"))
    if not isinstance(c, dict) or c.get("ranges") != len(ranges) or c.get("cameras") != len(cams) \
            or c.get("switches") != n_switch or c.get("matches") != len(doc["matches"] or []):
        rep.errors.append("counts do not match")
    scan_leakage(doc, "", rep)
    return rep


__all__ = ["make_speakers", "validate_speakers", "switch_list", "quality", "SPEAKERS_SCHEMA"]
