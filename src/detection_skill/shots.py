"""SHOTS: video -> shots document (`detection-skill/shots@1`): the measured shot list (ADR-003).

input check + fingerprint -> engine.prepare -> ffmpeg-skill probe (frame rate, frame count) -> every frame through
ffmpeg-skill `look --at` (frame k asked for at its time rounded down to the millisecond, `frames.seek_time`), at a
reduced width (default 320 px) -> the shot engine in a worker subprocess (PySceneDetect by default, TransNetV2 when
asked and installed) -> boundaries -> contiguous shots that cover every frame -> validated document.

ffmpeg-skill `scenes` (scdet) stays the quick cut list; this is the frame-accurate one the edit view and qc-skill's
cut checks read. A boundary's `frame` is the first frame of the next shot. Shots are [start_frame, end_frame] with
both frames inclusive; `start` = start_frame / rate and `end` = (end_frame + 1) / rate, in seconds.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from . import __version__
from .detections import ValidationReport, _check_run_fields, _num, check_source, scan_leakage, utc_now, SHA_RE
from .engines import DEFAULT_SHOT_ENGINE, SHOT_ENGINES, get_engine
from .engines.base import DetectionEngine, EngineRequest
from .errors import DetectionError
from .frames import Runner, extract_frames, frame_count, is_image, probe_video, require, seek_time, subprocess_runner
from .models import resolve_models_dir
from .service import DEFAULT_TIMEOUT, fingerprint_file, run_engine

SHOTS_SCHEMA = "detection-skill/shots@1"
TOOL = "detection/shots"
SHOT_WIDTH = 320
DEFAULT_MAX_FRAMES_SHOTS = 20000
BOUNDARY_KINDS = ("cut", "gradual")
SHOT_ID_RE = re.compile(r"^shot_\d{3,}$")
REQUIRED = ("schema", "source", "frames", "engine", "parameters", "boundaries", "shots", "counts", "skill", "skill_version",
            "tool", "created_at", "processing_seconds", "warnings")
TIME_DIGITS = 3


def detect_shots(media: str, *, engine_id: Optional[str] = None, threshold: Optional[float] = None,
                 min_shot_frames: Optional[int] = None, width: int = SHOT_WIDTH, max_frames: int = DEFAULT_MAX_FRAMES_SHOTS,
                 models_dir: Optional[str] = None, model: Optional[str] = None, ffmpeg_skill: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT, engine: Optional[DetectionEngine] = None,
                 runner: Runner = subprocess_runner) -> Dict[str, Any]:
    t0 = time.time()
    if engine is None:
        engine_id = engine_id or DEFAULT_SHOT_ENGINE
        if engine_id not in SHOT_ENGINES:
            raise DetectionError("INVALID_ARGUMENT", f"--engine must be one of {list(SHOT_ENGINES)}")
    if timeout <= 0:
        raise DetectionError("INVALID_ARGUMENT", "--timeout must be > 0")
    if not isinstance(max_frames, int) or max_frames < 1:
        raise DetectionError("INVALID_ARGUMENT", "--max-frames must be >= 1")
    if not isinstance(width, int) or width < 16:
        raise DetectionError("INVALID_ARGUMENT", "--width must be an integer >= 16")
    if min_shot_frames is not None and (not isinstance(min_shot_frames, int) or min_shot_frames < 1):
        raise DetectionError("INVALID_ARGUMENT", "--min-shot-frames must be >= 1")
    if threshold is not None and (not _num(threshold) or threshold <= 0):
        raise DetectionError("INVALID_ARGUMENT", "--threshold must be > 0")
    if not os.path.isfile(media):
        raise DetectionError("INPUT_NOT_FOUND", f"input not found: {os.path.basename(media)}")
    if is_image(media):
        raise DetectionError("INVALID_ARGUMENT", "shots need a video; an image is one frame")
    eng = engine if engine is not None else get_engine(engine_id)  # type: ignore[arg-type]
    if eng.kind != "shot":
        raise DetectionError("INVALID_ARGUMENT", f"engine {eng.id} detects {eng.kind}, not shots")
    prepared = eng.prepare(models_dir=resolve_models_dir(models_dir), model=model, params={})
    warnings: List[str] = list(prepared.warnings)
    fingerprint = fingerprint_file(media)
    skill = require(ffmpeg_skill)
    facts = probe_video(skill, media, runner=runner)
    rate = facts.get("frame_rate") or facts.get("fps")
    if not rate:
        raise DetectionError("UNSUPPORTED_MEDIA", f"{os.path.basename(media)}: frame rate unknown (ffmpeg-skill probe)")
    if facts.get("variable_frame_rate"):
        warnings.append("ffmpeg-skill probe suspects a variable frame rate: frame numbers assume a constant rate")
    n = frame_count(facts)
    if n > max_frames:
        raise DetectionError("INVALID_ARGUMENT", f"{n} frames, more than --max-frames {max_frames}",
                             "shots look at every frame; pass a larger --max-frames for a long video")
    out_w = min(int(width), int(facts["width"]))
    out_w -= out_w % 2
    params: Dict[str, Any] = {"fps": rate}
    if threshold is not None:
        params["threshold"] = float(threshold)
    if min_shot_frames is not None:
        params["min_shot_frames"] = int(min_shot_frames)
    run_dir = tempfile.mkdtemp(prefix="detection-shots-")
    try:
        times = [seek_time(k, rate) for k in range(n)]
        frames = extract_frames(skill, media, times, out_w, run_dir, runner=runner, timeout=timeout)
        ereq = EngineRequest(frames=frames, model_path=prepared.model_path, params=params)
        t_engine = time.time()
        result = run_engine(eng, ereq, timeout, run_dir, injected=engine is not None)
        engine_seconds = round(time.time() - t_engine, 3)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    data = result.data or {}
    if int(data.get("frames", -1)) != n:
        raise DetectionError("INVALID_RESULT", f"shot engine saw {data.get('frames')} frame(s), expected {n}")
    boundaries = _boundaries(data.get("boundaries") or [], n, rate, eng.id)
    shots = _shots(boundaries, n, rate, float(facts["duration"]), eng.id)
    engine_doc = {"id": eng.id, "kind": "shot", "version": prepared.version, "runtime": eng.runtime, "model": prepared.model,
                  "model_sha256": prepared.model_sha256 or None, "model_license": prepared.license, "engine_license": eng.license,
                  "files": prepared.files, "execution_mode": eng.execution_mode, "engine_seconds": engine_seconds,
                  "details": {k: v for k, v in (result.runtime or {}).items() if isinstance(v, (str, int, float)) or v is None}}
    if data.get("merged"):
        warnings.append(f"{data['merged']} detector cut(s) merged into a neighbouring boundary (same transition, or closer "
                        f"than the minimum shot length)")
    source = {"filename": os.path.basename(media), "fingerprint": fingerprint, "size_bytes": os.path.getsize(media),
              "media_type": "video", "duration": facts["duration"], "width": int(facts["width"]), "height": int(facts["height"]),
              "fps": round(float(rate), 6)}
    frames_doc = {"method": "every_frame", "frame_rate": round(float(rate), 6), "frame_count": n,
                  "seek_rule": "frame k at floor_ms(k / frame_rate) with ffmpeg-skill look --at (first frame at or after T)",
                  "frames_tool": "ffmpeg-skill/look", "ffmpeg_skill_version": skill.version,
                  "extracted_size": {"width": int((result.runtime or {}).get("frame_width") or out_w),
                                     "height": int((result.runtime or {}).get("frame_height") or 0)}}
    doc = {
        "schema": SHOTS_SCHEMA,
        "source": source,
        "frames": frames_doc,
        "engine": engine_doc,
        "parameters": dict(data.get("parameters") or {}),
        "boundaries": boundaries,
        "shots": shots,
        "counts": _counts(boundaries, shots, n),
        "skill": "detection-skill",
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": utc_now(),
        "processing_seconds": round(time.time() - t0, 3),
        "warnings": warnings,
    }
    rep = validate_shots(doc, expected_fingerprint=fingerprint)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "shot engine output failed shots validation; nothing returned", rep.errors[:10])
    return doc


def _t(frame: int, rate: float) -> float:
    return round(frame / float(rate), TIME_DIGITS)


def _boundaries(raw: List[Dict[str, Any]], n: int, rate: float, engine_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in sorted(raw, key=lambda b: int(b["frame"])):
        frame = int(b["frame"])
        if not 0 < frame < n or (out and frame <= out[-1]["frame"]):
            continue
        kind = b.get("kind")
        if kind not in BOUNDARY_KINDS:
            raise DetectionError("INVALID_RESULT", f"boundary kind {kind!r} is not one of {BOUNDARY_KINDS}")
        score = float(b.get("score", 0.0))
        transition = None
        if kind == "gradual":
            a, z = int(b["start_frame"]), int(b["end_frame"])
            transition = {"start_frame": a, "end_frame": z, "start": _t(a, rate), "end": _t(z + 1, rate), "frames": z - a + 1}
        out.append({"index": len(out), "frame": frame, "t": _t(frame, rate), "kind": kind,
                    "score": round(min(1.0, max(0.0, score if math.isfinite(score) else 0.0)), 4), "transition": transition,
                    "engine": engine_id, "metric": {k: v for k, v in (b.get("metric") or {}).items()
                                                    if isinstance(v, (str, int, float, list)) or v is None}})
    return out


def _shots(boundaries: List[Dict[str, Any]], n: int, rate: float, duration: float, engine_id: str) -> List[Dict[str, Any]]:
    starts = [0] + [b["frame"] for b in boundaries]
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else n - 1
        b = boundaries[i - 1] if i else None
        end_t = min(_t(e + 1, rate), round(duration, TIME_DIGITS)) if i + 1 == len(starts) else _t(e + 1, rate)
        out.append({"id": f"shot_{i + 1:03d}", "index": i, "start_frame": s, "end_frame": e, "frames": e - s + 1,
                    "start": _t(s, rate), "end": end_t, "duration": round(end_t - _t(s, rate), TIME_DIGITS),
                    "starts_with": b["kind"] if b else "start", "score": b["score"] if b else None, "engine": engine_id})
    return out


def _counts(boundaries: List[Dict[str, Any]], shots: List[Dict[str, Any]], n: int) -> Dict[str, int]:
    return {"frames": n, "shots": len(shots), "boundaries": len(boundaries),
            "cuts": sum(1 for b in boundaries if b["kind"] == "cut"), "gradual": sum(1 for b in boundaries if b["kind"] == "gradual")}


def shot_spans(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{index, id, start, end, starts_with}] in seconds, for consumers (reframe)."""
    return [{"index": s["index"], "id": s["id"], "start": s["start"], "end": s["end"], "starts_with": s["starts_with"]}
            for s in doc["shots"]]


def validate_shots(doc: Any, expected_fingerprint: Optional[str] = None) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("shots document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != SHOTS_SCHEMA:
        rep.errors.append(f"schema must be {SHOTS_SCHEMA}")
    _check_run_fields(doc, rep, TOOL)
    check_source(doc["source"], rep, expected_fingerprint)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    if src.get("media_type") != "video":
        rep.errors.append("shots are measured on a video")
    fr = doc["frames"]
    n = fr.get("frame_count") if isinstance(fr, dict) else None
    rate = fr.get("frame_rate") if isinstance(fr, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or not _num(rate) or rate <= 0:
        rep.errors.append("frames needs frame_count (integer >= 1) and frame_rate (> 0)")
        return rep
    eng = doc["engine"]
    if not isinstance(eng, dict) or not isinstance(eng.get("id"), str) or not eng.get("id") or eng.get("kind") != "shot":
        rep.errors.append("engine needs a non-empty id and kind 'shot'")
        eng = {}
    elif eng.get("model_sha256") is not None and not SHA_RE.match(str(eng["model_sha256"])):
        rep.errors.append("engine.model_sha256 must be 64 lowercase hex or null (no model file)")
    bs, shots = doc["boundaries"], doc["shots"]
    if not isinstance(bs, list) or not isinstance(shots, list) or not shots:
        rep.errors.append("boundaries must be a list and shots a non-empty list")
        return rep
    prev = 0
    for i, b in enumerate(bs):
        w = f"boundaries[{i}]"
        if not isinstance(b, dict) or any(k not in b for k in ("index", "frame", "t", "kind", "score", "transition", "engine")):
            rep.errors.append(f"{w}: needs index, frame, t, kind, score, transition, engine")
            continue
        if b["index"] != i:
            rep.errors.append(f"{w}: index must be {i}")
        f = b["frame"]
        if not isinstance(f, int) or isinstance(f, bool) or not prev < f < n:
            rep.errors.append(f"{w}: frame must be an integer in (previous boundary, frame_count)")
        else:
            prev = f
            if not _num(b["t"]) or abs(b["t"] - f / rate) > 0.0011:
                rep.errors.append(f"{w}: t must be frame / frame_rate")
        if b["kind"] not in BOUNDARY_KINDS:
            rep.errors.append(f"{w}: kind must be one of {BOUNDARY_KINDS}")
        if not _num(b["score"]) or not 0 <= b["score"] <= 1:
            rep.errors.append(f"{w}: score must be in [0, 1]")
        tr = b["transition"]
        if b["kind"] == "cut" and tr is not None:
            rep.errors.append(f"{w}: a cut has transition null")
        if b["kind"] == "gradual":
            if not isinstance(tr, dict) or not all(isinstance(tr.get(k), int) for k in ("start_frame", "end_frame")) \
                    or not 0 <= tr["start_frame"] <= tr["end_frame"] < n:
                rep.errors.append(f"{w}: a gradual boundary needs transition start_frame <= end_frame inside the video")
        if b["engine"] != eng.get("id"):
            rep.errors.append(f"{w}: engine must equal the document's engine id")
    expect_start = 0
    for i, s in enumerate(shots):
        w = f"shots[{i}]"
        if not isinstance(s, dict) or any(k not in s for k in ("id", "index", "start_frame", "end_frame", "frames", "start", "end",
                                                               "starts_with", "score", "engine")):
            rep.errors.append(f"{w}: needs id, index, start_frame, end_frame, frames, start, end, starts_with, score, engine")
            continue
        if s["index"] != i or not isinstance(s["id"], str) or not SHOT_ID_RE.match(s["id"]):
            rep.errors.append(f"{w}: index must be {i} and id shot_nnn")
        if s["start_frame"] != expect_start:
            rep.errors.append(f"{w}: shots must cover every frame in order (start_frame {expect_start} expected)")
        if not isinstance(s["end_frame"], int) or s["end_frame"] < s["start_frame"] or s["frames"] != s["end_frame"] - s["start_frame"] + 1:
            rep.errors.append(f"{w}: end_frame >= start_frame and frames = end_frame - start_frame + 1")
            continue
        expect_start = s["end_frame"] + 1
        if i == 0:
            if s["starts_with"] != "start" or s["score"] is not None:
                rep.errors.append(f"{w}: the first shot starts_with 'start' and has score null")
        elif i - 1 < len(bs) and isinstance(bs[i - 1], dict):
            b = bs[i - 1]
            if s["start_frame"] != b.get("frame") or s["starts_with"] != b.get("kind") or s["score"] != b.get("score"):
                rep.errors.append(f"{w}: must start at boundaries[{i - 1}] with its kind and score")
        if not _num(s["start"]) or not _num(s["end"]) or s["end"] <= s["start"]:
            rep.errors.append(f"{w}: needs start < end in seconds")
        if s["engine"] != eng.get("id"):
            rep.errors.append(f"{w}: engine must equal the document's engine id")
    if expect_start != n:
        rep.errors.append(f"shots must end at the last frame ({n - 1})")
    if len(shots) != len(bs) + 1:
        rep.errors.append("there must be one more shot than boundaries")
    c = doc["counts"]
    want = {"frames": n, "shots": len(shots), "boundaries": len(bs),
            "cuts": sum(1 for b in bs if isinstance(b, dict) and b.get("kind") == "cut"),
            "gradual": sum(1 for b in bs if isinstance(b, dict) and b.get("kind") == "gradual")}
    if c != want:
        rep.errors.append(f"counts must be {want}")
    scan_leakage(doc, "", rep)
    return rep


__all__ = ["detect_shots", "validate_shots", "shot_spans", "SHOTS_SCHEMA"]
