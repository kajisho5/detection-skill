"""Detections: the document FACES / TEXT / OBJECTS write (`detection-skill/detections@1`), how it is built from an
engine result, and its validator. `schemas/detections.schema.json` is the same contract as a JSON Schema; this
module is the executable form and is what `detection check` runs.

Rules kept by construction and checked here:
- a box is a measured fact: frame time `t` (seconds on the source timeline, null for an image), `box` [x, y, w, h]
  in source pixels (top-left origin) and `box_norm` (the same divided by the frame size), `score` in [0, 1],
  and its provenance `engine`, `model`, `model_sha256` on every detection.
- boxes lie inside the frame (engine boxes that reach past an edge are clipped to it).
- a face box is never an identity: its `label` is null and no identity, name or embedding key may appear.
- no path, command line or credential appears anywhere in the document.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional

from .engines.boxes import clip

DETECTIONS_SCHEMA = "detection-skill/detections@1"
KINDS = ("face", "text", "object")
TOOLS = {"face": "detection/faces", "text": "detection/text", "object": "detection/objects"}
METHODS = ("every", "at", "scenes", "image")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
BOX_TOLERANCE = 0.5   # px
NORM_TOLERANCE = 1e-3
TIME_TOLERANCE = 0.05  # s past the duration
PX_DIGITS, NORM_DIGITS, SCORE_DIGITS = 2, 5, 4

REQUIRED = ("schema", "kind", "source", "frame_size", "sampling", "engine", "parameters", "frames", "counts", "skill",
            "skill_version", "tool", "created_at", "processing_seconds", "warnings")
REQUIRED_SOURCE = ("filename", "fingerprint", "size_bytes", "media_type", "duration", "width", "height")
REQUIRED_ENGINE = ("id", "kind", "version", "model", "model_sha256", "model_license", "files", "execution_mode")
REQUIRED_DET = ("t", "kind", "box", "box_norm", "score", "label", "engine", "model", "model_sha256")
IDENTITY_KEYS = {"identity", "name", "person", "person_id", "identity_id", "embedding", "feature", "features", "descriptor",
                 "face_id", "recognition"}
FORBIDDEN_KEYS = {"command", "argv", "cmd", "shell", "api_key", "apikey", "token", "secret", "password", "authorization",
                  "access_key", "secret_key", "credential", "credentials"}
CREDENTIAL_PATTERNS = [re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
                       re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
                       re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")]
MEASURED_TEXT_KEYS = {"label"}
ABS_PATH_PATTERNS = [re.compile(r"^(/|~/)[^\s]+/[^\s]*"), re.compile(r"^[A-Za-z]:\\[^\s]+")]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---- building ----------------------------------------------------------------------------------
def build_detection(raw: Dict[str, Any], *, kind: str, t: Optional[float], sx: float, sy: float, width: int, height: int,
                    engine: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One engine box (frame pixels) -> one detection (source pixels, clipped). None when nothing is left after
    clipping (a box entirely outside the frame)."""
    b = raw.get("box")
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(_num(v) for v in b):
        return None
    box = clip([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], width, height)
    if box[2] < 1e-6 or box[3] < 1e-6:
        return None
    score = float(raw.get("score", 0.0))
    if not math.isfinite(score):
        return None
    det: Dict[str, Any] = {
        "t": t, "kind": kind,
        "box": [round(v, PX_DIGITS) for v in box],
        "box_norm": [round(box[0] / width, NORM_DIGITS), round(box[1] / height, NORM_DIGITS),
                     round(box[2] / width, NORM_DIGITS), round(box[3] / height, NORM_DIGITS)],
        "score": round(min(1.0, max(0.0, score)), SCORE_DIGITS),
        "label": None if kind == "face" else (str(raw.get("label")) if raw.get("label") is not None else None),
        "engine": engine["id"], "model": engine["model"], "model_sha256": engine["model_sha256"],
    }
    if kind == "text":
        det["lang"] = str(raw.get("lang") or "")
        det["words"] = int(raw.get("words") or 1)
    if kind == "face" and isinstance(raw.get("landmarks"), list):
        det["landmarks"] = [[round(min(max(p[0] * sx, 0.0), width), 1), round(min(max(p[1] * sy, 0.0), height), 1)]
                            for p in raw["landmarks"] if isinstance(p, (list, tuple)) and len(p) == 2]
    return det


def build_document(*, kind: str, source: Dict[str, Any], sampling: Dict[str, Any], engine: Dict[str, Any],
                   parameters: Dict[str, Any], frames: List[Dict[str, Any]], skill_version: str, processing_seconds: float,
                   warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    n_det = sum(len(f["detections"]) for f in frames)
    return {
        "schema": DETECTIONS_SCHEMA,
        "kind": kind,
        "source": source,
        "frame_size": {"width": source["width"], "height": source["height"]},
        "sampling": sampling,
        "engine": engine,
        "parameters": parameters,
        "frames": frames,
        "counts": {"frames": len(frames), "frames_with_detections": sum(1 for f in frames if f["detections"]),
                   "detections": n_det},
        "skill": "detection-skill",
        "skill_version": skill_version,
        "tool": TOOLS[kind],
        "created_at": utc_now(),
        "processing_seconds": round(processing_seconds, 3),
        "warnings": list(warnings or []),
    }


# ---- validation --------------------------------------------------------------------------------
class ValidationReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def check_source(src: Any, rep: ValidationReport, expected_fingerprint: Optional[str] = None) -> None:
    if not isinstance(src, dict):
        rep.errors.append("source must be an object")
        return
    for k in REQUIRED_SOURCE:
        if k not in src:
            rep.errors.append(f"source.{k} is required")
    fp = src.get("fingerprint")
    if not isinstance(fp, str) or not FINGERPRINT_RE.match(fp):
        rep.errors.append("source.fingerprint must be 'sha256:<64 hex>'")
    elif expected_fingerprint and fp != expected_fingerprint:
        rep.errors.append("source.fingerprint does not match the input file")
    fn = src.get("filename")
    if not isinstance(fn, str) or not fn:
        rep.errors.append("source.filename must be a non-empty string")
    elif "/" in fn or "\\" in fn:
        rep.errors.append("source.filename must be a bare file name, not a path")
    sb = src.get("size_bytes")
    if not isinstance(sb, int) or isinstance(sb, bool) or sb < 0:
        rep.errors.append("source.size_bytes must be a non-negative integer")
    mt = src.get("media_type")
    if mt not in ("video", "image"):
        rep.errors.append("source.media_type must be 'video' or 'image'")
    d = src.get("duration")
    if mt == "video" and (not _num(d) or d <= 0):
        rep.errors.append("source.duration must be a positive number for a video")
    if mt == "image" and d is not None:
        rep.errors.append("source.duration must be null for an image")
    for k in ("width", "height"):
        v = src.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            rep.errors.append(f"source.{k} must be a positive integer")


def validate_detections(doc: Any, expected_fingerprint: Optional[str] = None) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("detections document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != DETECTIONS_SCHEMA:
        rep.errors.append(f"schema must be {DETECTIONS_SCHEMA}, got {doc['schema']!r}")
    kind = doc["kind"]
    if kind not in KINDS:
        rep.errors.append(f"kind must be one of {KINDS}")
        return rep
    _check_run_fields(doc, rep, TOOLS[kind])
    check_source(doc["source"], rep, expected_fingerprint)
    src = doc["source"] if isinstance(doc["source"], dict) else {}
    fs = doc["frame_size"]
    width = height = None
    if not isinstance(fs, dict) or not all(isinstance(fs.get(k), int) and not isinstance(fs.get(k), bool) and fs[k] > 0
                                           for k in ("width", "height")):
        rep.errors.append("frame_size must be {width, height} positive integers")
    else:
        width, height = fs["width"], fs["height"]
        if (src.get("width"), src.get("height")) != (width, height):
            rep.errors.append("frame_size must equal source width/height (boxes are in source pixels)")
    is_image = src.get("media_type") == "image"
    duration = src.get("duration") if _num(src.get("duration")) else None

    sm = doc["sampling"]
    if not isinstance(sm, dict) or sm.get("method") not in METHODS or not isinstance(sm.get("times"), list):
        rep.errors.append(f"sampling must be an object with method in {METHODS} and times []")
    elif (sm["method"] == "image") != is_image:
        rep.errors.append("sampling.method 'image' is for an image source and only for it")
    elif sm["method"] == "every" and (not _num(sm.get("every")) or sm["every"] <= 0):
        rep.errors.append("sampling.every must be a positive number when method is 'every'")
    elif sm["method"] == "scenes" and not isinstance(sm.get("scene_points"), list):
        rep.errors.append("sampling.scene_points must be a list when method is 'scenes'")

    eng = doc["engine"]
    sha = None
    if not isinstance(eng, dict):
        rep.errors.append("engine must be an object")
        eng = {}
    else:
        for k in REQUIRED_ENGINE:
            if k not in eng:
                rep.errors.append(f"engine.{k} is required")
        for k in ("id", "version", "model", "model_license"):
            if k in eng and (not isinstance(eng[k], str) or not eng[k]):
                rep.errors.append(f"engine.{k} must be a non-empty string")
        if eng.get("kind") not in (None, kind):
            rep.errors.append("engine.kind must equal kind")
        sha = eng.get("model_sha256")
        if not isinstance(sha, str) or not SHA_RE.match(sha):
            rep.errors.append("engine.model_sha256 must be 64 lowercase hex")
        files = eng.get("files")
        if not isinstance(files, list) or not files or any(
                not isinstance(f, dict) or not isinstance(f.get("file"), str) or not SHA_RE.match(str(f.get("sha256", "")))
                for f in files):
            rep.errors.append("engine.files must list every model file with file and sha256")
        if eng.get("execution_mode") not in ("local", "remote"):
            rep.errors.append("engine.execution_mode must be 'local' or 'remote'")

    par = doc["parameters"]
    min_score = 0.0
    if not isinstance(par, dict) or not _num(par.get("min_score")) or not 0 <= par["min_score"] <= 1:
        rep.errors.append("parameters.min_score must be a number in [0, 1]")
    else:
        min_score = float(par["min_score"])

    frames = doc["frames"]
    n_det = n_with = 0
    if not isinstance(frames, list):
        rep.errors.append("frames must be a list")
        frames = []
    prev_t: Optional[float] = None
    for i, fr in enumerate(frames):
        w = f"frames[{i}]"
        if not isinstance(fr, dict) or "t" not in fr or not isinstance(fr.get("detections"), list):
            rep.errors.append(f"{w}: must be an object with index, t, detections")
            continue
        if fr.get("index") != i:
            rep.errors.append(f"{w}: index must be {i}")
        t = fr["t"]
        if is_image:
            if t is not None:
                rep.errors.append(f"{w}: t must be null for an image")
        elif not _num(t) or t < 0:
            rep.errors.append(f"{w}: t must be a number >= 0")
        else:
            if duration is not None and t > duration + TIME_TOLERANCE:
                rep.errors.append(f"{w}: t {t} is past the duration {duration}")
            if prev_t is not None and t <= prev_t:
                rep.errors.append(f"{w}: frames must be in strictly increasing time order")
            prev_t = t
        if fr["detections"]:
            n_with += 1
        for j, d in enumerate(fr["detections"]):
            n_det += 1
            _check_det(d, f"{w}.detections[{j}]", kind, t, width, height, min_score, eng, rep)
    c = doc["counts"]
    if not isinstance(c, dict) or c.get("frames") != len(frames) or c.get("detections") != n_det or c.get("frames_with_detections") != n_with:
        rep.errors.append(f"counts must be {{frames: {len(frames)}, frames_with_detections: {n_with}, detections: {n_det}}}")
    if isinstance(sm, dict) and isinstance(sm.get("times"), list) and not is_image:
        if [f.get("t") for f in frames if isinstance(f, dict)] != sm["times"]:
            rep.errors.append("sampling.times must list the frame times in order")
    scan_leakage(doc, "", rep)
    return rep


def _check_det(d: Any, w: str, kind: str, t: Any, width: Optional[int], height: Optional[int], min_score: float,
               eng: Dict[str, Any], rep: ValidationReport) -> None:
    if not isinstance(d, dict) or any(k not in d for k in REQUIRED_DET):
        rep.errors.append(f"{w}: needs {', '.join(REQUIRED_DET)}")
        return
    if d["t"] != t:
        rep.errors.append(f"{w}: t must equal the frame's t")
    if d["kind"] != kind:
        rep.errors.append(f"{w}: kind must be {kind}")
    box, norm = d["box"], d["box_norm"]
    if not (isinstance(box, list) and len(box) == 4 and all(_num(v) for v in box)):
        rep.errors.append(f"{w}: box must be [x, y, w, h] numbers")
        return
    if box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0:
        rep.errors.append(f"{w}: box needs x, y >= 0 and w, h > 0")
    if width and height:
        if box[0] + box[2] > width + BOX_TOLERANCE or box[1] + box[3] > height + BOX_TOLERANCE:
            rep.errors.append(f"{w}: box reaches past the frame {width}x{height}")
        if not (isinstance(norm, list) and len(norm) == 4 and all(_num(v) for v in norm)):
            rep.errors.append(f"{w}: box_norm must be four numbers")
        else:
            want = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            if any(abs(a - b) > NORM_TOLERANCE for a, b in zip(norm, want)):
                rep.errors.append(f"{w}: box_norm does not equal box / frame_size")
    s = d["score"]
    if not _num(s) or not 0 <= s <= 1:
        rep.errors.append(f"{w}: score must be in [0, 1]")
    elif s < min_score - 1e-9:
        rep.errors.append(f"{w}: score {s} is below parameters.min_score {min_score}")
    if kind == "face":
        if d["label"] is not None:
            rep.errors.append(f"{w}: a face has label null (a face box is never an identity)")
        bad = sorted(IDENTITY_KEYS & set(d))
        if bad:
            rep.errors.append(f"{w}: identity keys are not allowed on a face: {bad}")
    elif not isinstance(d["label"], str) or not d["label"]:
        rep.errors.append(f"{w}: label must be a non-empty string ({'OCR text' if kind == 'text' else 'object class'})")
    if kind == "text" and (not isinstance(d.get("lang"), str) or not d["lang"]):
        rep.errors.append(f"{w}: lang is required on text")
    if d["engine"] != eng.get("id") or d["model"] != eng.get("model") or d["model_sha256"] != eng.get("model_sha256"):
        rep.errors.append(f"{w}: engine/model/model_sha256 must equal the document's engine")


def _check_run_fields(doc: Dict[str, Any], rep: ValidationReport, tool: str) -> None:
    if not isinstance(doc["created_at"], str) or not ISO_RE.match(doc["created_at"]):
        rep.errors.append("created_at must be an ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    if not _num(doc["processing_seconds"]) or doc["processing_seconds"] < 0:
        rep.errors.append("processing_seconds must be a non-negative number")
    if not isinstance(doc["warnings"], list) or not all(isinstance(w, str) for w in doc["warnings"]):
        rep.errors.append("warnings must be a list of strings")
    for k in ("skill", "skill_version", "tool"):
        if not isinstance(doc[k], str) or not doc[k]:
            rep.errors.append(f"{k} must be a non-empty string")
    if doc["tool"] != tool:
        rep.errors.append(f"tool must be {tool}")


def scan_leakage(node: Any, path: str, rep: ValidationReport) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                rep.errors.append(f"forbidden key {path}.{k}: documents must not carry commands or credentials")
            if k in MEASURED_TEXT_KEYS:
                continue  # OCR text is what the frame shows, not something the skill leaked
            scan_leakage(v, f"{path}.{k}", rep)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            scan_leakage(v, f"{path}[{i}]", rep)
    elif isinstance(node, str):
        if any(p.search(node) for p in CREDENTIAL_PATTERNS):
            rep.errors.append(f"credential-like value at {path}")
        elif any(p.match(node) for p in ABS_PATH_PATTERNS):
            rep.errors.append(f"absolute path leaked at {path}")
