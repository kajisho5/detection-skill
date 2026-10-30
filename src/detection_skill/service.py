"""FACES / TEXT / OBJECTS: image or video -> detections document (`detection-skill/detections@1`).

input check + fingerprint -> engine.prepare (model files and their sha256, before any frame is made) ->
frames (an image is its own frame; a video's frames come from ffmpeg-skill probe / scenes / look) -> engine in a
worker subprocess with a timeout that kills its whole process group -> boxes scaled to source pixels, clipped,
provenance on every box -> validated document. An engine object passed in (the test suite's fake) runs in a
thread with the same timeout.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .detections import build_detection, build_document, validate_detections
from .engines import DEFAULT_ENGINE, get_engine
from .engines.base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .engines.tesseract_engine import DEFAULT_PSM, VERTICAL_PSM, resolve_langs
from .engines.yolox_engine import COCO_CLASSES
from .errors import DetectionError
from .frames import (Runner, extract_frames, is_image, plan_at, plan_every, plan_scenes, probe_video, require,
                     subprocess_runner)
from .models import resolve_models_dir

WORKER_MODULE = "detection_skill.engines.worker"
DEFAULT_TIMEOUT = 3600.0
DEFAULT_EVERY = 1.0
DEFAULT_MAX_FRAMES = 2000
DEFAULT_MIN_SCORE = {"face": 0.6, "text": 0.3, "object": 0.3}
LEVELS = ("line", "word")
CHILD_ENV_KEYS = ("PATH", "HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "LC_ALL", "PYTHONUTF8")


def fingerprint_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def parse_classes(classes: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for c in classes or []:
        for name in str(c).split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name not in COCO_CLASSES:
                raise DetectionError("INVALID_ARGUMENT", f"unknown class {name!r} (COCO's 80 class names)", list(COCO_CLASSES))
            if name not in out:
                out.append(name)
    return out


def detect(kind: str, media: str, *, every: Optional[float] = None, at: Optional[List[float]] = None, scenes: bool = False,
           min_score: Optional[float] = None, max_frames: int = DEFAULT_MAX_FRAMES, models_dir: Optional[str] = None,
           model: Optional[str] = None, lang: Optional[str] = None, vertical: bool = False, psm: Optional[int] = None,
           level: str = "line", classes: Optional[List[str]] = None, tessdata_dir: Optional[str] = None,
           ffmpeg_skill: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT, engine: Optional[DetectionEngine] = None,
           runner: Runner = subprocess_runner) -> Dict[str, Any]:
    t0 = time.time()
    if kind not in DEFAULT_ENGINE:
        raise DetectionError("INVALID_ARGUMENT", f"kind must be one of {sorted(DEFAULT_ENGINE)}")
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE[kind]
    if not isinstance(min_score, (int, float)) or not 0.0 <= float(min_score) <= 1.0:
        raise DetectionError("INVALID_ARGUMENT", "--min-score must be in [0, 1]")
    if timeout <= 0:
        raise DetectionError("INVALID_ARGUMENT", "--timeout must be > 0")
    if not isinstance(max_frames, int) or max_frames < 1:
        raise DetectionError("INVALID_ARGUMENT", "--max-frames must be >= 1")
    chosen = [n for n, v in (("--every", every is not None), ("--at", bool(at)), ("--scenes", bool(scenes))) if v]
    if len(chosen) > 1:
        raise DetectionError("INVALID_ARGUMENT", f"choose one sampling: {' / '.join(chosen)}")
    if not os.path.isfile(media):
        raise DetectionError("INPUT_NOT_FOUND", f"input not found: {os.path.basename(media)}")
    image = is_image(media)
    if image and chosen:
        raise DetectionError("INVALID_ARGUMENT", f"{chosen[0]} is for a video; an image is one frame")

    params: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {"min_score": float(min_score)}
    if kind == "text":
        if level not in LEVELS:
            raise DetectionError("INVALID_ARGUMENT", f"--level must be one of {LEVELS}")
        langs = resolve_langs(lang, vertical)
        use_psm = psm if psm is not None else (VERTICAL_PSM if vertical else DEFAULT_PSM)
        if not isinstance(use_psm, int) or not 0 <= use_psm <= 13:
            raise DetectionError("INVALID_ARGUMENT", "--psm must be 0..13")
        params.update(langs=langs, psm=use_psm, level=level)
        parameters.update(lang="+".join(langs), vertical=bool(vertical), psm=use_psm, level=level)
    elif kind == "object":
        wanted = parse_classes(classes)
        params["classes"] = wanted
        parameters["classes"] = wanted or None
    elif classes or lang or vertical:
        raise DetectionError("INVALID_ARGUMENT", "--classes / --lang / --vertical do not apply to faces")

    eng = engine if engine is not None else get_engine(DEFAULT_ENGINE[kind])
    if eng.kind != kind:
        raise DetectionError("INVALID_ARGUMENT", f"engine {eng.id} detects {eng.kind}, not {kind}")
    prepared = eng.prepare(models_dir=resolve_models_dir(models_dir), model=model, tessdata_dir=tessdata_dir, params=params)
    warnings: List[str] = list(prepared.warnings)
    fingerprint = fingerprint_file(media)

    run_dir = tempfile.mkdtemp(prefix="detection-")
    try:
        if image:
            frames = [{"index": 0, "t": None, "path": os.path.abspath(media)}]
            sampling: Dict[str, Any] = {"method": "image", "every": None, "times": [], "scene_points": None,
                                        "frames_tool": None, "ffmpeg_skill_version": None, "extracted_size": None}
            facts: Dict[str, Any] = {"duration": None, "width": None, "height": None}
        else:
            skill = require(ffmpeg_skill)
            facts = probe_video(skill, media, runner=runner)
            scene_points = None
            if scenes:
                times, scene_points = plan_scenes(skill, media, facts["duration"], facts["fps"], runner=runner, timeout=timeout)
                method, step = "scenes", None
            elif at:
                times, method, step = plan_at(list(at), facts["duration"], facts["fps"]), "at", None
            else:
                step = float(every if every is not None else DEFAULT_EVERY)
                times, method = plan_every(step, facts["duration"], facts["fps"]), "every"
            if len(times) > max_frames:
                raise DetectionError("INVALID_ARGUMENT", f"{len(times)} frames planned, more than --max-frames {max_frames}",
                                     "raise --every, or pass a larger --max-frames")
            frames = extract_frames(skill, media, times, facts["width"], run_dir, runner=runner, timeout=timeout)
            sampling = {"method": method, "every": step, "times": times, "scene_points": scene_points,
                        "frames_tool": "ffmpeg-skill/look", "ffmpeg_skill_version": skill.version, "extracted_size": None}
        ereq = EngineRequest(frames=frames, model_path=prepared.model_path, tessdata_dir=prepared.tessdata_dir,
                             tesseract=prepared.tesseract, min_score=float(min_score), params=params)
        t_engine = time.time()
        result = run_engine(eng, ereq, timeout, run_dir, injected=engine is not None)
        engine_seconds = round(time.time() - t_engine, 3)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    by_index = {int(f["index"]): f for f in result.frames}
    if set(by_index) != {f["index"] for f in frames}:
        raise DetectionError("INVALID_RESULT", "engine did not return exactly one result per frame")
    first = by_index[0]
    if image:
        facts["width"], facts["height"] = int(first["width"]), int(first["height"])
    width, height = int(facts["width"]), int(facts["height"])
    if not image:
        sizes = {(int(f["width"]), int(f["height"])) for f in result.frames}
        sampling["extracted_size"] = {"width": int(first["width"]), "height": int(first["height"])}
        if len(sizes) > 1:
            warnings.append(f"frames came back in {len(sizes)} sizes; each was scaled to the source size")
        if sizes != {(width, height)}:
            warnings.append(f"frames were extracted at {first['width']}x{first['height']}; boxes are scaled to the source "
                            f"{width}x{height}")
    engine_doc = {"id": eng.id, "kind": kind, "version": prepared.version, "runtime": eng.runtime, "model": prepared.model,
                  "model_sha256": prepared.model_sha256, "model_license": prepared.license, "engine_license": eng.license,
                  "files": prepared.files, "execution_mode": eng.execution_mode, "engine_seconds": engine_seconds,
                  "details": {k: v for k, v in result.runtime.items() if isinstance(v, (str, int, float, list)) or v is None}}
    out_frames: List[Dict[str, Any]] = []
    dropped = 0
    for fr in frames:
        raw = by_index[fr["index"]]
        fw, fh = float(raw["width"]), float(raw["height"])
        if fw <= 0 or fh <= 0:
            raise DetectionError("INVALID_RESULT", "engine reported a frame without a size")
        sx, sy = width / fw, height / fh
        dets = []
        for r in raw.get("detections") or []:
            d = build_detection(r, kind=kind, t=fr["t"], sx=sx, sy=sy, width=width, height=height, engine=engine_doc)
            if d is None:
                dropped += 1
            elif d["score"] >= min_score:
                dets.append(d)
        dets.sort(key=lambda d: (d["box"][1], d["box"][0]) if kind == "text" else (-d["score"], d["box"][0]))
        out_frames.append({"index": fr["index"], "t": fr["t"], "detections": dets})
    if dropped:
        warnings.append(f"{dropped} engine box(es) outside the frame or malformed were dropped")
    source = {"filename": os.path.basename(media), "fingerprint": fingerprint, "size_bytes": os.path.getsize(media),
              "media_type": "image" if image else "video", "duration": facts["duration"], "width": width, "height": height}
    if not image and facts.get("fps"):
        source["fps"] = facts["fps"]
    doc = build_document(kind=kind, source=source, sampling=sampling, engine=engine_doc, parameters=parameters,
                         frames=out_frames, skill_version=__version__, processing_seconds=time.time() - t0, warnings=warnings)
    rep = validate_detections(doc, expected_fingerprint=fingerprint)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "engine output failed detections validation; nothing returned", rep.errors[:10])
    return doc


def run_engine(engine: DetectionEngine, ereq: EngineRequest, timeout: float, run_dir: str, injected: bool = False) -> EngineResult:
    if injected:
        return _run_in_thread(engine, ereq, timeout)
    req_path = os.path.join(run_dir, "engine_request.json")
    out_path = os.path.join(run_dir, "engine_result.json")
    with open(req_path, "w", encoding="utf-8") as fh:
        json.dump({"engine_id": engine.id, "request": ereq.to_dict()}, fh, ensure_ascii=False)
    code, err = run_process_group([sys.executable, "-m", WORKER_MODULE, req_path, out_path], timeout)
    if code == 124:
        raise DetectionError("ENGINE_TIMEOUT", f"engine exceeded the timeout ({timeout:g}s) and was stopped", {"timeout": timeout})
    if not os.path.exists(out_path):
        tail = "\n".join(err.strip().splitlines()[-5:])
        raise DetectionError("ENGINE_FAILED", f"engine worker exited {code} without a result", tail)
    with open(out_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not payload.get("ok"):
        e = payload.get("error") or {}
        raise DetectionError(e.get("code", "ENGINE_FAILED"), e.get("message", "engine failed"), e.get("detail"))
    return EngineResult.from_dict(payload["result"])


def worker_env() -> Dict[str, str]:
    env = {k: os.environ[k] for k in CHILD_ENV_KEYS if k in os.environ}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


def run_process_group(argv: List[str], timeout: float) -> Tuple[int, str]:
    """Run argv in its own process group; kill the whole group on timeout (returns exit 124)."""
    kw: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "env": worker_env(), "shell": False}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(argv, **kw)
    try:
        _, err = proc.communicate(timeout=timeout)
        return proc.returncode, (err or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        return 124, ""


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_in_thread(engine: DetectionEngine, ereq: EngineRequest, timeout: float) -> EngineResult:
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = engine.detect(ereq)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise DetectionError("ENGINE_TIMEOUT", f"engine exceeded the timeout ({timeout:g}s)", {"timeout": timeout})
    if "error" in box:
        err = box["error"]
        if isinstance(err, DetectionError):
            raise err
        raise DetectionError("ENGINE_FAILED", f"{type(err).__name__}: {err}")
    return box["result"]


__all__ = ["detect", "run_engine", "run_process_group", "parse_classes", "Prepared"]
