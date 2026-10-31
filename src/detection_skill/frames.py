"""Frames, and the only media boundary: ffmpeg-skill's own scripts, run as
`python3 <ffmpeg-skill>/scripts/<tool>.py ... --json` (ADR-001). This skill never runs ffmpeg or ffprobe itself
and never decodes video: `probe` gives duration and size, `scenes` gives change points, `look --at` writes the
frames as PNG at the source's own width with no timecode burnt in.

ffmpeg-skill is found, in order: --ffmpeg-skill, $DETECTION_FFMPEG_SKILL_DIR, $FFMPEG_SKILL_DIR,
~/.claude/skills/ffmpeg-skill, ./vendor/ffmpeg-skill, ../ffmpeg-skill (the same order as slide-sync-skill).

An image file (png, jpg, ...) is its own single frame and needs no ffmpeg-skill.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import DetectionError

ENV_DIRS = ("DETECTION_FFMPEG_SKILL_DIR", "FFMPEG_SKILL_DIR")
REQUIRED_TOOLS = ("probe", "look", "scenes")
SUPPORTED_MAJORS = (2, 3)
VERSION_RANGE = ">=2.0.0 <4.0.0"
# ffmpeg-skill 3.0.0 dropped Python 3.9: its scripts run under this interpreter (sys.executable), so a 3.x
# checkout needs Python 3.10+ here too. On Python 3.9, use an ffmpeg-skill 2.x checkout (ADR-002).
PYTHON_FLOOR_BY_MAJOR = {3: (3, 10)}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
LOOK_BATCH = 40
TIME_DIGITS = 3

Runner = Callable[[List[str], float], Tuple[int, str, str]]


@dataclass
class FfmpegSkill:
    root: str
    version: str
    tools: List[str]

    def script(self, tool: str) -> str:
        return os.path.join(self.root, "scripts", f"{tool}.py")

    @property
    def major(self) -> Optional[int]:
        try:
            return int(self.version.split(".")[0])
        except ValueError:
            return None


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _candidate(path: str) -> Optional[FfmpegSkill]:
    path = os.path.expanduser(path)
    scripts = os.path.join(path, "scripts")
    if not os.path.isfile(os.path.join(scripts, "look.py")):
        return None
    version = "unknown"
    try:
        with open(os.path.join(path, "package.json"), encoding="utf-8") as fh:
            version = str(json.load(fh).get("version", "unknown"))
    except (OSError, ValueError):
        pass
    tools = sorted(f[:-3] for f in os.listdir(scripts) if f.endswith(".py") and not f.startswith("_"))
    return FfmpegSkill(os.path.realpath(path), version, tools)


def candidates(explicit: Optional[str] = None) -> List[str]:
    out = [explicit] if explicit else []
    out += [os.environ[e] for e in ENV_DIRS if os.environ.get(e)]
    out += [os.path.expanduser("~/.claude/skills/ffmpeg-skill"), os.path.join(os.getcwd(), "vendor", "ffmpeg-skill"),
            os.path.join(os.getcwd(), "..", "ffmpeg-skill")]
    return out


def locate(explicit: Optional[str] = None) -> Optional[FfmpegSkill]:
    for c in candidates(explicit):
        found = _candidate(c)
        if found:
            return found
    return None


def require(explicit: Optional[str] = None) -> FfmpegSkill:
    skill = locate(explicit)
    if skill is None:
        raise DetectionError("FFMPEG_SKILL_NOT_FOUND", "ffmpeg-skill not found (frames of a video come from its look/scenes)",
                             "pass --ffmpeg-skill DIR or set $DETECTION_FFMPEG_SKILL_DIR; see `detection doctor`")
    if skill.major not in SUPPORTED_MAJORS:
        raise DetectionError("FFMPEG_SKILL_NOT_FOUND", f"ffmpeg-skill {skill.version} is outside {VERSION_RANGE}")
    floor = python_floor(skill)
    if floor:
        raise DetectionError("FFMPEG_SKILL_NOT_FOUND",
                             f"ffmpeg-skill {skill.version} needs Python {floor} or later (this is {sys.version.split()[0]})",
                             "run detection under Python 3.10+, or use an ffmpeg-skill 2.x checkout on Python 3.9")
    missing = [t for t in REQUIRED_TOOLS if t not in skill.tools]
    if missing:
        raise DetectionError("FFMPEG_SKILL_NOT_FOUND", f"ffmpeg-skill {skill.version} lacks tool(s): {', '.join(missing)}")
    return skill


def python_floor(skill: FfmpegSkill, version_info: Optional[Tuple[int, ...]] = None) -> Optional[str]:
    """The Python version this ffmpeg-skill major needs when the running interpreter is older, else None."""
    floor = PYTHON_FLOOR_BY_MAJOR.get(skill.major) if skill.major is not None else None
    vi = tuple(version_info if version_info is not None else sys.version_info)[:2]
    return ".".join(map(str, floor)) if floor and vi < floor else None


def subprocess_runner(argv: List[str], timeout: float) -> Tuple[int, str, str]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, shell=False,
                       stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


def run_tool(skill: FfmpegSkill, tool: str, args: List[str], *, timeout: float = 1800.0,
             runner: Runner = subprocess_runner) -> Dict[str, Any]:
    """Run one ffmpeg-skill tool with --json and return its document."""
    argv = [sys.executable, skill.script(tool)] + list(args) + ["--json"]
    try:
        rc, out, err = runner(argv, timeout)
    except subprocess.TimeoutExpired:
        raise DetectionError("ENGINE_TIMEOUT", f"ffmpeg-skill {tool} did not finish within {timeout:.0f}s")
    last = (err.strip().splitlines() or [""])[-1]
    try:
        doc = json.loads(out) if out.strip() else None
    except ValueError:
        doc = None
    if rc != 0:
        msg = ((doc or {}).get("error") or {}).get("message") if isinstance(doc, dict) else None
        raise DetectionError("FRAME_EXTRACTION_FAILED", f"ffmpeg-skill {tool} exited {rc}", msg or last)
    if not isinstance(doc, dict):
        raise DetectionError("FRAME_EXTRACTION_FAILED", f"ffmpeg-skill {tool} did not print a JSON object", last)
    return doc


def engine_doctor(skill: FfmpegSkill, *, timeout: float = 60.0, runner: Runner = subprocess_runner) -> Dict[str, Any]:
    """ffmpeg-skill's own doctor (`scripts/_contract.py doctor --json`). Never raises."""
    argv = [sys.executable, os.path.join(skill.root, "scripts", "_contract.py"), "doctor", "--json"]
    try:
        _rc, out, _err = runner(argv, timeout)
        doc = json.loads(out)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {"ok": False, "ffmpeg": None, "tools": {}, "detail": "ffmpeg-skill doctor did not answer"}
    if not isinstance(doc, dict):
        doc = {}
    tools = {t: ((doc.get("tools") or {}).get(t) or {}).get("usable") for t in REQUIRED_TOOLS}
    return {"ok": bool(doc.get("ok")) and all(v in ("yes", None) for v in tools.values()), "ffmpeg": doc.get("ffmpeg"),
            "tools": tools, "detail": None}


# ---- video facts and sampling --------------------------------------------------------------------
def probe_video(skill: FfmpegSkill, video: str, *, runner: Runner = subprocess_runner, timeout: float = 300.0) -> Dict[str, Any]:
    doc = run_tool(skill, "probe", [video], timeout=timeout, runner=runner)
    v = doc.get("video")
    if not isinstance(v, dict) or not v.get("width") or not v.get("height"):
        raise DetectionError("UNSUPPORTED_MEDIA", f"{os.path.basename(video)} has no video stream (ffmpeg-skill probe)")
    w, h = int(v["width"]), int(v["height"])
    if v.get("rotation") in (90, -90, 270, -270):
        w, h = h, w
    duration = doc.get("duration") or v.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = None
    if not duration or duration <= 0 or not math.isfinite(duration):
        raise DetectionError("UNSUPPORTED_MEDIA", f"{os.path.basename(video)}: duration unknown or zero")
    fps = v.get("fps")
    try:
        fps = float(fps) if fps else None
    except (TypeError, ValueError):
        fps = None
    return {"duration": duration, "width": w, "height": h, "fps": fps}


def last_time(duration: float, fps: Optional[float]) -> float:
    """The latest time a frame is asked for: one frame before the end, so `look` always has a frame to write."""
    step = 1.0 / fps if fps and fps > 0 else 0.04
    return max(0.0, round(duration - step, TIME_DIGITS))


def plan_every(every: float, duration: float, fps: Optional[float]) -> List[float]:
    if not every or every <= 0:
        raise DetectionError("INVALID_ARGUMENT", "--every must be > 0 seconds")
    end = last_time(duration, fps)
    out: List[float] = []
    k = 0
    while True:
        t = round(k * every, TIME_DIGITS)
        if t > end + 1e-9:
            break
        out.append(t)
        k += 1
    return out


def plan_at(times: List[float], duration: float, fps: Optional[float]) -> List[float]:
    end = last_time(duration, fps)
    out: List[float] = []
    for t in times:
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise DetectionError("INVALID_ARGUMENT", f"--at {t!r}: times must be finite and >= 0")
        if t > duration:
            raise DetectionError("INVALID_ARGUMENT", f"--at {t:g} is beyond the duration ({duration:.3f}s)")
        out.append(round(min(float(t), end), TIME_DIGITS))
    return sorted(set(out))


def plan_scenes(skill: FfmpegSkill, video: str, duration: float, fps: Optional[float], *, runner: Runner = subprocess_runner,
                timeout: float = 1800.0) -> Tuple[List[float], List[Dict[str, Any]]]:
    """One sample per scene from ffmpeg-skill scenes: the scene's midpoint (away from the cut, where frames
    can be blended)."""
    doc = run_tool(skill, "scenes", [video], timeout=timeout, runner=runner)
    scenes = doc.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise DetectionError("FRAME_EXTRACTION_FAILED", "ffmpeg-skill scenes returned no scenes")
    end = last_time(duration, fps)
    points: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes):
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            raise DetectionError("FRAME_EXTRACTION_FAILED", "ffmpeg-skill scenes returned a scene without start/end")
        t = round(min(max(0.0, (a + b) / 2.0), end), TIME_DIGITS)
        points.append({"scene": int(s.get("index", i)), "start": round(a, TIME_DIGITS), "end": round(b, TIME_DIGITS), "t": t})
    times = sorted({p["t"] for p in points})
    return times, points


def extract_frames(skill: FfmpegSkill, video: str, times: List[float], width: int, out_dir: str, *,
                   runner: Runner = subprocess_runner, timeout: float = 1800.0) -> List[Dict[str, Any]]:
    """`look --at` in batches: PNG frames at the source width (`--width`), no timecode (`--no-timecode`).
    Returns [{index, t, path}] in time order."""
    frames: List[Dict[str, Any]] = []
    for b in range(0, len(times), LOOK_BATCH):
        batch = times[b:b + LOOK_BATCH]
        base = os.path.join(out_dir, f"b{b // LOOK_BATCH:04d}")
        args = [video]
        for t in batch:
            args += ["--at", f"{t:.{TIME_DIGITS}f}"]
        args += ["-o", base, "--no-timecode", "--width", str(int(width))]
        doc = run_tool(skill, "look", args, timeout=timeout, runner=runner)
        outs = doc.get("outputs")
        if not isinstance(outs, list) or len(outs) != len(batch):
            outs = [f"{base}_{t:.{TIME_DIGITS}f}s.png" for t in batch]
        for t, p in zip(batch, outs):
            path = p if os.path.isabs(p) else os.path.join(out_dir, os.path.basename(p))
            if not os.path.isfile(path):
                raise DetectionError("FRAME_EXTRACTION_FAILED", f"ffmpeg-skill look wrote no frame for t={t:.3f}s")
            frames.append({"index": len(frames), "t": t, "path": path})
    return frames
