"""Shots: PySceneDetect (BSD-3-Clause) on the frames ffmpeg-skill decoded, plus a blend test for gradual transitions
(ADR-003). The default shot engine.

Every frame of the video, in order, goes through PySceneDetect's ContentDetector (HSV + edge change against the
previous frame above `threshold`, 27 by default, with its flash filter and `min_scene_len`) and AdaptiveDetector
(the same score against a rolling average, ratio above `adaptive_threshold`, 3.0). Their cuts are frame accurate on
hard cuts. Neither detects a dissolve, so each frame is also run through the blend test of `shotcore.blend_runs` on
a 64 px wide copy; a run of blended frames is a gradual transition. PySceneDetect is used through its detector API
only: it never opens the video (`scenedetect` video backends are not used), so frames still come from ffmpeg-skill.
"""
from __future__ import annotations

import importlib.util
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..errors import DetectionError
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .shotcore import BLEND_DEFAULTS, blend_h, blend_runs, merge_boundaries
from .yunet_engine import read_image

INSTALL = "pip install 'detection-skill[shots]'  (scenedetect, opencv-python-headless)"
DEFAULTS = {"threshold": 27.0, "adaptive_threshold": 3.0, "min_content_val": 15.0, "min_shot_frames": 15}
BLEND_WIDTH = 64


def _scenedetect_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("scenedetect")
    except Exception:  # noqa: BLE001 - PackageNotFoundError or no metadata
        return None


class PySceneDetectEngine(DetectionEngine):
    id = "pyscenedetect"
    kind = "shot"
    description = "PySceneDetect ContentDetector + AdaptiveDetector (cuts) and a blend test (gradual), CPU"
    runtime = "scenedetect + opencv-python-headless"
    license = "BSD-3-Clause"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        for mod in ("scenedetect", "cv2", "numpy"):
            if importlib.util.find_spec(mod) is None:
                raise DetectionError("ENGINE_UNAVAILABLE", f"{mod} is not installed (shots need PySceneDetect and OpenCV)", INSTALL)
        if model:
            raise DetectionError("INVALID_ARGUMENT", "--model is for --engine transnetv2; PySceneDetect uses no model file")
        version = _scenedetect_version() or "unknown"
        return Prepared(model=f"PySceneDetect {version} ContentDetector+AdaptiveDetector, blend test", model_sha256="",
                        files=[], license="none (no model file)", version=version)

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
            import numpy as np
            from scenedetect.detectors import AdaptiveDetector, ContentDetector
            from scenedetect.stats_manager import StatsManager
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"PySceneDetect / OpenCV not importable: {exc}", INSTALL)
        p = dict(DEFAULTS)
        p.update({k: v for k, v in (req.params or {}).items() if v is not None})
        fps = float(p["fps"])
        min_len = int(p["min_shot_frames"])
        h = int(p.get("blend_half_window") or blend_h(fps))
        far_min = float(p.get("blend_far_min", BLEND_DEFAULTS["far_min"]))
        ratio_max = float(p.get("blend_ratio_max", BLEND_DEFAULTS["ratio_max"]))
        stats = StatsManager()
        content = ContentDetector(threshold=float(p["threshold"]), min_scene_len=min_len)
        adaptive = AdaptiveDetector(adaptive_threshold=float(p["adaptive_threshold"]), min_scene_len=min_len,
                                    min_content_val=float(p["min_content_val"]))
        content.stats_manager = stats
        adaptive.stats_manager = stats
        found: Dict[int, List[str]] = {}
        ring: Deque[Tuple[int, Any]] = deque(maxlen=2 * h + 1)
        passing: List[Tuple[int, float, float]] = []
        frames = sorted(req.frames, key=lambda f: int(f["index"]))
        width = height = 0
        last = -1
        for fr in frames:
            k = int(fr["index"])
            if k != last + 1:
                raise DetectionError("INVALID_RESULT", "shot frames must be consecutive from frame 0")
            last = k
            img = read_image(fr["path"])
            height, width = img.shape[:2]
            for c in content.process_frame(k, img):
                found.setdefault(int(c), []).append("content")
            for c in adaptive.process_frame(k, img):
                found.setdefault(int(c), []).append("adaptive")
            sh = max(2, int(round(height * BLEND_WIDTH / float(width))))
            ring.append((k, cv2.resize(img, (BLEND_WIDTH, sh), interpolation=cv2.INTER_AREA).astype(np.float32)))
            if len(ring) == ring.maxlen:
                (_ka, a), (km, m), (_kb, b) = ring[0], ring[h], ring[-1]
                far = float(np.mean(np.abs(a - b)))
                if far >= far_min:
                    ratio = float(np.mean(np.abs(m - (a + b) / 2.0))) / far
                    if ratio <= ratio_max:
                        passing.append((km, far, ratio))
        for c in (content.post_process(last) or []):
            found.setdefault(int(c), []).append("content")
        key_ratio = adaptive.get_metrics()[-1]
        cuts = []
        for frame, who in sorted(found.items()):
            cv = stats.get_metrics(frame, [ContentDetector.FRAME_SCORE_KEY])[0] or 0.0
            ar = stats.get_metrics(frame, [key_ratio])[0] or 0.0
            score = max(min(1.0, cv / (2.0 * float(p["threshold"]))), min(1.0, ar / (2.0 * float(p["adaptive_threshold"]))))
            cuts.append({"frame": frame, "kind": "cut", "score": round(score, 4),
                         "metric": {"detectors": sorted(set(who)), "content_val": round(float(cv), 3), "adaptive_ratio": round(float(ar), 3)}})
        gradual = blend_runs(passing, h, min_core=int(p.get("blend_min_core", BLEND_DEFAULTS["min_core"])), frame_count=last + 1)
        boundaries, merged = merge_boundaries(cuts, gradual, min_shot_frames=min_len, absorb=BLEND_DEFAULTS["absorb"])
        return EngineResult(frames=[], runtime={"scenedetect": _scenedetect_version(), "frame_width": width, "frame_height": height},
                            data={"boundaries": boundaries, "frames": last + 1, "merged": merged,
                                  "parameters": {"threshold": float(p["threshold"]), "adaptive_threshold": float(p["adaptive_threshold"]),
                                                 "min_content_val": float(p["min_content_val"]), "min_shot_frames": min_len,
                                                 "blend_half_window": h, "blend_far_min": far_min, "blend_ratio_max": ratio_max,
                                                 "blend_width": BLEND_WIDTH}})
