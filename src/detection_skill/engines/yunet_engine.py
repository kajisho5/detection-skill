"""Faces: YuNet (OpenCV Zoo, MIT) through OpenCV's `FaceDetectorYN` on CPU (ADR-001).

YuNet finds faces of roughly 10 px to 300 px. A frame larger than SECOND_SCALE_ABOVE px on its long side is
also run downscaled to SECOND_SCALE_SIDE px, so a large face in a 1080p/4K frame is found too; the two passes
are merged by non-maximum suppression. A box is a measured face region with a score and five landmarks. It is
never an identity: no embedding, no recognition, no name.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional

from ..errors import DetectionError
from ..models import KNOWN_MODELS, MODEL_LICENSES, describe_file, find_model
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .boxes import nms

SECOND_SCALE_ABOVE = 960
SECOND_SCALE_SIDE = 640
NMS_IOU = 0.3
INSTALL = "pip install 'detection-skill[faces]'  (opencv-python-headless)"


def read_image(path: str) -> Any:
    import cv2
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)  # works with non-ASCII paths on Windows, unlike cv2.imread
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise DetectionError("UNSUPPORTED_MEDIA", f"cannot decode image {os.path.basename(path)}")
    return img


class YunetEngine(DetectionEngine):
    id = "yunet"
    kind = "face"
    description = "YuNet face detector (OpenCV Zoo) via cv2.FaceDetectorYN, CPU"
    runtime = "opencv-python-headless"
    license = "MIT"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        if importlib.util.find_spec("cv2") is None:
            raise DetectionError("ENGINE_UNAVAILABLE", "OpenCV (cv2) is not installed", INSTALL)
        path = find_model("face", models_dir, model)
        d = describe_file(path)
        warnings = [] if d["known"] else [f"model {d['file']}: sha256 not in the skill's table of known files (recorded as measured)"]
        known = KNOWN_MODELS.get(d["sha256"], {})
        return Prepared(model=known.get("model") or d["file"], model_sha256=d["sha256"], files=[d], license=MODEL_LICENSES["face"],
                        version=_cv2_version(), model_path=path, warnings=warnings)

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"OpenCV is not importable: {exc}", INSTALL)
        if not hasattr(cv2, "FaceDetectorYN"):
            raise DetectionError("ENGINE_UNAVAILABLE", "this OpenCV has no FaceDetectorYN (needs >= 4.5.4)", INSTALL)
        score = max(0.0, float(req.min_score or 0.0))
        det = cv2.FaceDetectorYN.create(req.model_path, "", (320, 320), score, NMS_IOU, 5000)
        frames: List[Dict[str, Any]] = []
        for fr in req.frames:
            img = read_image(fr["path"])
            h, w = img.shape[:2]
            found = _detect(det, img, 1.0)
            side = max(w, h)
            if side > SECOND_SCALE_ABOVE:
                s = SECOND_SCALE_SIDE / float(side)
                small = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
                found += _detect(det, small, 1.0 / s)
            frames.append({"index": fr["index"], "width": w, "height": h, "detections": nms(found, NMS_IOU)})
        return EngineResult(frames=frames, runtime={"opencv": _cv2_version()})


def _detect(det: Any, img: Any, scale_back: float) -> List[Dict[str, Any]]:
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _n, faces = det.detect(img)
    out: List[Dict[str, Any]] = []
    if faces is None:
        return out
    for f in faces:
        vals = [float(v) * scale_back for v in f[:14]]
        out.append({"box": vals[0:4], "score": float(f[14]), "label": None,
                    "landmarks": [[vals[4 + 2 * i], vals[5 + 2 * i]] for i in range(5)]})
    return out


def _cv2_version() -> str:
    try:
        from importlib.metadata import version
        for dist in ("opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
            try:
                return version(dist)
            except Exception:  # noqa: BLE001 - PackageNotFoundError, try the next name
                continue
    except ImportError:  # pragma: no cover
        pass
    try:
        import cv2
        return str(cv2.__version__)
    except ImportError:
        return "unknown"
