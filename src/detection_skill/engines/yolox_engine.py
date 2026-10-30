"""Objects: YOLOX (Megvii, Apache-2.0) ONNX through onnxruntime on CPU (ADR-001), COCO's 80 classes.

Pre- and post-processing follow the YOLOX repository's ONNXRuntime demo: letterbox into the model's input size
(padding 114, BGR, 0-255, no normalisation), decode the grid outputs with strides 8/16/32, score = objectness x
class probability, then class-aware non-maximum suppression (IoU 0.45). The input size is read from the model.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Optional

from ..errors import DetectionError
from ..models import KNOWN_MODELS, MODEL_LICENSES, describe_file, find_model
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .boxes import nms
from .yunet_engine import read_image

NMS_IOU = 0.45
MAX_CANDIDATES = 1000  # highest-scoring boxes per frame that enter NMS
STRIDES = (8, 16, 32)
PAD_VALUE = 114
INSTALL = "pip install 'detection-skill[objects]'  (onnxruntime, opencv-python-headless)"

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class YoloxEngine(DetectionEngine):
    id = "yolox"
    kind = "object"
    description = "YOLOX (Megvii) COCO object detector, ONNX via onnxruntime, CPU"
    runtime = "onnxruntime"
    license = "Apache-2.0"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        missing = [m for m in ("onnxruntime", "cv2", "numpy") if importlib.util.find_spec(m) is None]
        if missing:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not installed: {', '.join(missing)}", INSTALL)
        path = find_model("object", models_dir, model)
        d = describe_file(path)
        warnings = [] if d["known"] else [f"model {d['file']}: sha256 not in the skill's table of known files (recorded as measured)"]
        known = KNOWN_MODELS.get(d["sha256"], {})
        return Prepared(model=known.get("model") or d["file"], model_sha256=d["sha256"], files=[d], license=MODEL_LICENSES["object"],
                        version=_ort_version(), model_path=path, warnings=warnings, extra={"classes": "coco80"})

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not importable: {exc}", INSTALL)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(req.model_path, sess_options=so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = inp.shape
        try:
            ih, iw = int(shape[2]), int(shape[3])
        except (TypeError, ValueError, IndexError):
            ih = iw = 640
        wanted = set(req.params.get("classes") or [])
        min_score = float(req.min_score or 0.0)
        grids, strides = _grids(np, ih, iw)
        frames: List[Dict[str, Any]] = []
        for fr in req.frames:
            img = read_image(fr["path"])
            h, w = img.shape[:2]
            r = min(ih / h, iw / w)
            nh, nw = max(1, int(h * r)), max(1, int(w * r))
            padded = np.full((ih, iw, 3), PAD_VALUE, dtype=np.uint8)
            padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
            out = sess.run(None, {inp.name: x})[0][0]
            if out.shape[0] != grids.shape[0]:
                raise DetectionError("ENGINE_FAILED", f"unexpected YOLOX output shape {tuple(out.shape)} for input {iw}x{ih}")
            xy = (out[:, :2] + grids) * strides
            wh = np.exp(out[:, 2:4]) * strides
            scores = out[:, 4:5] * out[:, 5:]
            cls = scores.argmax(1)
            best = scores[np.arange(scores.shape[0]), cls]
            keep = np.where(best >= min_score)[0]
            keep = keep[np.argsort(-best[keep])][:MAX_CANDIDATES]
            dets: List[Dict[str, Any]] = []
            for i in keep:
                name = COCO_CLASSES[int(cls[i])] if int(cls[i]) < len(COCO_CLASSES) else f"class_{int(cls[i])}"
                if wanted and name not in wanted:
                    continue
                cx, cy, bw, bh = (float(v) / r for v in (xy[i, 0], xy[i, 1], wh[i, 0], wh[i, 1]))
                dets.append({"box": [cx - bw / 2, cy - bh / 2, bw, bh], "score": float(best[i]), "label": name})
            frames.append({"index": fr["index"], "width": w, "height": h, "detections": nms(dets, NMS_IOU, by_label=True)})
        return EngineResult(frames=frames, runtime={"onnxruntime": _ort_version(), "input_size": [iw, ih]})


def _grids(np: Any, ih: int, iw: int) -> Any:
    grids, strides = [], []
    for s in STRIDES:
        hs, ws = ih // s, iw // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        strides.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def _ort_version() -> str:
    try:
        from importlib.metadata import version
        return version("onnxruntime")
    except Exception:  # noqa: BLE001
        return "unknown"
