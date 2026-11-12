"""Objects, optional: RF-DETR Small / Medium (Roboflow; code and these weights Apache-2.0) as an ONNX export through
onnxruntime on CPU (ADR-008). A second object detector next to YOLOX, which stays the default.

No ONNX file is published upstream for these weights: the user exports one, once, with the `rfdetr` package (which needs
PyTorch for the export only; this skill never imports torch) and puts it in the models directory (`detection doctor`
prints the commands):

    pip install "rfdetr[onnx]==1.12.0"
    python -c "from rfdetr import RFDETRSmall; RFDETRSmall().export(output_dir='rfdetr-export')"
    -> rfdetr-export/rfdetr-small.onnx   (RFDETRMedium -> rfdetr-medium.onnx)

Inference follows rfdetr's ONNX reference (`export/_onnx/inference.py`, `export/_runtime/decode.py`):

- Input `input` [1, 3, H, W] float32, H x W fixed by the export (Small 512, Medium 576): the frame resized to H x W
  (bilinear, no letterbox, no antialias), RGB, scaled to 0-1, normalised with mean (0.485, 0.456, 0.406) and std
  (0.229, 0.224, 0.225).
- Outputs, matched by name: `dets` [1, Q, 4] boxes as normalised (cx, cy, w, h) and `labels` [1, Q, 91] raw logits.
- Decoding: a sigmoid per class (not a softmax), the Q x 91 query/class pairs ranked together and the best Q kept
  (descending score, then ascending index), then the score threshold; no NMS (a DETR predicts a set). Boxes are scaled
  to the frame and clipped to it.
- Classes: the official COCO checkpoints keep COCO's sparse category ids, so slot k is COCO category id k (1 person ...
  90 toothbrush) and the 11 slots without a category (0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83) are dropped. Labels
  are the same 80 names YOLOX uses. A file with another number of slots (a custom-trained model) is refused.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import DetectionError
from ..models import ENV_MODELS_DIR, KNOWN_MODELS, MODEL_LICENSES, RFDETR_MODEL_PREFERENCE, describe_file, find_model
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .yolox_engine import COCO_CLASSES
from .yunet_engine import read_image

# COCO category ids of the 80 classes, in COCO_CLASSES order (rfdetr assets/coco_classes.py COCO_CLASSES).
COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
SLOT_NAMES: Dict[int, str] = dict(zip(COCO_IDS, COCO_CLASSES))
CLASS_SLOTS = 91
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
INSTALL = "pip install 'detection-skill[objects]'  (onnxruntime, opencv-python-headless)"
RFDETR_VERSION = "1.12.0"
EXPORT = (
    f'pip install "rfdetr[onnx]=={RFDETR_VERSION}"',
    "python -c \"from rfdetr import RFDETRSmall; RFDETRSmall().export(output_dir='rfdetr-export')\"",
    f'cp rfdetr-export/rfdetr-small.onnx "${{{ENV_MODELS_DIR}:?set {ENV_MODELS_DIR} to the models directory}}/"',
)
SOURCE = ("optional (detection objects --engine rfdetr): RF-DETR Small or Medium (code and weights Apache-2.0, "
          "https://github.com/roboflow/rf-detr) has no published ONNX file; export it once with the rfdetr package, in any "
          "environment (the export needs PyTorch, this skill does not), and save it as rfdetr-small.onnx or "
          f"rfdetr-medium.onnx in the models directory (${ENV_MODELS_DIR}): " + " && ".join(EXPORT) +
          " (RFDETRMedium writes rfdetr-medium.onnx); the default object engine (YOLOX) needs none of it")


def class_name(slot: int) -> Optional[str]:
    return SLOT_NAMES.get(int(slot))


def preprocess(img: Any, height: int, width: int, np: Any, cv2: Any) -> Any:
    """BGR uint8 frame -> [1, 3, H, W] float32: RGB, 0-1, bilinear resize (half-pixel, no antialias, on the float image as
    rfdetr does), ImageNet normalisation."""
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = (rgb - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def decode(boxes_cwh: Any, logits: Any, width: int, height: int, np: Any, min_score: float = 0.0,
           num_select: Optional[int] = None, wanted: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One image's `dets` [Q, 4] and `labels` [Q, 91] -> [{box [x, y, w, h] in pixels, score, label}], best first."""
    logits = np.asarray(logits, dtype=np.float32)
    boxes_cwh = np.asarray(boxes_cwh, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != CLASS_SLOTS or boxes_cwh.shape != (logits.shape[0], 4):
        raise DetectionError("ENGINE_FAILED", f"unexpected RF-DETR outputs: dets {tuple(boxes_cwh.shape)}, labels "
                                              f"{tuple(logits.shape)} (want [Q, 4] and [Q, {CLASS_SLOTS}], an official "
                                              "COCO checkpoint export)")
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
    flat = scores.reshape(-1)
    q_count, c_count = scores.shape
    k = min(num_select or q_count, flat.size)
    order = np.lexsort((np.arange(flat.size), -flat))[:k]
    wanted_set = set(wanted or [])
    out: List[Dict[str, Any]] = []
    for i in order.tolist():
        s = float(flat[i])
        if s < min_score:
            break
        q, slot = divmod(i, c_count)
        name = class_name(slot)
        if name is None or (wanted_set and name not in wanted_set):
            continue
        cx, cy, bw, bh = (float(v) for v in boxes_cwh[q])
        x1, y1 = max(0.0, (cx - bw / 2) * width), max(0.0, (cy - bh / 2) * height)
        x2, y2 = min(float(width), (cx + bw / 2) * width), min(float(height), (cy + bh / 2) * height)
        if x2 > x1 and y2 > y1:
            out.append({"box": [x1, y1, x2 - x1, y2 - y1], "score": s, "label": name})
    return out


def io_layout(session: Any) -> Tuple[Any, int, int, Any, int, int]:
    """(input, height, width, numpy dtype name, index of dets, index of labels), outputs matched by name."""
    inp = session.get_inputs()[0]
    shape = list(inp.shape)
    if len(shape) != 4 or not all(isinstance(v, int) for v in shape[2:]):
        raise DetectionError("ENGINE_FAILED", f"RF-DETR input {inp.name} has shape {shape}; an export has a fixed [1, 3, H, W]")
    names = [o.name for o in session.get_outputs()]
    dets = next((i for i, n in enumerate(names) if "dets" in n), None)
    labels = next((i for i, n in enumerate(names) if "labels" in n), None)
    if dets is None or labels is None:
        raise DetectionError("ENGINE_FAILED", f"RF-DETR export without outputs dets / labels (has {names})")
    dtype = "float16" if "float16" in str(inp.type) else "float32"
    return inp, int(shape[2]), int(shape[3]), dtype, dets, labels


def run_frame(session: Any, img: Any, np: Any, cv2: Any, min_score: float = 0.0,
              wanted: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    inp, ih, iw, dtype, di, li = io_layout(session)
    x = preprocess(img, ih, iw, np, cv2).astype(dtype)
    outs = session.run(None, {inp.name: x})
    h, w = img.shape[:2]
    return decode(np.asarray(outs[di])[0], np.asarray(outs[li])[0], w, h, np, min_score=min_score, wanted=wanted)


class RfdetrEngine(DetectionEngine):
    id = "rfdetr"
    kind = "object"
    description = "RF-DETR Small / Medium (Roboflow) COCO detector, ONNX export via onnxruntime, CPU; optional"
    runtime = "onnxruntime"
    license = "Apache-2.0"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        missing = [m for m in ("onnxruntime", "cv2", "numpy") if importlib.util.find_spec(m) is None]
        if missing:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not installed: {', '.join(missing)} (RF-DETR runs through ONNX Runtime)",
                                 INSTALL)
        path = find_model("rfdetr", models_dir, model, how=list(EXPORT))
        d = describe_file(path)
        known = KNOWN_MODELS.get(d["sha256"], {})
        warnings = [] if d["known"] else [f"model {d['file']}: sha256 not in the skill's table of known files (recorded as "
                                          "measured; an RF-DETR export is made by you, so its sha256 is yours)"]
        if d["file"] in RFDETR_MODEL_PREFERENCE:
            license_ = MODEL_LICENSES["rfdetr"]
        else:
            license_ = "unknown (not checked)"
            warnings.append(f"model {d['file']} is not rfdetr-small.onnx or rfdetr-medium.onnx: its weights' licence is not "
                            "checked (RF-DETR Nano and Large are Apache-2.0 too; Atto, Femto, Pico, XL and 2XL are under "
                            "the Roboflow PML 1.0, not Apache-2.0)")
        model_name = known.get("model") or {"rfdetr-small.onnx": "RF-DETR Small (COCO, ONNX export)",
                                            "rfdetr-medium.onnx": "RF-DETR Medium (COCO, ONNX export)"}.get(d["file"], d["file"])
        return Prepared(model=model_name, model_sha256=d["sha256"], files=[d], license=license_, version=_ort_version(),
                        model_path=path, warnings=warnings, extra={"classes": "coco80 (sparse COCO ids, 91 slots)"})

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not importable: {exc}", INSTALL)
        if not req.model_path or not os.path.isfile(req.model_path):
            raise DetectionError("MODEL_NOT_FOUND", "RF-DETR ONNX file not found", list(EXPORT))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(req.model_path, sess_options=so, providers=["CPUExecutionProvider"])
        _, ih, iw, dtype, _, _ = io_layout(sess)
        wanted = list(req.params.get("classes") or [])
        min_score = float(req.min_score or 0.0)
        frames: List[Dict[str, Any]] = []
        for fr in req.frames:
            img = read_image(fr["path"])
            h, w = img.shape[:2]
            frames.append({"index": fr["index"], "width": w, "height": h,
                           "detections": run_frame(sess, img, np, cv2, min_score=min_score, wanted=wanted)})
        return EngineResult(frames=frames, runtime={"onnxruntime": _ort_version(), "input_size": [iw, ih], "input_dtype": dtype,
                                                     "class_slots": CLASS_SLOTS, "nms": "none (set prediction)"})


def _ort_version() -> str:
    try:
        from importlib.metadata import version
        return version("onnxruntime")
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = ["RfdetrEngine", "COCO_IDS", "SLOT_NAMES", "class_name", "preprocess", "decode", "run_frame"]
