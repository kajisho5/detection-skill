"""Shots, optional: TransNetV2 (Souček & Lokoč 2020; code and weights MIT) as an ONNX file through onnxruntime
(ADR-003). Meant for gradual transitions, which PySceneDetect's detectors do not see.

The skill ships no weights and downloads none: put an ONNX export of TransNetV2 named `transnetv2.onnx` (or
`transnetv2*.onnx`) in the models directory. `doctor` reports it MISSING otherwise, and `detection shots` keeps
working with the default engine. Inference follows the upstream recipe: frames 48x27 RGB, windows of 100 frames
with step 50 and 25 frames of padding (`shotcore.transnet_windows`), the middle 50 predictions of each window,
sigmoid on logits, threshold 0.5. Not measured in this repository: CI has no TransNetV2 weights.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Dict, List, Optional

from ..errors import DetectionError
from ..models import KNOWN_MODELS, MODEL_LICENSES, describe_file, find_model
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .shotcore import TN_PAD, TN_STEP, sigmoid_if_logits, transnet_boundaries, transnet_windows
from .yunet_engine import read_image

INSTALL = "pip install 'detection-skill[objects]'  (onnxruntime) and transnetv2.onnx in the models directory"
INPUT_W, INPUT_H = 48, 27
DEFAULT_THRESHOLD = 0.5


class TransNetV2Engine(DetectionEngine):
    id = "transnetv2"
    kind = "shot"
    description = "TransNetV2 shot-boundary network (ONNX export) via onnxruntime, CPU; optional"
    runtime = "onnxruntime"
    license = "MIT"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        for mod in ("onnxruntime", "cv2", "numpy"):
            if importlib.util.find_spec(mod) is None:
                raise DetectionError("ENGINE_UNAVAILABLE", f"{mod} is not installed (TransNetV2 runs through ONNX Runtime)", INSTALL)
        path = find_model("shot", models_dir, model)
        d = describe_file(path)
        known = KNOWN_MODELS.get(d["sha256"], {})
        warnings = [] if d["known"] else [f"model {d['file']}: sha256 not in the skill's table of known files (recorded as measured)"]
        return Prepared(model=known.get("model") or d["file"], model_sha256=d["sha256"], files=[d], license=MODEL_LICENSES["shot"],
                        version=_ort_version(), model_path=path, warnings=warnings)

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"onnxruntime / OpenCV not importable: {exc}", INSTALL)
        p = dict(req.params or {})
        threshold = float(p.get("threshold") or DEFAULT_THRESHOLD)
        frames = sorted(req.frames, key=lambda f: int(f["index"]))
        small = []
        for fr in frames:
            img = read_image(fr["path"])
            small.append(cv2.cvtColor(cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        n = len(small)
        sess = ort.InferenceSession(req.model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dtype = np.uint8 if "uint8" in str(inp.type) else np.float32
        single: List[float] = []
        many: Optional[List[float]] = []
        for win in transnet_windows(n):
            batch = np.stack([small[i] for i in win]).astype(dtype)[None, ...]
            outs = sess.run(None, {inp.name: batch})
            s = np.asarray(outs[0]).reshape(-1)[TN_PAD:TN_PAD + TN_STEP]
            single.extend(sigmoid_if_logits(s.tolist()))
            if many is not None and len(outs) > 1:
                m = np.asarray(outs[1]).reshape(-1)[TN_PAD:TN_PAD + TN_STEP]
                many.extend(sigmoid_if_logits(m.tolist()))
            else:
                many = None
        single = single[:n]
        many = many[:n] if many else None
        boundaries = transnet_boundaries(single, many, threshold)
        return EngineResult(frames=[], runtime={"onnxruntime": _ort_version(), "input": str(inp.type)},
                            data={"boundaries": boundaries, "frames": n, "merged": 0,
                                  "parameters": {"threshold": threshold, "input_size": [INPUT_W, INPUT_H]}})


def _ort_version() -> str:
    try:
        from importlib.metadata import version
        return version("onnxruntime")
    except Exception:  # noqa: BLE001
        return "unknown"
