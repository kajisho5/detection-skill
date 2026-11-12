"""Engine registry: one default engine per kind (ADR-001), the shot engines (ADR-003: PySceneDetect by default,
TransNetV2 optional) and the optional second engines for text and objects (ADR-008: PP-OCRv6, RF-DETR). Only engines
listed here can be selected or run by the worker; a test engine is injected as an object by the test suite and is never
registered. A default never changes because an optional engine is installed: the optional ones run only when asked for
with --engine."""
from __future__ import annotations

from typing import Dict, List, Tuple, Type

from ..errors import DetectionError
from .base import DetectionEngine
from .ppocrv6_engine import PPOCRv6Engine
from .pyscenedetect_engine import PySceneDetectEngine
from .rfdetr_engine import RfdetrEngine
from .tesseract_engine import TesseractEngine
from .transnetv2_engine import TransNetV2Engine
from .yolox_engine import YoloxEngine
from .yunet_engine import YunetEngine

KINDS = ("face", "text", "object")
ENGINES: Dict[str, Type[DetectionEngine]] = {YunetEngine.id: YunetEngine, TesseractEngine.id: TesseractEngine,
                                              YoloxEngine.id: YoloxEngine, PySceneDetectEngine.id: PySceneDetectEngine,
                                              TransNetV2Engine.id: TransNetV2Engine, PPOCRv6Engine.id: PPOCRv6Engine,
                                              RfdetrEngine.id: RfdetrEngine}
DEFAULT_ENGINE = {"face": YunetEngine.id, "text": TesseractEngine.id, "object": YoloxEngine.id}
# The engines `--engine` can choose per kind, the default first.
ENGINES_BY_KIND: Dict[str, Tuple[str, ...]] = {"face": (YunetEngine.id,), "text": (TesseractEngine.id, PPOCRv6Engine.id),
                                               "object": (YoloxEngine.id, RfdetrEngine.id)}
TEXT_ENGINES = ENGINES_BY_KIND["text"]
OBJECT_ENGINES = ENGINES_BY_KIND["object"]
SHOT_ENGINES = (PySceneDetectEngine.id, TransNetV2Engine.id)
DEFAULT_SHOT_ENGINE = PySceneDetectEngine.id
OPTIONAL_ENGINES = (TransNetV2Engine.id, PPOCRv6Engine.id, RfdetrEngine.id)


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> DetectionEngine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise DetectionError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()
