"""Engine registry: one engine per kind (ADR-001). Only engines listed here can be selected or run by the worker;
a test engine is injected as an object by the test suite and is never registered."""
from __future__ import annotations

from typing import Dict, List, Type

from ..errors import DetectionError
from .base import DetectionEngine
from .tesseract_engine import TesseractEngine
from .yolox_engine import YoloxEngine
from .yunet_engine import YunetEngine

KINDS = ("face", "text", "object")
ENGINES: Dict[str, Type[DetectionEngine]] = {YunetEngine.id: YunetEngine, TesseractEngine.id: TesseractEngine,
                                              YoloxEngine.id: YoloxEngine}
DEFAULT_ENGINE = {"face": YunetEngine.id, "text": TesseractEngine.id, "object": YoloxEngine.id}


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> DetectionEngine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise DetectionError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()
