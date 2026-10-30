"""Test-only engines and image helpers. They live in tests/ and are never registered in the package's engine
registry, so nothing outside the test suite can select them. They are injected as objects (run in a thread with
the service's timeout), which exercises everything around the engine: sampling, frames from (fake or real)
ffmpeg-skill, scaling to source pixels, clipping, provenance, validation, tracks, the CLI.

ScriptedEngine returns boxes from a function of the frame time, in the frame's own pixels, so a test knows the
answer exactly. It cannot measure a real engine; the real-engine tests do that.
"""
from __future__ import annotations

import hashlib
import struct
import time
import zlib
from typing import Any, Callable, Dict, List, Optional

from detection_skill.engines.base import DetectionEngine, EngineRequest, EngineResult, Prepared

MODEL_SHA = hashlib.sha256(b"scripted-model").hexdigest()


def write_png(path: str, width: int, height: int, rgb=(40, 40, 40)) -> None:
    """A solid-colour 8-bit RGB PNG with the standard library only."""
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def png_size(path: str):
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", head[16:24])


class ScriptedEngine(DetectionEngine):
    id = "scripted"
    runtime = "test"
    license = "MIT"

    def __init__(self, kind: str = "face", boxes: Optional[Callable[[Any, int, int], List[Dict[str, Any]]]] = None,
                 delay: float = 0.0, fail: Optional[Exception] = None, warnings: Optional[List[str]] = None) -> None:
        self.kind = kind
        self.boxes = boxes or (lambda t, w, h: [])
        self.delay, self.fail, self.warnings = delay, fail, warnings or []
        self.requests: List[EngineRequest] = []
        self.prepared_with: Dict[str, Any] = {}

    def prepare(self, *, models_dir=None, model=None, tessdata_dir=None, params=None) -> Prepared:
        self.prepared_with = {"models_dir": models_dir, "model": model, "tessdata_dir": tessdata_dir, "params": params}
        return Prepared(model="test/scripted", model_sha256=MODEL_SHA, files=[{"file": "scripted.onnx", "sha256": MODEL_SHA,
                        "size_bytes": 1, "known": False}], license="MIT", version="1.0", warnings=list(self.warnings))

    def detect(self, req: EngineRequest) -> EngineResult:
        self.requests.append(req)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        frames = []
        for fr in req.frames:
            w, h = png_size(fr["path"])
            dets = [d for d in self.boxes(fr["t"], w, h) if d.get("score", 1.0) >= req.min_score]
            frames.append({"index": fr["index"], "width": w, "height": h, "detections": dets})
        return EngineResult(frames=frames, runtime={"test": True})
