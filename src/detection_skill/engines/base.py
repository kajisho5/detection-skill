"""Engine contract. An engine turns image files (frames) into raw boxes in that image's own pixels; everything
else (sampling, frames, scaling to source pixels, clipping, provenance, schema) is the service's job, so every
engine yields the same document.

`prepare` runs in the calling process and must be cheap: it finds the model files and the runtime and returns
what they are (file names and sha256). `detect` may be slow and runs in a worker process.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineRequest:
    frames: List[Dict[str, Any]]                # [{index, t, path}]
    model_path: Optional[str] = None            # the ONNX file (faces, objects)
    tessdata_dir: Optional[str] = None          # text
    tesseract: Optional[str] = None             # text: the binary
    min_score: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EngineResult:
    """frames: [{index, width, height, detections: [{box: [x, y, w, h], score, label, lang?, landmarks?}]}] with
    boxes in the frame image's own pixels, in any order."""
    frames: List[Dict[str, Any]]
    runtime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"frames": self.frames, "runtime": dict(self.runtime)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        return cls(frames=list(d["frames"]), runtime=dict(d.get("runtime") or {}))


@dataclass
class Prepared:
    """What `prepare` found: the model (name, sha256 of the file or of the file set) and the files."""
    model: str
    model_sha256: str
    files: List[Dict[str, Any]]
    license: str
    version: str
    model_path: Optional[str] = None
    tessdata_dir: Optional[str] = None
    tesseract: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class DetectionEngine:
    """Subclass and register in `engines/__init__.py`."""
    id = "base"
    kind = "face"            # face | text | object
    description = ""
    runtime = ""
    license = ""             # licence of the engine code
    execution_mode = "local"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:  # pragma: no cover - interface
        raise NotImplementedError

    def detect(self, req: EngineRequest) -> EngineResult:  # pragma: no cover - interface
        raise NotImplementedError
