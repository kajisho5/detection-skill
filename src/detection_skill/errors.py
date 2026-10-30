"""One error type with a stable code. The code decides the CLI exit code (contract.ERRORS)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class DetectionError(Exception):
    def __init__(self, code: str, message: str, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message, "detail": self.detail}}
