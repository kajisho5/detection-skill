"""Engine worker: `python -m detection_skill.engines.worker <request.json> <result.json>`.

The service starts this as a child process in its own process group, so --timeout can kill a long native call
(ONNX Runtime, OpenCV, a stuck tesseract) for real and an engine crash cannot take the caller down. The argv is
fixed; only two file paths in a private temporary directory vary. Errors go to the result file as structured
JSON, never as free text on stdout.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def main(argv=None) -> int:
    from ..errors import DetectionError
    from . import get_engine
    from .base import EngineRequest

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.stderr.write("usage: worker <request.json> <result.json>\n")
        return 2
    req_path, out_path = argv
    try:
        with open(req_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        engine = get_engine(doc["engine_id"])
        result = engine.detect(EngineRequest.from_dict(doc["request"]))
        payload = {"ok": True, "result": result.to_dict()}
        code = 0
    except DetectionError as exc:
        payload = exc.to_dict()
        code = 1
    except Exception as exc:  # anything else is an engine failure, reported structurally
        payload = {"ok": False, "error": {"code": "ENGINE_FAILED", "message": f"{type(exc).__name__}: {exc}",
                                          "detail": traceback.format_exc().splitlines()[-6:]}}
        code = 1
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, out_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
