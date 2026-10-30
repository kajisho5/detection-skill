"""`detection doctor`: what is installed, what is missing and how to get it. Never downloads; looks up Python
packages without importing them, hashes model files, asks tesseract for its version and languages."""
from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any, Dict, Optional

from . import __version__
from .frames import SUPPORTED_MAJOR, engine_doctor, locate
from .models import (ENV_MODELS_DIR, ENV_TESSDATA_DIR, REQUIRED_LANGS, describe_traineddata, find_tesseract, inspect_models,
                     resolve_models_dir, tessdata_dir, tessdata_langs, tesseract_version)

FACE_MODEL_SOURCE = ("download face_detection_yunet_2026may.onnx (MIT) from "
                     "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet into the models directory "
                     f"and set ${ENV_MODELS_DIR}")
OBJECT_MODEL_SOURCE = ("download a YOLOX ONNX file (Apache-2.0, e.g. yolox_s.onnx or yolox_nano.onnx) from "
                       "https://github.com/Megvii-BaseDetection/YOLOX/releases into the models directory "
                       f"and set ${ENV_MODELS_DIR}")
TESSDATA_SOURCE = ("for tessdata_best: download jpn.traineddata, jpn_vert.traineddata and eng.traineddata from "
                   f"https://github.com/tesseract-ocr/tessdata_best into a directory and set ${ENV_TESSDATA_DIR}")


def _pkg(module: str, dist: str) -> Dict[str, Any]:
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    version: Optional[str] = None
    if found:
        try:
            from importlib.metadata import PackageNotFoundError, version as _v
            for name in (dist, "opencv-python", "opencv-contrib-python") if module == "cv2" else (dist,):
                try:
                    version = _v(name)
                    break
                except PackageNotFoundError:
                    continue
        except ImportError:  # pragma: no cover
            version = None
    return {"status": "OK" if found else "MISSING", "found": found, "version": version}


def doctor(models_dir: Optional[str] = None, tessdata: Optional[str] = None, ffmpeg_skill: Optional[str] = None) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    py_ok = sys.version_info >= (3, 9)
    checks["python"] = {"status": "OK" if py_ok else "MISSING", "version": platform.python_version(),
                        "install": None if py_ok else "Python >= 3.9"}

    skill = locate(ffmpeg_skill)
    if skill is None:
        checks["ffmpeg_skill"] = {"status": "MISSING", "used_for": "frames of a video (probe, scenes, look); images need none",
                                  "install": "git clone https://github.com/kajisho5/ffmpeg-skill ~/.claude/skills/ffmpeg-skill "
                                             "(or pass --ffmpeg-skill / set $DETECTION_FFMPEG_SKILL_DIR)"}
    else:
        compatible = skill.major == SUPPORTED_MAJOR
        ed = engine_doctor(skill) if compatible else {"ok": False, "ffmpeg": None, "tools": {}}
        status = "INCOMPATIBLE" if not compatible else ("OK" if ed["ok"] else "INCOMPLETE")
        checks["ffmpeg_skill"] = {"status": status, "version": skill.version, "required": ">=2.0.0 <3.0.0", "ffmpeg": ed.get("ffmpeg"),
                                  "tools": ed.get("tools"), "used_for": "frames of a video (probe, scenes, look)",
                                  "install": None if status == "OK" else
                                  "ffmpeg-skill 2.x with a working ffmpeg (run its `python3 scripts/_contract.py doctor`)"}

    cv2 = _pkg("cv2", "opencv-python-headless")
    checks["opencv"] = dict(cv2, used_for="faces (YuNet), image decoding for objects",
                            install=None if cv2["found"] else "pip install 'detection-skill[faces]'  (opencv-python-headless)")
    ort = _pkg("onnxruntime", "onnxruntime")
    checks["onnxruntime"] = dict(ort, used_for="objects (YOLOX)",
                                 install=None if ort["found"] else "pip install 'detection-skill[objects]'  (onnxruntime)")
    np_ = _pkg("numpy", "numpy")
    checks["numpy"] = dict(np_, install=None if np_["found"] else "installed with the faces / objects extras")

    mdir = resolve_models_dir(models_dir)
    models = inspect_models(mdir)
    face, obj = models["face"], models["object"]
    face["install"] = None if face["status"] == "OK" else FACE_MODEL_SOURCE
    obj["install"] = None if obj["status"] == "OK" else OBJECT_MODEL_SOURCE
    checks["face_model"] = face
    checks["object_model"] = obj

    binary = find_tesseract()
    if not binary:
        checks["tesseract"] = {"status": "MISSING", "used_for": "text (OCR)",
                               "install": "apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert / "
                                          "brew install tesseract tesseract-lang / choco install tesseract"}
        checks["tessdata"] = {"status": "MISSING", "langs": [], "install": TESSDATA_SOURCE}
    else:
        version = tesseract_version(binary)
        major = None
        try:
            major = int((version or "").split(".")[0])
        except ValueError:
            pass
        checks["tesseract"] = {"status": "OK" if major == 5 else ("INCOMPATIBLE" if major else "INCOMPLETE"), "version": version,
                               "required": ">=5.0 <6.0", "used_for": "text (OCR)",
                               "install": None if major == 5 else "Tesseract 5 (the LSTM engine)"}
        directory = tessdata_dir(binary, tessdata)
        have = tessdata_langs(directory)
        missing = [lg for lg in REQUIRED_LANGS if lg not in have]
        files = [describe_traineddata(directory, lg) for lg in REQUIRED_LANGS if lg in have]  # type: ignore[arg-type]
        variants = sorted({f["variant"] for f in files})
        entry: Dict[str, Any] = {"status": "OK" if not missing else ("INCOMPLETE" if have else "MISSING"),
                                 "dir_source": "--tessdata-dir/$" + ENV_TESSDATA_DIR if (tessdata or _env_tessdata())
                                 else "tesseract's own", "langs": have, "missing": missing, "files": files,
                                 "variants": variants,
                                 "install": None if not missing else
                                 f"add {', '.join(m + '.traineddata' for m in missing)} (apt: tesseract-ocr-jpn / -jpn-vert); "
                                 + TESSDATA_SOURCE}
        if files and variants != ["tessdata_best"]:
            entry["note"] = "not tessdata_best by sha256 (ADR-001 recommends tessdata_best): " + TESSDATA_SOURCE
        checks["tessdata"] = entry

    ffm_ok = checks["ffmpeg_skill"]["status"] == "OK"
    faces_ok = cv2["found"] and np_["found"] and face["status"] == "OK"
    text_ok = checks["tesseract"]["status"] == "OK" and checks["tessdata"]["status"] == "OK"
    objects_ok = ort["found"] and cv2["found"] and np_["found"] and obj["status"] == "OK"
    ready = {"faces": faces_ok, "text": text_ok, "objects": objects_ok, "video_frames": ffm_ok,
             "tracks": py_ok, "check": py_ok}
    problems = [f"{k}: {v['status']}" + (f" ({v.get('detail')})" if v.get("detail") else "")
                for k, v in checks.items() if v["status"] != "OK"]
    return {
        "ok": all(ready.values()),
        "skill": "detection-skill",
        "version": __version__,
        "ready": ready,
        "checks": checks,
        "problems": problems,
        "network": "never used at run time; model files are read from the models directory and the tessdata directory",
    }


def _env_tessdata() -> bool:
    import os
    return bool(os.environ.get(ENV_TESSDATA_DIR))
