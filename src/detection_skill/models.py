"""Model files and the OCR binary: where they are, what they are (sha256), never how to download them.

ADR-001: model files live in a directory the user points to (`--models-dir` or $DETECTION_MODELS_DIR);
Tesseract's traineddata in `--tessdata-dir` / $DETECTION_TESSDATA_DIR, else the tesseract binary's own
tessdata directory. Every result records the sha256 of every model file it used. A file whose sha256 is in
the table below is named by it ("known"); any other file still runs, with its measured sha256 recorded and a
warning that the skill has not seen that file before.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .errors import DetectionError

ENV_MODELS_DIR = "DETECTION_MODELS_DIR"
ENV_TESSDATA_DIR = "DETECTION_TESSDATA_DIR"
ENV_TESSERACT = "DETECTION_TESSERACT"

FACE_MODEL_DEFAULT = "face_detection_yunet_2026may.onnx"
FACE_MODEL_GLOB_PREFIX = "face_detection_yunet_"
OBJECT_MODEL_PREFERENCE = ("yolox_s.onnx", "yolox_tiny.onnx", "yolox_nano.onnx", "yolox_m.onnx", "yolox_l.onnx",
                           "yolox_x.onnx", "yolox_darknet.onnx")
OBJECT_MODEL_PREFIX = "yolox_"

# sha256 -> what the file is. Measured on 2026-10-30 from the upstream sources named in README.
KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0": {
        "file": "face_detection_yunet_2026may.onnx", "model": "YuNet 2026may (OpenCV Zoo)", "license": "MIT", "kind": "face"},
    "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d": {
        "file": "yolox_nano.onnx", "model": "YOLOX-Nano (Megvii YOLOX release 0.1.1rc0, COCO)", "license": "Apache-2.0",
        "kind": "object"},
}
KNOWN_TESSDATA: Dict[str, Dict[str, str]] = {
    "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba": {"lang": "eng", "variant": "tessdata_best"},
    "36bdf9ac823f5911e624c30d0553e890b8abc7c31a65b3ef14da943658c40b79": {"lang": "jpn", "variant": "tessdata_best"},
    "1258be6eb2a9851f18043234ad18cca13ed32690bfff62b335c898bbea371548": {"lang": "jpn_vert", "variant": "tessdata_best"},
}
MODEL_LICENSES = {"face": "MIT", "object": "Apache-2.0", "text": "Apache-2.0"}
REQUIRED_LANGS = ("jpn", "jpn_vert", "eng")
LFS_POINTER = b"version https://git-lfs"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def resolve_models_dir(explicit: Optional[str] = None) -> Optional[str]:
    d = explicit or os.environ.get(ENV_MODELS_DIR)
    return os.path.abspath(os.path.expanduser(d)) if d else None


def describe_file(path: str) -> Dict[str, Any]:
    """name, sha256, size and whether the sha256 is one the skill knows."""
    digest = sha256_file(path)
    known = KNOWN_MODELS.get(digest)
    return {"file": os.path.basename(path), "sha256": digest, "size_bytes": os.path.getsize(path), "known": known is not None,
            "model": known["model"] if known else None}


def find_model(kind: str, models_dir: Optional[str], name: Optional[str] = None) -> str:
    """Path of the ONNX file for `kind` ('face' or 'object'). `name` is a file name inside the models directory
    (or a path to a file). Raises MODEL_NOT_FOUND with the reason."""
    if name and (os.sep in name or (os.altsep and os.altsep in name)) and os.path.isfile(name):
        path = os.path.abspath(name)
    else:
        if not models_dir:
            raise DetectionError("MODEL_NOT_FOUND", f"no models directory: pass --models-dir or set ${ENV_MODELS_DIR}",
                                 "run `detection doctor` for where to get the model files")
        if not os.path.isdir(models_dir):
            raise DetectionError("MODEL_NOT_FOUND", f"models directory does not exist: {os.path.basename(models_dir)}")
        path = os.path.join(models_dir, name) if name else _default_model(kind, models_dir)
        if not path or not os.path.isfile(path):
            want = name or (FACE_MODEL_DEFAULT if kind == "face" else " or ".join(OBJECT_MODEL_PREFERENCE[:3]))
            raise DetectionError("MODEL_NOT_FOUND", f"{want} not found in the models directory",
                                 "run `detection doctor` for where to get the model files")
    if is_lfs_pointer(path):
        raise DetectionError("MODEL_NOT_FOUND", f"{os.path.basename(path)} is a Git LFS pointer, not the model (git lfs pull)")
    if not path.lower().endswith(".onnx"):
        raise DetectionError("INVALID_ARGUMENT", "model file must be an .onnx file")
    return path


def _default_model(kind: str, models_dir: str) -> Optional[str]:
    names = sorted(os.listdir(models_dir))
    if kind == "face":
        if FACE_MODEL_DEFAULT in names:
            return os.path.join(models_dir, FACE_MODEL_DEFAULT)
        yunet = [n for n in names if n.startswith(FACE_MODEL_GLOB_PREFIX) and n.endswith(".onnx")]
        return os.path.join(models_dir, yunet[-1]) if yunet else None
    for n in OBJECT_MODEL_PREFERENCE:
        if n in names:
            return os.path.join(models_dir, n)
    yolox = [n for n in names if n.startswith(OBJECT_MODEL_PREFIX) and n.endswith(".onnx")]
    return os.path.join(models_dir, yolox[0]) if yolox else None


def inspect_models(models_dir: Optional[str]) -> Dict[str, Any]:
    """doctor's view of the models directory, per kind: OK / MISSING / INCOMPLETE (an LFS pointer)."""
    out: Dict[str, Any] = {"dir_set": bool(models_dir), "dir_exists": bool(models_dir and os.path.isdir(models_dir))}
    for kind in ("face", "object"):
        entry: Dict[str, Any] = {"status": "MISSING", "file": None, "sha256": None, "known": None, "detail": None}
        if not out["dir_exists"]:
            entry["detail"] = f"no models directory (--models-dir or ${ENV_MODELS_DIR})"
        else:
            path = _default_model(kind, models_dir)  # type: ignore[arg-type]
            if not path:
                entry["detail"] = "no " + (FACE_MODEL_DEFAULT if kind == "face" else "yolox_*.onnx") + " in the models directory"
            elif is_lfs_pointer(path):
                entry.update(status="INCOMPLETE", file=os.path.basename(path), detail="Git LFS pointer, not the model (git lfs pull)")
            else:
                d = describe_file(path)
                entry.update(status="OK", file=d["file"], sha256=d["sha256"], known=d["known"], model=d["model"])
                if not d["known"]:
                    entry["detail"] = "sha256 not in the skill's table of known files; it will run and its sha256 is recorded"
        out[kind] = entry
    return out


# ---- tesseract ----------------------------------------------------------------------------------
def find_tesseract() -> Optional[str]:
    env = os.environ.get(ENV_TESSERACT)
    if env and os.path.isfile(env):
        return env
    return shutil.which("tesseract")


def _run(argv: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, shell=False,
                          stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")


def tesseract_version(binary: str) -> Optional[str]:
    try:
        p = _run([binary, "--version"])
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"tesseract\s+v?(\d+\.\d+(?:\.\d+)?\S*)", (p.stdout or "") + (p.stderr or ""), re.I)
    return m.group(1) if m else None


def tessdata_dir(binary: str, explicit: Optional[str] = None) -> Optional[str]:
    """The directory holding the traineddata: the one given, $DETECTION_TESSDATA_DIR, else the binary's own
    (read from `tesseract --list-langs`)."""
    d = explicit or os.environ.get(ENV_TESSDATA_DIR)
    if d:
        return os.path.abspath(os.path.expanduser(d))
    try:
        p = _run([binary, "--list-langs"])
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"([^"]+)"', (p.stdout or "") + (p.stderr or ""))
    return os.path.abspath(m.group(1)) if m else None


def tessdata_langs(directory: Optional[str]) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(f[: -len(".traineddata")] for f in os.listdir(directory) if f.endswith(".traineddata"))


def describe_traineddata(directory: str, lang: str) -> Dict[str, Any]:
    path = os.path.join(directory, f"{lang}.traineddata")
    digest = sha256_file(path)
    known = KNOWN_TESSDATA.get(digest)
    return {"file": f"{lang}.traineddata", "lang": lang, "sha256": digest, "size_bytes": os.path.getsize(path),
            "known": known is not None, "variant": known["variant"] if known else "unknown (not tessdata_best by sha256)"}


def combined_sha256(files: List[Dict[str, Any]]) -> str:
    """One sha256 for a set of files: sha256 over sorted 'file=sha256' lines. Each file's own sha256 is listed in
    the document's engine.files."""
    text = "".join(f"{f['file']}={f['sha256']}\n" for f in sorted(files, key=lambda f: f["file"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
