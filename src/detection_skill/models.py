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
# Optional (ADR-003): an ONNX export of TransNetV2 for shot boundaries. No file is known by sha256 yet: the skill
# has not measured one, so any file runs with its measured sha256 and a warning.
SHOT_MODEL_DEFAULT = "transnetv2.onnx"
SHOT_MODEL_PREFIX = "transnetv2"
# Optional (ADR-008): PP-OCRv6 text detection + recognition, three files per tier, the official ONNX exports renamed;
# medium is looked for first, then small (tiny leaves out Japanese and is not looked for).
PPOCRV6_TIERS = ("medium", "small")
PPOCRV6_ROLES = ("det", "rec", "dict")
# Optional (ADR-008): RF-DETR Small / Medium, exported to ONNX by the user (`rfdetr[onnx]`, see README); the names the
# export writes, preferred in this order and the only two recorded as Apache-2.0 (Nano and Large are Apache-2.0 too;
# Atto, Femto, Pico, XL and 2XL are under the Roboflow PML 1.0). Without either, another rfdetr*.onnx is picked and runs
# with model_license "unknown (not checked)" and a warning (RfdetrEngine.prepare).
RFDETR_MODEL_PREFERENCE = ("rfdetr-small.onnx", "rfdetr-medium.onnx")
RFDETR_MODEL_PREFIX = "rfdetr"

# sha256 -> what the file is. Measured on 2026-10-30 from the upstream sources named in README; the PP-OCRv6 files on
# 2026-11-12 (the .onnx sha256 from Hugging Face's LFS record of each file, the .yml hashed from the file itself).
KNOWN_MODELS: Dict[str, Dict[str, str]] = {
    "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0": {
        "file": "face_detection_yunet_2026may.onnx", "model": "YuNet 2026may (OpenCV Zoo)", "license": "MIT", "kind": "face"},
    "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d": {
        "file": "yolox_nano.onnx", "model": "YOLOX-Nano (Megvii YOLOX release 0.1.1rc0, COCO)", "license": "Apache-2.0",
        "kind": "object"},
    "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1": {
        "file": "PP-OCRv6_medium_det.onnx", "model": "PP-OCRv6_medium_det (PaddlePaddle/PP-OCRv6_medium_det_onnx@6132380)",
        "license": "Apache-2.0", "kind": "text"},
    "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba": {
        "file": "PP-OCRv6_medium_rec.onnx", "model": "PP-OCRv6_medium_rec (PaddlePaddle/PP-OCRv6_medium_rec_onnx@50c7eac)",
        "license": "Apache-2.0", "kind": "text"},
    "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129": {
        "file": "PP-OCRv6_medium_rec.yml", "model": "PP-OCRv6_medium_rec character dictionary (inference.yml@50c7eac)",
        "license": "Apache-2.0", "kind": "text"},
    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e": {
        "file": "PP-OCRv6_small_det.onnx", "model": "PP-OCRv6_small_det (PaddlePaddle/PP-OCRv6_small_det_onnx@28fe589)",
        "license": "Apache-2.0", "kind": "text"},
    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634": {
        "file": "PP-OCRv6_small_rec.onnx", "model": "PP-OCRv6_small_rec (PaddlePaddle/PP-OCRv6_small_rec_onnx@b8f84f0)",
        "license": "Apache-2.0", "kind": "text"},
    "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1": {
        "file": "PP-OCRv6_small_rec.yml", "model": "PP-OCRv6_small_rec character dictionary (inference.yml@b8f84f0)",
        "license": "Apache-2.0", "kind": "text"},
}
KNOWN_TESSDATA: Dict[str, Dict[str, str]] = {
    "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba": {"lang": "eng", "variant": "tessdata_best"},
    "36bdf9ac823f5911e624c30d0553e890b8abc7c31a65b3ef14da943658c40b79": {"lang": "jpn", "variant": "tessdata_best"},
    "1258be6eb2a9851f18043234ad18cca13ed32690bfff62b335c898bbea371548": {"lang": "jpn_vert", "variant": "tessdata_best"},
}
MODEL_LICENSES = {"face": "MIT", "object": "Apache-2.0", "text": "Apache-2.0", "shot": "MIT", "ppocrv6": "Apache-2.0",
                  "rfdetr": "Apache-2.0"}
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


MODEL_WANTED = {"face": FACE_MODEL_DEFAULT, "shot": SHOT_MODEL_DEFAULT, "object": " or ".join(OBJECT_MODEL_PREFERENCE[:3]),
                "rfdetr": " or ".join(RFDETR_MODEL_PREFERENCE)}
DOCTOR_HINT = "run `detection doctor` for where to get the model files"


def find_model(kind: str, models_dir: Optional[str], name: Optional[str] = None, how: Optional[Any] = None) -> str:
    """Path of the ONNX file for `kind` ('face', 'object', 'shot' or 'rfdetr'). `name` is a file name inside the models
    directory (or a path to a file). Raises MODEL_NOT_FOUND with the reason; `how` (default: run doctor) is its detail
    when the file is missing, e.g. the commands that get it."""
    how = how if how is not None else DOCTOR_HINT
    if name and (os.sep in name or (os.altsep and os.altsep in name)) and os.path.isfile(name):
        path = os.path.abspath(name)
    else:
        if not models_dir:
            raise DetectionError("MODEL_NOT_FOUND", f"no models directory: pass --models-dir or set ${ENV_MODELS_DIR}", how)
        if not os.path.isdir(models_dir):
            raise DetectionError("MODEL_NOT_FOUND", f"models directory does not exist: {os.path.basename(models_dir)}", how)
        path = os.path.join(models_dir, name) if name else _default_model(kind, models_dir)
        if not path or not os.path.isfile(path):
            raise DetectionError("MODEL_NOT_FOUND", f"{name or MODEL_WANTED[kind]} not found in the models directory", how)
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
    if kind == "shot":
        if SHOT_MODEL_DEFAULT in names:
            return os.path.join(models_dir, SHOT_MODEL_DEFAULT)
        tn = [n for n in names if n.lower().startswith(SHOT_MODEL_PREFIX) and n.endswith(".onnx")]
        return os.path.join(models_dir, tn[0]) if tn else None
    if kind == "rfdetr":
        for n in RFDETR_MODEL_PREFERENCE:
            if n in names:
                return os.path.join(models_dir, n)
        rf = [n for n in names if n.lower().startswith(RFDETR_MODEL_PREFIX) and n.endswith(".onnx")]
        return os.path.join(models_dir, rf[0]) if rf else None
    for n in OBJECT_MODEL_PREFERENCE:
        if n in names:
            return os.path.join(models_dir, n)
    yolox = [n for n in names if n.startswith(OBJECT_MODEL_PREFIX) and n.endswith(".onnx")]
    return os.path.join(models_dir, yolox[0]) if yolox else None


def ppocrv6_names(tier: str) -> Dict[str, str]:
    """The three PP-OCRv6 files of a tier, by role: the detector, the recognizer and its character dictionary."""
    return {"det": f"PP-OCRv6_{tier}_det.onnx", "rec": f"PP-OCRv6_{tier}_rec.onnx", "dict": f"PP-OCRv6_{tier}_rec.yml"}


def find_ppocrv6(models_dir: Optional[str], how: Optional[Any] = None) -> Dict[str, Any]:
    """{tier, files: {det, rec, dict}} for the first tier (medium, then small) whose three files are all in the models
    directory. Raises MODEL_NOT_FOUND naming the missing files (of the first tier with any file), `how` as the detail."""
    how = how if how is not None else DOCTOR_HINT
    if not models_dir:
        raise DetectionError("MODEL_NOT_FOUND", f"no models directory: pass --models-dir or set ${ENV_MODELS_DIR}", how)
    if not os.path.isdir(models_dir):
        raise DetectionError("MODEL_NOT_FOUND", f"models directory does not exist: {os.path.basename(models_dir)}", how)
    partial = None
    for tier in PPOCRV6_TIERS:
        files = {role: os.path.join(models_dir, n) for role, n in ppocrv6_names(tier).items()}
        have = {role: os.path.isfile(p) for role, p in files.items()}
        if all(have.values()):
            for p in files.values():
                if is_lfs_pointer(p):
                    raise DetectionError("MODEL_NOT_FOUND", f"{os.path.basename(p)} is a Git LFS pointer, not the model file", how)
            return {"tier": tier, "files": files}
        if partial is None and any(have.values()):
            partial = [os.path.basename(p) for role, p in files.items() if not have[role]]
    missing = partial or list(ppocrv6_names(PPOCRV6_TIERS[0]).values())
    raise DetectionError("MODEL_NOT_FOUND", f"PP-OCRv6 file(s) not found in the models directory: {', '.join(missing)}", how)


def inspect_ppocrv6(models_dir: Optional[str]) -> Dict[str, Any]:
    """doctor's view of the optional PP-OCRv6 files (ADR-008): OK (one tier complete), INCOMPLETE (some files of a tier,
    or a Git LFS pointer) or MISSING, with the files still missing."""
    entry: Dict[str, Any] = {"status": "MISSING", "tier": None, "files": [], "missing": [], "detail": None}
    if not (models_dir and os.path.isdir(models_dir)):
        entry.update(detail=f"no models directory (--models-dir or ${ENV_MODELS_DIR})",
                     missing=list(ppocrv6_names(PPOCRV6_TIERS[0]).values()))
        return entry
    try:
        found = find_ppocrv6(models_dir)
    except DetectionError as exc:
        names = set(os.listdir(models_dir))
        partial = next((t for t in PPOCRV6_TIERS if any(n in names for n in ppocrv6_names(t).values())), None)
        tier_names = ppocrv6_names(partial or PPOCRV6_TIERS[0]).values()
        entry.update(status="INCOMPLETE" if partial else "MISSING", tier=partial, detail=exc.message,
                     missing=[n for n in tier_names if n not in names])
        return entry
    files = [dict(describe_file(found["files"][role]), role=role) for role in PPOCRV6_ROLES]
    entry.update(status="OK", tier=found["tier"], files=files)
    if not all(f["known"] for f in files):
        entry["detail"] = "a file's sha256 is not in the skill's table of known files; it will run and its sha256 is recorded"
    return entry


def inspect_models(models_dir: Optional[str]) -> Dict[str, Any]:
    """doctor's view of the models directory, per kind: OK / MISSING / INCOMPLETE (an LFS pointer). 'shot' is the
    optional TransNetV2 file (ADR-003), 'rfdetr' the optional RF-DETR export and 'ppocrv6' the optional PP-OCRv6 files
    (ADR-008)."""
    out: Dict[str, Any] = {"dir_set": bool(models_dir), "dir_exists": bool(models_dir and os.path.isdir(models_dir))}
    for kind in ("face", "object", "shot", "rfdetr"):
        entry: Dict[str, Any] = {"status": "MISSING", "file": None, "sha256": None, "known": None, "detail": None}
        if not out["dir_exists"]:
            entry["detail"] = f"no models directory (--models-dir or ${ENV_MODELS_DIR})"
        else:
            path = _default_model(kind, models_dir)  # type: ignore[arg-type]
            if not path:
                entry["detail"] = "no " + {"face": FACE_MODEL_DEFAULT, "shot": SHOT_MODEL_DEFAULT,
                                           "rfdetr": " or ".join(RFDETR_MODEL_PREFERENCE)}.get(kind, "yolox_*.onnx") + \
                    " in the models directory"
            elif is_lfs_pointer(path):
                entry.update(status="INCOMPLETE", file=os.path.basename(path), detail="Git LFS pointer, not the model (git lfs pull)")
            else:
                d = describe_file(path)
                entry.update(status="OK", file=d["file"], sha256=d["sha256"], known=d["known"], model=d["model"])
                if not d["known"]:
                    entry["detail"] = "sha256 not in the skill's table of known files; it will run and its sha256 is recorded"
        out[kind] = entry
    out["ppocrv6"] = inspect_ppocrv6(models_dir)
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
