"""Text: the `tesseract` binary (Tesseract 5, Apache-2.0) with `jpn`, `jpn_vert` and `eng` traineddata, ideally
from tessdata_best (ADR-001; the same OCR provider as stepcap and slide-sync).

Each frame is one `tesseract <image> stdout -l <langs> --psm N --oem 1 tsv` run (argv list, no shell). Words are
grouped into lines (Tesseract's block/paragraph/line numbers). A line's box is the union of its words, its score
the mean word confidence / 100, its text the words joined with a space, except that no space is kept between two
Japanese characters (Tesseract separates CJK glyphs). Vertical Japanese is read with `jpn_vert` and page
segmentation mode 5 (a single vertical block) when asked for.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from ..errors import DetectionError
from ..models import (MODEL_LICENSES, REQUIRED_LANGS, combined_sha256, describe_traineddata, find_tesseract,
                      tessdata_dir, tessdata_langs, tesseract_version)
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared

LANG_RE = re.compile(r"^[A-Za-z_]+(\+[A-Za-z_]+)*$")
CJK = r"　-〿぀-ゟ゠-ヿ一-鿿㐀-䶿＀-￯々ー"
CJK_SPACE_RE = re.compile(f"(?<=[{CJK}]) +(?=[{CJK}])")
DEFAULT_LANG = "jpn+eng"
DEFAULT_PSM = 3
VERTICAL_PSM = 5
PER_FRAME_TIMEOUT = 300.0
INSTALL = ("install Tesseract 5 (apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert / brew install tesseract "
           "tesseract-lang / choco install tesseract) and, for tessdata_best, put jpn/jpn_vert/eng.traineddata from "
           "https://github.com/tesseract-ocr/tessdata_best in a directory passed as --tessdata-dir")


def resolve_langs(lang: Optional[str], vertical: bool) -> List[str]:
    lang = lang or DEFAULT_LANG
    if not LANG_RE.match(lang):
        raise DetectionError("INVALID_ARGUMENT", f"--lang must look like jpn+eng, got {lang!r}")
    langs = lang.split("+")
    if vertical:
        langs = ["jpn_vert" if lg == "jpn" else lg for lg in langs]
        if "jpn_vert" not in langs:
            langs.insert(0, "jpn_vert")
    out: List[str] = []
    for lg in langs:
        if lg not in out:
            out.append(lg)
    return out


def join_words(words: List[str]) -> str:
    return CJK_SPACE_RE.sub("", " ".join(w for w in words if w)).strip()


class TesseractEngine(DetectionEngine):
    id = "tesseract"
    kind = "text"
    description = "Tesseract 5 OCR (LSTM) via the tesseract binary, jpn / jpn_vert / eng"
    runtime = "tesseract"
    license = "Apache-2.0"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        params = params or {}
        binary = find_tesseract()
        if not binary:
            raise DetectionError("ENGINE_UNAVAILABLE", "tesseract binary not found on PATH", INSTALL)
        version = tesseract_version(binary) or "unknown"
        directory = tessdata_dir_or_default(binary, tessdata_dir)
        have = tessdata_langs(directory)
        langs = params.get("langs") or resolve_langs(None, False)
        missing = [lg for lg in langs if lg not in have]
        if missing:
            raise DetectionError("LANGUAGE_NOT_FOUND", f"traineddata missing for: {', '.join(missing)}",
                                 {"tessdata_has": have, "install": INSTALL})
        files = [describe_traineddata(directory, lg) for lg in langs]  # type: ignore[arg-type]
        warnings = [f"{f['file']} is not tessdata_best by sha256 (another tessdata variant; accuracy may differ)"
                    for f in files if not f["known"]]
        best = all(f["variant"] == "tessdata_best" for f in files)
        name = "tessdata " + "+".join(langs) + (" (tessdata_best)" if best else "")
        return Prepared(model=name, model_sha256=combined_sha256(files), files=files, license=MODEL_LICENSES["text"],
                        version=version, tessdata_dir=directory, tesseract=binary, warnings=warnings)

    def detect(self, req: EngineRequest) -> EngineResult:
        p = req.params
        langs = p.get("langs") or [DEFAULT_LANG]
        psm = int(p.get("psm") or DEFAULT_PSM)
        level = p.get("level") or "line"
        min_score = float(req.min_score or 0.0)
        frames: List[Dict[str, Any]] = []
        for fr in req.frames:
            tsv = run_tesseract(req.tesseract or "tesseract", fr["path"], "+".join(langs), psm, req.tessdata_dir)
            size, words = parse_tsv(tsv)
            if size is None:
                raise DetectionError("ENGINE_FAILED", f"tesseract returned no page for frame {fr['index']}")
            dets = group(words, level)
            dets = [dict(d, lang="+".join(langs)) for d in dets if d["score"] >= min_score and d["label"]]
            frames.append({"index": fr["index"], "width": size[0], "height": size[1], "detections": dets})
        return EngineResult(frames=frames, runtime={"tesseract": tesseract_version(req.tesseract or "tesseract"), "psm": psm,
                                                     "oem": 1, "level": level})


def tessdata_dir_or_default(binary: str, explicit: Optional[str]) -> Optional[str]:
    d = tessdata_dir(binary, explicit)
    if explicit and (not d or not os.path.isdir(d)):
        raise DetectionError("LANGUAGE_NOT_FOUND", "--tessdata-dir does not exist")
    return d


def run_tesseract(binary: str, image: str, lang: str, psm: int, tessdata: Optional[str]) -> str:
    argv = [binary, image, "stdout", "-l", lang, "--psm", str(psm), "--oem", "1"]
    if tessdata:
        argv += ["--tessdata-dir", tessdata]
    # -c instead of the `tsv` config file: a --tessdata-dir with only traineddata has no configs/ directory
    argv += ["-c", "tessedit_create_tsv=1", "-c", "tessedit_create_txt=0"]
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR",
                                                         "LANG", "LC_ALL")}
    env["OMP_THREAD_LIMIT"] = "1"
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, stdin=subprocess.DEVNULL,
                              timeout=PER_FRAME_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise DetectionError("ENGINE_TIMEOUT", f"tesseract exceeded {PER_FRAME_TIMEOUT:g}s on one frame")
    except OSError as exc:
        raise DetectionError("ENGINE_UNAVAILABLE", f"cannot run tesseract: {exc}", INSTALL)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise DetectionError("ENGINE_FAILED", f"tesseract exited {proc.returncode}", tail)
    return proc.stdout.decode("utf-8", "replace")


def parse_tsv(tsv: str) -> Tuple[Optional[Tuple[int, int]], List[Dict[str, Any]]]:
    """(page width/height, words [{key: (block, par, line), box, conf, text}])."""
    size: Optional[Tuple[int, int]] = None
    words: List[Dict[str, Any]] = []
    lines = tsv.splitlines()
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        try:
            level = int(cols[0])
            left, top, width, height = (int(c) for c in cols[6:10])
            conf = float(cols[10])
        except ValueError:
            continue
        if level == 1 and size is None:
            size = (width, height)
        elif level == 5:
            text = cols[11].strip()
            if text and conf >= 0 and width > 0 and height > 0:
                words.append({"key": (int(cols[2]), int(cols[3]), int(cols[4])), "box": [left, top, width, height],
                              "conf": conf, "text": text})
    return size, words


def group(words: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    if level == "word":
        return [{"box": [float(v) for v in w["box"]], "score": round(w["conf"] / 100.0, 4), "label": w["text"], "words": 1}
                for w in words]
    lines: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for w in words:
        lines.setdefault(w["key"], []).append(w)
    out: List[Dict[str, Any]] = []
    for key in sorted(lines):
        ws = lines[key]
        x1 = min(w["box"][0] for w in ws)
        y1 = min(w["box"][1] for w in ws)
        x2 = max(w["box"][0] + w["box"][2] for w in ws)
        y2 = max(w["box"][1] + w["box"][3] for w in ws)
        out.append({"box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": round(sum(w["conf"] for w in ws) / len(ws) / 100.0, 4),
                    "label": join_words([w["text"] for w in ws]), "words": len(ws)})
    return out


__all__ = ["TesseractEngine", "resolve_langs", "join_words", "parse_tsv", "group", "REQUIRED_LANGS"]
