"""Text, optional: PP-OCRv6 (PaddlePaddle; code and models Apache-2.0) text detection + recognition as the official ONNX
exports through onnxruntime on CPU (ADR-008). A second OCR engine next to Tesseract, which stays the default.

The skill ships no model file and downloads none. Three files go in the models directory (`detection doctor` prints the
download commands, pinned to a Hugging Face revision; README lists their sha256):

    PP-OCRv6_medium_det.onnx   PaddlePaddle/PP-OCRv6_medium_det_onnx  inference.onnx
    PP-OCRv6_medium_rec.onnx   PaddlePaddle/PP-OCRv6_medium_rec_onnx  inference.onnx
    PP-OCRv6_medium_rec.yml    PaddlePaddle/PP-OCRv6_medium_rec_onnx  inference.yml (the recognizer's character dictionary)

(or the `small` tier under the same names). Pre- and post-processing follow PaddleOCR's general OCR pipeline (PaddleX
`configs/pipelines/OCR.yaml` with its text detection and text recognition processors):

- Detection input: the frame in BGR, resized so that its short side is at least 64 px (the long side at most 4000 px),
  each side then rounded to a multiple of 32; scaled to 0-1 and normalised per channel with mean (0.485, 0.456, 0.406)
  and std (0.229, 0.224, 0.225), applied in B, G, R order as PaddleOCR does; NCHW float32. Output: a text probability
  map [N, 1, H, W] (DB head, sigmoid).
- DB post-processing: map > 0.3 is text; each contour's minimum-area rectangle (shorter side >= 3 px) scores the mean
  probability inside it and is kept at >= 0.6; it is grown on every side by area x 1.5 / perimeter (PaddleOCR's unclip,
  which for a rectangle is exactly this, so no polygon-clipping package is needed), kept when its shorter side is
  >= 5 px, and scaled back to the frame.
- Recognition input: each box cut out with a perspective warp onto its own rectangle (turned 90 degrees when it is at
  least 1.5 times taller than wide), resized to height 48 keeping its aspect ratio, (x / 255 - 0.5) / 0.5, padded with
  zeros on the right to at least 320 px (at most 3200); NCHW float32. Output: per-step character probabilities
  [N, T, C] (CTC head, softmax), C = 1 (blank) + the dictionary + 1 (space).
- CTC greedy decoding: the most probable class per step, repeats collapsed, blanks dropped; the line's score is the
  mean probability of the kept steps.

A detection is one text line: `box` the axis-aligned box around the detected quadrilateral (a slanted line is
over-covered, never cut), `label` the recognised text, `score` the recognition score, `lang` "multi" (one model for
its languages; nothing is chosen). Vertical Japanese, page segmentation modes and word level are Tesseract's.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import DetectionError
from ..models import ENV_MODELS_DIR, KNOWN_MODELS, MODEL_LICENSES, combined_sha256, describe_file, find_ppocrv6
from .base import DetectionEngine, EngineRequest, EngineResult, Prepared
from .yunet_engine import read_image

LANG = "multi"
DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE, DET_MAX_SIDE_LEN = 64, "min", 4000
DET_MEAN, DET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DB_THRESH, DB_BOX_THRESH, DB_UNCLIP_RATIO, DB_MAX_CANDIDATES, DB_MIN_SIZE = 0.3, 0.6, 1.5, 3000, 3
REC_HEIGHT, REC_MIN_WIDTH, REC_MAX_WIDTH = 48, 320, 3200
ROTATE_RATIO = 1.5
INSTALL = "pip install 'detection-skill[objects]'  (onnxruntime, opencv-python-headless)"

_HF = "https://huggingface.co/PaddlePaddle"
DET_REVISION = "61323801669c338b7891481ec7bac61ce31b576a"   # PP-OCRv6_medium_det_onnx
REC_REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"   # PP-OCRv6_medium_rec_onnx
DOWNLOAD = (
    f'cd "${ENV_MODELS_DIR}"',
    f"curl -fL -o PP-OCRv6_medium_det.onnx {_HF}/PP-OCRv6_medium_det_onnx/resolve/{DET_REVISION}/inference.onnx",
    f"curl -fL -o PP-OCRv6_medium_rec.onnx {_HF}/PP-OCRv6_medium_rec_onnx/resolve/{REC_REVISION}/inference.onnx",
    f"curl -fL -o PP-OCRv6_medium_rec.yml {_HF}/PP-OCRv6_medium_rec_onnx/resolve/{REC_REVISION}/inference.yml",
)
SOURCE = ("optional (detection text --engine ppocrv6): PP-OCRv6 medium (Apache-2.0), the official ONNX exports from "
          f"{_HF}/PP-OCRv6_medium_det_onnx and {_HF}/PP-OCRv6_medium_rec_onnx, saved as PP-OCRv6_medium_det.onnx, "
          f"PP-OCRv6_medium_rec.onnx and PP-OCRv6_medium_rec.yml (the recognizer's inference.yml) in the models directory "
          f"(${ENV_MODELS_DIR}): " + " && ".join(DOWNLOAD) + "; the default text engine (tesseract) needs none of them")


# ---- pieces that need no numpy (unit-tested everywhere) ------------------------------------------
def parse_character_dict(text: str) -> List[str]:
    """The characters of `PostProcess: character_dict:` in a PaddleOCR inference.yml, in order. A small reader for the
    YAML that file uses (one scalar per `- ` item: plain, 'single' or "double" quoted), so no YAML package is needed;
    anything else is refused rather than guessed, because one shifted entry would misread every character after it."""
    out: Optional[List[str]] = None
    in_post = False
    # YAML's own whitespace is space and tab only: str.strip() / splitlines() would also eat entries such as U+3000
    # (the ideographic space, a character of the PP-OCRv6 dictionary) or U+001C.
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        stripped = line.strip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if out is not None:
            if stripped == "-" or stripped.startswith("- "):
                out.append(_scalar(stripped[2:] if stripped != "-" else ""))
                continue
            break
        if indent == 0:
            in_post = stripped == "PostProcess:"
        elif in_post and stripped == "character_dict:":
            out = []
    if not out:
        raise ValueError("no PostProcess.character_dict list")
    return out


_PLAIN_BAD_START = set("'\"|>&*!%@`{}[]#,?:-~")


def _scalar(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return json.loads(v)
    if not v or v[0] in _PLAIN_BAD_START or v in ("null", "Null", "NULL") or " #" in v:
        raise ValueError(f"unsupported character_dict entry {v!r}")
    return v


def characters_for(dictionary: Sequence[str]) -> List[str]:
    """The recognizer's classes: blank first, the dictionary, a space last (PaddleOCR's CTCLabelDecode, use_space_char)."""
    return ["blank"] + list(dictionary) + [" "]


def det_resize_shape(h: int, w: int, limit_side_len: int = DET_LIMIT_SIDE_LEN, limit_type: str = DET_LIMIT_TYPE,
                     max_side_len: int = DET_MAX_SIDE_LEN) -> Tuple[int, int]:
    """(height, width) the detector sees: PaddleOCR's DetResizeForTest (type 0) for an h x w frame."""
    if limit_type == "max":
        ratio = float(limit_side_len) / (h if h > w else w) if max(h, w) > limit_side_len else 1.0
    elif limit_type == "min":
        ratio = float(limit_side_len) / (h if h < w else w) if min(h, w) < limit_side_len else 1.0
    elif limit_type == "resize_long":
        ratio = float(limit_side_len) / max(h, w)
    else:
        raise ValueError(f"limit_type {limit_type!r}")
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > max_side_len:
        r = float(max_side_len) / max(rh, rw)
        rh, rw = int(rh * r), int(rw * r)
    return max(int(round(rh / 32) * 32), 32), max(int(round(rw / 32) * 32), 32)


def rec_widths(h: int, w: int, height: int = REC_HEIGHT, min_width: int = REC_MIN_WIDTH,
               max_width: int = REC_MAX_WIDTH) -> Tuple[int, int]:
    """(resized width, padded input width) of an h x w line crop: PaddleOCR's OCRReisizeNormImg for one image."""
    padded = int(height * max(min_width / height, w / float(h)))
    if padded > max_width:
        return max_width, max_width
    return min(padded, int(math.ceil(height * w / float(h)))), padded


def ctc_decode(indices: Sequence[int], probs: Sequence[float], characters: Sequence[str]) -> Tuple[str, float]:
    """Greedy CTC: repeats collapsed, blank (class 0) dropped; score = mean probability of the kept steps (0 if none)."""
    text: List[str] = []
    kept: List[float] = []
    prev: Optional[int] = None
    for k, p in zip(indices, probs):
        k = int(k)
        if k != prev and k != 0:
            text.append(characters[k])
            kept.append(float(p))
        prev = k
    return "".join(text), (sum(kept) / len(kept) if kept else 0.0)


def order_quad(points: Sequence[Sequence[float]]) -> List[List[float]]:
    """Four corners as [top-left, top-right, bottom-right, bottom-left], PaddleOCR's get_mini_boxes order."""
    pts = sorted([[float(p[0]), float(p[1])] for p in points], key=lambda p: p[0])
    left = pts[:2] if pts[1][1] > pts[0][1] else [pts[1], pts[0]]
    right = pts[2:] if pts[3][1] > pts[2][1] else [pts[3], pts[2]]
    return [left[0], right[0], right[1], left[1]]


def quad_box(quad: Sequence[Sequence[float]]) -> List[float]:
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]


# ---- numpy / OpenCV pieces ------------------------------------------------------------------------
def det_input(img: Any, np: Any, cv2: Any) -> Any:
    h, w = img.shape[:2]
    rh, rw = det_resize_shape(h, w)
    im = img if (rh, rw) == (h, w) else cv2.resize(img, (rw, rh))
    x = (im.astype(np.float32) / 255.0 - np.array(DET_MEAN, np.float32)) / np.array(DET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def _box_score(pred: Any, pts: Any, np: Any, cv2: Any) -> float:
    """Mean probability inside the quadrilateral (PaddleOCR's box_score_fast)."""
    h, w = pred.shape[:2]
    xmin = int(max(0, min(math.floor(pts[:, 0].min()), w - 1)))
    xmax = int(max(0, min(math.ceil(pts[:, 0].max()), w - 1)))
    ymin = int(max(0, min(math.floor(pts[:, 1].min()), h - 1)))
    ymax = int(max(0, min(math.ceil(pts[:, 1].max()), h - 1)))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = pts.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1].astype(np.float32), mask)[0])


def db_boxes(pred: Any, dest_w: int, dest_h: int, np: Any, cv2: Any, thresh: float = DB_THRESH,
             box_thresh: float = DB_BOX_THRESH, unclip_ratio: float = DB_UNCLIP_RATIO,
             max_candidates: int = DB_MAX_CANDIDATES) -> List[Tuple[List[List[float]], float]]:
    """Text quadrilaterals in dest pixels and their scores from a probability map (PaddleOCR's DBPostProcess, quad)."""
    pred = np.asarray(pred, dtype=np.float32)
    height, width = pred.shape
    bitmap = (pred > thresh).astype(np.uint8) * 255
    found = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    sx, sy = dest_w / float(width), dest_h / float(height)
    out: List[Tuple[List[List[float]], float]] = []
    for contour in list(contours)[:max_candidates]:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        if min(rw, rh) < DB_MIN_SIZE:
            continue
        pts = np.array(order_quad(cv2.boxPoints(rect)), dtype=np.float32)
        score = _box_score(pred, pts, np, cv2)
        if score < box_thresh:
            continue
        grow = rw * rh * unclip_ratio / (2.0 * (rw + rh))
        if min(rw, rh) + 2 * grow < DB_MIN_SIZE + 2:
            continue
        quad = order_quad(cv2.boxPoints(((cx, cy), (rw + 2 * grow, rh + 2 * grow), angle)))
        quad = [[float(max(0, min(round(x * sx), dest_w))), float(max(0, min(round(y * sy), dest_h)))] for x, y in quad]
        out.append((quad, score))
    return out


def crop_line(img: Any, quad: Sequence[Sequence[float]], np: Any, cv2: Any) -> Optional[Any]:
    """The line cut out onto its own rectangle (PaddleOCR's get_minarearect + get_rotate_crop_image)."""
    rect = cv2.minAreaRect(np.array(quad, dtype=np.float32).astype(np.int32))
    pts = np.array(order_quad(cv2.boxPoints(rect)), dtype=np.float32)
    cw = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    ch = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if cw < 1 or ch < 1:
        return None
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    m = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, m, (cw, ch), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if crop.shape[0] * 1.0 / crop.shape[1] >= ROTATE_RATIO:
        crop = np.rot90(crop)
    return crop


def rec_input(crop: Any, np: Any, cv2: Any, static_width: Optional[int] = None) -> Any:
    h, w = crop.shape[:2]
    if static_width:
        resized_w, padded_w = static_width, static_width
    else:
        resized_w, padded_w = rec_widths(h, w)
    im = cv2.resize(np.ascontiguousarray(crop), (resized_w, REC_HEIGHT)).astype(np.float32)
    x = (im.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    out = np.zeros((3, REC_HEIGHT, padded_w), dtype=np.float32)
    out[:, :, :resized_w] = x
    return out[None]


def _as_probs(out: Any, np: Any) -> Any:
    """The export ends in a softmax; an export without it gets one here."""
    if out.size and (float(out.min()) < -1e-6 or abs(float(out.sum(-1).mean()) - 1.0) > 1e-2):
        e = np.exp(out - out.max(-1, keepdims=True))
        out = e / e.sum(-1, keepdims=True)
    return out


def read_lines(det: Any, rec: Any, characters: Sequence[str], img: Any, np: Any, cv2: Any,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Text lines in one BGR frame: [{box (frame pixels), score, label, lang, words}], via a detector and a recognizer
    session (anything with onnxruntime's get_inputs() and run())."""
    h, w = img.shape[:2]
    d_in = det.get_inputs()[0]
    prob = np.asarray(det.run(None, {d_in.name: det_input(img, np, cv2)})[0])
    if prob.ndim != 4 or prob.shape[0] != 1 or prob.shape[1] != 1:
        raise DetectionError("ENGINE_FAILED", f"unexpected PP-OCRv6 detector output shape {tuple(prob.shape)} (want [1, 1, H, W])")
    r_in = rec.get_inputs()[0]
    width = r_in.shape[3] if len(r_in.shape) == 4 and isinstance(r_in.shape[3], int) else None
    lines: List[Dict[str, Any]] = []
    for quad, _det_score in db_boxes(prob[0, 0], w, h, np, cv2):
        crop = crop_line(img, quad, np, cv2)
        if crop is None:
            continue
        out = np.asarray(rec.run(None, {r_in.name: rec_input(crop, np, cv2, width)})[0])
        if out.ndim != 3 or out.shape[0] != 1:
            raise DetectionError("ENGINE_FAILED", f"unexpected PP-OCRv6 recognizer output shape {tuple(out.shape)} (want [1, T, C])")
        if out.shape[2] != len(characters):
            raise DetectionError("ENGINE_FAILED", f"the character dictionary ({len(characters) - 2} characters + blank + "
                                                  f"space) does not fit the recognizer's {out.shape[2]} classes: use the "
                                                  "inference.yml of the same model")
        p = _as_probs(out[0], np)
        text, score = ctc_decode(p.argmax(-1).tolist(), p.max(-1).tolist(), characters)
        text = text.strip()
        box = quad_box(quad)
        if text and score >= min_score and box[2] > 0 and box[3] > 0:
            lines.append({"box": box, "score": float(score), "label": text, "lang": LANG, "words": max(1, len(text.split()))})
    return lines


def load_characters(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return characters_for(parse_character_dict(fh.read()))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DetectionError("MODEL_NOT_FOUND", f"{os.path.basename(path)} has no readable character dictionary: {exc}",
                             list(DOWNLOAD))


class PPOCRv6Engine(DetectionEngine):
    id = "ppocrv6"
    kind = "text"
    description = "PP-OCRv6 (PaddleOCR) text detection + recognition, official ONNX exports via onnxruntime, CPU; optional"
    runtime = "onnxruntime"
    license = "Apache-2.0"

    def prepare(self, *, models_dir: Optional[str] = None, model: Optional[str] = None, tessdata_dir: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> Prepared:
        missing = [m for m in ("onnxruntime", "cv2", "numpy") if importlib.util.find_spec(m) is None]
        if missing:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not installed: {', '.join(missing)} (PP-OCRv6 runs through ONNX Runtime)",
                                 INSTALL)
        found = find_ppocrv6(models_dir, how=list(DOWNLOAD))
        paths = found["files"]
        load_characters(paths["dict"])  # refuse an unreadable dictionary before any frame is made
        files = [dict(describe_file(paths[role]), role=role) for role in ("det", "rec", "dict")]
        warnings = [f"model {f['file']}: sha256 not in the skill's table of known files (recorded as measured)"
                    for f in files if not f["known"]]
        tier = found["tier"]
        known = all(f["known"] for f in files)
        name = f"PP-OCRv6_{tier} det + rec" + (" (official ONNX exports)" if known else "")
        return Prepared(model=name, model_sha256=combined_sha256(files), files=files, license=MODEL_LICENSES["ppocrv6"],
                        version=_ort_version(), model_path=paths["rec"], model_files=dict(paths), warnings=warnings,
                        extra={"tier": tier})

    def detect(self, req: EngineRequest) -> EngineResult:
        try:
            import cv2
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise DetectionError("ENGINE_UNAVAILABLE", f"not importable: {exc}", INSTALL)
        files = req.model_files or {}
        if not all(files.get(r) for r in ("det", "rec", "dict")):
            raise DetectionError("MODEL_NOT_FOUND", "PP-OCRv6 needs its detector, recognizer and dictionary files", list(DOWNLOAD))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        det = ort.InferenceSession(files["det"], sess_options=so, providers=["CPUExecutionProvider"])
        rec = ort.InferenceSession(files["rec"], sess_options=so, providers=["CPUExecutionProvider"])
        characters = load_characters(files["dict"])
        min_score = float(req.min_score or 0.0)
        frames: List[Dict[str, Any]] = []
        for fr in req.frames:
            img = read_image(fr["path"])
            h, w = img.shape[:2]
            frames.append({"index": fr["index"], "width": w, "height": h,
                           "detections": read_lines(det, rec, characters, img, np, cv2, min_score)})
        return EngineResult(frames=frames, runtime=runtime_details(len(characters) - 2))


def runtime_details(dictionary_size: int) -> Dict[str, Any]:
    return {"onnxruntime": _ort_version(), "det_limit_side_len": DET_LIMIT_SIDE_LEN, "det_limit_type": DET_LIMIT_TYPE,
            "det_max_side_len": DET_MAX_SIDE_LEN, "db_thresh": DB_THRESH, "db_box_thresh": DB_BOX_THRESH,
            "db_unclip_ratio": DB_UNCLIP_RATIO, "rec_height": REC_HEIGHT, "rec_min_width": REC_MIN_WIDTH,
            "dictionary_size": dictionary_size, "box": "axis-aligned box around the detected quadrilateral"}


def _ort_version() -> str:
    try:
        from importlib.metadata import version
        return version("onnxruntime")
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = ["PPOCRv6Engine", "parse_character_dict", "characters_for", "det_resize_shape", "rec_widths", "ctc_decode",
           "order_quad", "quad_box", "db_boxes", "crop_line", "rec_input", "read_lines", "KNOWN_MODELS"]
