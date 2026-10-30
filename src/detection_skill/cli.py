"""detection CLI. Every command prints one JSON document with --json.

  detection faces talk.mp4 --every 1 -o faces.json --json                 # face boxes once a second
  detection faces talk.mp4 --scenes -o faces.json                          # one frame per scene (ffmpeg-skill scenes)
  detection text slide.png --lang jpn+eng -o text.json --json              # an image is one frame
  detection text poster.mp4 --at 3,7.5 --vertical -o text.json             # vertical Japanese (jpn_vert)
  detection objects hall.mp4 --every 2 --classes person,laptop -o objects.json
  detection tracks faces.json -o tracks.json --json                        # boxes linked over time, redact rectangles
  detection check faces.json --json
  detection contract --json
  detection doctor --json

Exit codes: 0 done, 2 invalid input or arguments, 3 engine, model file, tessdata or ffmpeg-skill missing,
4 engine or frame extraction failed or timed out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .contract import ERRORS, contract
from .detections import DETECTIONS_SCHEMA, validate_detections
from .doctor import doctor
from .errors import DetectionError
from .service import DEFAULT_MAX_FRAMES, DEFAULT_MIN_SCORE, DEFAULT_TIMEOUT, LEVELS, detect
from .tracks import TRACKS_SCHEMA, fingerprint_bytes, make_tracks, validate_tracks

KIND_OF = {"faces": "face", "text": "text", "objects": "object"}


def _emit(doc: Dict[str, Any], as_json: bool, human: str = "") -> None:
    if as_json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    elif human:
        print(human)


def _fail(code: str, message: str, as_json: bool, detail: Optional[Any] = None) -> int:
    doc = {"ok": False, "error": {"code": code, "message": message, "detail": detail}}
    if as_json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(f"error: {message}" + (f" ({detail})" if detail else ""), file=sys.stderr)
    return ERRORS.get(code, {}).get("exit_code", 1)


def _same(a: str, b: str) -> bool:
    return os.path.abspath(a) == os.path.abspath(b) or (os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b))


def _check_output(path: str, inputs: List[str], overwrite: bool) -> None:
    if any(_same(path, i) for i in inputs):
        raise DetectionError("OUTPUT_IS_INPUT", "-o must not be an input file")
    if os.path.exists(path) and not overwrite:
        raise DetectionError("OUTPUT_EXISTS", f"refusing to overwrite {os.path.basename(path)}: pass --overwrite")


def _write_json(path: str, doc: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: str, code: str = "INVALID_DETECTIONS") -> Any:
    if not os.path.isfile(path):
        raise DetectionError("INPUT_NOT_FOUND", f"not found: {os.path.basename(path)}")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise DetectionError(code, f"{os.path.basename(path)} is not JSON: {exc}")


def _times(values: Optional[List[str]]) -> Optional[List[float]]:
    if not values:
        return None
    out: List[float] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                raise DetectionError("INVALID_ARGUMENT", f"--at {part!r} is not a number of seconds")
    return out


def cmd_detect(a: argparse.Namespace) -> int:
    kind = KIND_OF[a.command]
    if a.output:
        _check_output(a.output, [a.input], a.overwrite)
    doc = detect(kind, a.input, every=a.every, at=_times(a.at), scenes=a.scenes, min_score=a.min_score, max_frames=a.max_frames,
                 models_dir=getattr(a, "models_dir", None), model=getattr(a, "model", None), lang=getattr(a, "lang", None),
                 vertical=getattr(a, "vertical", False), psm=getattr(a, "psm", None), level=getattr(a, "level", "line"),
                 classes=getattr(a, "classes", None), tessdata_dir=getattr(a, "tessdata_dir", None), ffmpeg_skill=a.ffmpeg_skill,
                 timeout=a.timeout)
    if a.output:
        _write_json(a.output, doc)
    c = doc["counts"]
    lines = [f"{c['detections']} {kind} detection(s) in {c['frames_with_detections']} of {c['frames']} frame(s)"]
    for fr in doc["frames"]:
        for d in fr["detections"]:
            t = "image" if fr["t"] is None else f"{fr['t']:9.3f}s"
            lab = "" if d["label"] is None else f"  {d['label']}"
            lines.append(f"  {t}  [{', '.join(f'{v:.0f}' for v in d['box'])}]  {d['score']:.2f}{lab}")
    _emit(dict(doc, ok=True, detections_file=os.path.abspath(a.output) if a.output else None), a.json, "\n".join(lines))
    return 0


def cmd_tracks(a: argparse.Namespace) -> int:
    _check_output(a.output, [a.detections], a.overwrite)
    if not os.path.isfile(a.detections):
        raise DetectionError("INPUT_NOT_FOUND", f"not found: {os.path.basename(a.detections)}")
    with open(a.detections, "rb") as fh:
        data = fh.read()
    try:
        doc = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise DetectionError("INVALID_DETECTIONS", f"{os.path.basename(a.detections)} is not JSON: {exc}")
    out = make_tracks(doc, min_iou=a.min_iou, max_shift=a.max_shift, max_gap=a.max_gap, pad=a.pad, extend=a.extend, window=a.window, min_hits=a.min_hits,
                      detections_file=a.detections, detections_fingerprint=fingerprint_bytes(data))
    rep = validate_tracks(out)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "tracks failed validation; nothing written", rep.errors[:10])
    _write_json(a.output, out)
    lines = [f"{out['counts']['tracks']} track(s) from {doc['counts']['detections']} box(es); wrote {a.output}"]
    for t in out["tracks"]:
        r = t["redact"][0]
        lines.append(f"  {t['id']}  {t['start']:.3f}-{t['end']:.3f}s  {t['hits']} box(es)  redact x={r['x']} y={r['y']} "
                     f"w={r['width']} h={r['height']} ({len(t['redact'])} rect(s))")
    _emit(dict(out, ok=True, tracks_file=os.path.abspath(a.output)), a.json, "\n".join(lines))
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    doc = load_json(a.file)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    rep = validate_tracks(doc) if schema == TRACKS_SCHEMA else validate_detections(doc)
    out = dict(rep.to_dict(), schema=schema if schema in (DETECTIONS_SCHEMA, TRACKS_SCHEMA) else None)
    _emit(out, a.json, "valid" if rep.ok else "INVALID\n" + "\n".join(f"  error: {e}" for e in rep.errors))
    return 0 if rep.ok else 2


def cmd_contract(a: argparse.Namespace) -> int:
    _emit(contract(), True)
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    doc = doctor(a.models_dir, a.tessdata_dir, a.ffmpeg_skill)
    human = "OK" if doc["ok"] else "\n".join(f"problem: {p}" for p in doc["problems"])
    _emit(doc, a.json, human)
    return 0 if doc["ok"] else 3


def _unit(v: str) -> float:
    try:
        x = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return x


def _positive(v: str) -> float:
    try:
        x = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not x > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def _nonneg(v: str) -> float:
    try:
        x = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not x >= 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def _add_detect(sub: Any, name: str, help_: str) -> argparse.ArgumentParser:
    kind = KIND_OF[name]
    d = sub.add_parser(name, help=help_)
    d.add_argument("input", help="a video (frames via ffmpeg-skill) or an image (png, jpg, ...)")
    g = d.add_mutually_exclusive_group()
    g.add_argument("--every", type=_positive, help="sample a frame every S seconds (default 1 for a video)")
    g.add_argument("--at", action="append", help="sample at these times, comma-separated and/or repeated")
    g.add_argument("--scenes", action="store_true", help="one frame per scene (ffmpeg-skill scenes, the scene's midpoint)")
    d.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help=f"refuse more frames than this (default {DEFAULT_MAX_FRAMES})")
    d.add_argument("--min-score", type=_unit, help=f"drop boxes scoring below this (default {DEFAULT_MIN_SCORE[kind]})")
    d.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout (default: $DETECTION_FFMPEG_SKILL_DIR, ~/.claude/skills/ffmpeg-skill, ...)")
    d.add_argument("--timeout", type=_positive, default=DEFAULT_TIMEOUT, help=f"seconds for the engine (default {DEFAULT_TIMEOUT:g})")
    d.add_argument("-o", "--output", help="write the detections JSON here")
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--json", action="store_true", help="print one JSON document")
    d.set_defaults(func=cmd_detect)
    return d


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="detection", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"detection-skill {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    f = _add_detect(sub, "faces", "face boxes (YuNet); never an identity")
    f.add_argument("--models-dir", help="directory with face_detection_yunet_2026may.onnx (default: $DETECTION_MODELS_DIR)")
    f.add_argument("--model", help="model file name inside the models directory")

    t = _add_detect(sub, "text", "text lines and their OCR text (Tesseract, ja/en)")
    t.add_argument("--lang", default=None, help="Tesseract languages, e.g. jpn+eng (default), eng, jpn")
    t.add_argument("--vertical", action="store_true", help="vertical Japanese: jpn_vert and page segmentation mode 5")
    t.add_argument("--psm", type=int, help="Tesseract page segmentation mode (default 3; 5 with --vertical; 11 for sparse text)")
    t.add_argument("--level", choices=LEVELS, default="line", help="one box per line (default) or per word")
    t.add_argument("--tessdata-dir", help="directory with jpn/jpn_vert/eng.traineddata (default: $DETECTION_TESSDATA_DIR, "
                                          "else tesseract's own)")

    o = _add_detect(sub, "objects", "object boxes (YOLOX, COCO classes)")
    o.add_argument("--classes", action="append", help="only these COCO classes, comma-separated (e.g. person,laptop)")
    o.add_argument("--models-dir", help="directory with a yolox_*.onnx file (default: $DETECTION_MODELS_DIR)")
    o.add_argument("--model", help="model file name inside the models directory")

    tr = sub.add_parser("tracks", help="link a video's boxes over time into tracks and redact rectangles (heuristic)")
    tr.add_argument("detections")
    tr.add_argument("-o", "--output", required=True, help="write the tracks JSON here")
    tr.add_argument("--min-iou", type=_unit, help="IoU needed to link two boxes (default 0.3)")
    tr.add_argument("--max-shift", type=_nonneg, help="also link a box whose centre moved at most this many box sizes "
                                                      "(default 1.0; 0 = IoU only)")
    tr.add_argument("--max-gap", type=_nonneg, help="seconds a track survives without a box (default 2.5 sampling steps, >= 0.5)")
    tr.add_argument("--pad", type=_nonneg, help="padding of a redact rectangle, fraction of its size per side (default 0.15)")
    tr.add_argument("--extend", type=_nonneg, help="seconds added before the first and after the last box (default one step)")
    tr.add_argument("--window", type=_nonneg, help="one redact rectangle per this many seconds of a track (default 0: one)")
    tr.add_argument("--min-hits", type=int, help="drop tracks with fewer boxes (default 1: keep all, for redaction)")
    tr.add_argument("--overwrite", action="store_true")
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_tracks)

    c = sub.add_parser("check", help="validate a detections or tracks document")
    c.add_argument("file")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    k = sub.add_parser("contract", help="machine-readable contract")
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_contract)

    dr = sub.add_parser("doctor", help="environment check (never downloads)")
    dr.add_argument("--models-dir")
    dr.add_argument("--tessdata-dir")
    dr.add_argument("--ffmpeg-skill")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.func(a)
    except DetectionError as e:
        return _fail(e.code, e.message, getattr(a, "json", False), e.detail)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
