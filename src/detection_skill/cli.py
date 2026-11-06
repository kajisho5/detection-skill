"""detection CLI. Every command prints one JSON document with --json.

  detection faces talk.mp4 --every 1 -o faces.json --json                 # face boxes once a second
  detection faces talk.mp4 --scenes -o faces.json                          # one frame per scene (ffmpeg-skill scenes)
  detection text slide.png --lang jpn+eng -o text.json --json              # an image is one frame
  detection text poster.mp4 --at 3,7.5 --vertical -o text.json             # vertical Japanese (jpn_vert)
  detection objects hall.mp4 --every 2 --classes person,laptop -o objects.json
  detection tracks faces.json -o tracks.json --json                        # boxes linked over time, redact rectangles
  detection shots talk.mp4 -o shots.json --json                            # every frame: cuts and dissolves (PySceneDetect)
  detection reframe tracks.json --aspect 9:16 --shots shots.json -o reframe.json   # a crop window per sample time
  detection faces camB.mp4 --every 0.12 --attributes -o b.json                # + size, pose, sharpness, mouth contrast
  detection pii text.json [--faces faces.json] [--locale intl,ja-JP] -o pii.json   # personal / patient data to confirm
  detection rank faces.json -o rank.json --top 5                              # frames for a thumbnail: eyes open, sharp, frontal
  detection speakers --camera a_tracks.json --camera b_tracks.json [--turns turns.json] -o speakers.json  # multicam switch list
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
from .engines import DEFAULT_SHOT_ENGINE, SHOT_ENGINES
from .errors import DetectionError
from .pii import PII_SCHEMA, make_pii, validate_pii
from .rank import RANK_SCHEMA, make_rank, validate_rank
from .reframe import REFRAME_SCHEMA, make_reframe, validate_reframe
from .service import DEFAULT_MAX_FRAMES, DEFAULT_MIN_SCORE, DEFAULT_TIMEOUT, LEVELS, detect
from .shots import DEFAULT_MAX_FRAMES_SHOTS, SHOT_WIDTH, SHOTS_SCHEMA, detect_shots, validate_shots
from .speakers import SPEAKERS_SCHEMA, make_speakers, validate_speakers
from .tracks import TRACKS_SCHEMA, fingerprint_bytes, make_tracks, validate_tracks

VALIDATORS = {DETECTIONS_SCHEMA: validate_detections, TRACKS_SCHEMA: validate_tracks, SHOTS_SCHEMA: validate_shots,
              REFRAME_SCHEMA: validate_reframe, SPEAKERS_SCHEMA: validate_speakers,
              PII_SCHEMA: validate_pii, RANK_SCHEMA: validate_rank}

KIND_OF = {"faces": "face", "text": "text", "objects": "object"}


def _json_text(doc: Dict[str, Any]) -> str:
    """UTF-8 JSON, or ASCII-escaped JSON when stdout cannot encode it (a Windows cp1252 console): valid JSON either way."""
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    try:
        text.encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        text = json.dumps(doc, ensure_ascii=True, indent=2)
    return text


def _emit(doc: Dict[str, Any], as_json: bool, human: str = "") -> None:
    if as_json:
        print(_json_text(doc))
    elif human:
        print(human)


def _fail(code: str, message: str, as_json: bool, detail: Optional[Any] = None) -> int:
    doc = {"ok": False, "error": {"code": code, "message": message, "detail": detail}}
    if as_json:
        print(_json_text(doc))
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
                 timeout=a.timeout, attributes=getattr(a, "attributes", False))
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


def _read_doc(path: str) -> Any:
    """(document, fingerprint of its bytes)."""
    if not os.path.isfile(path):
        raise DetectionError("INPUT_NOT_FOUND", f"not found: {os.path.basename(path)}")
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        return json.loads(data.decode("utf-8")), fingerprint_bytes(data)
    except ValueError as exc:
        raise DetectionError("INVALID_DETECTIONS", f"{os.path.basename(path)} is not JSON: {exc}")


def cmd_shots(a: argparse.Namespace) -> int:
    if a.output:
        _check_output(a.output, [a.input], a.overwrite)
    doc = detect_shots(a.input, engine_id=a.engine, threshold=a.threshold, min_shot_frames=a.min_shot_frames, width=a.width,
                       max_frames=a.max_frames, models_dir=a.models_dir, model=a.model, ffmpeg_skill=a.ffmpeg_skill,
                       timeout=a.timeout)
    if a.output:
        _write_json(a.output, doc)
    c = doc["counts"]
    lines = [f"{c['shots']} shot(s) in {c['frames']} frame(s): {c['cuts']} cut(s), {c['gradual']} gradual ({doc['engine']['id']})"]
    for s in doc["shots"]:
        lines.append(f"  {s['id']}  frames {s['start_frame']}-{s['end_frame']}  {s['start']:.3f}-{s['end']:.3f}s  {s['starts_with']}")
    _emit(dict(doc, ok=True, shots_file=os.path.abspath(a.output) if a.output else None), a.json, "\n".join(lines))
    return 0


def cmd_reframe(a: argparse.Namespace) -> int:
    inputs = [a.tracks] + [p for p in (a.shots, a.objects) if p]
    _check_output(a.output, inputs, a.overwrite)
    faces, ffp = _read_doc(a.tracks)
    files = {"faces": (a.tracks, ffp)}
    shots = objects = None
    if a.shots:
        shots, sfp = _read_doc(a.shots)
        files["shots"] = (a.shots, sfp)
    if a.objects:
        objects, ofp = _read_doc(a.objects)
        files["objects"] = (a.objects, ofp)
    classes = [c for v in (a.object_classes or []) for c in v.split(",")] or None
    out = make_reframe(faces, aspect=a.aspect, shots=shots, objects=objects, object_classes=classes, dead_zone=a.dead_zone,
                       max_speed=a.max_speed, files=files)
    rep = validate_reframe(out)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "reframe failed validation; nothing written", rep.errors[:10])
    _write_json(a.output, out)
    w = out["window"]
    lines = [f"{out['counts']['windows']} window(s) {w['width']}x{w['height']} ({out['aspect']['name']}), "
             f"{out['counts']['switches']} subject switch(es); wrote {a.output}"]
    for s in out["shots"]:
        lines.append(f"  shot {s['index']}  {s['start']:.3f}-{s['end']:.3f}s  x={s['x']} y={s['y']}  subject {s['subject']}")
    _emit(dict(out, ok=True, reframe_file=os.path.abspath(a.output)), a.json, "\n".join(lines))
    return 0


def _floats(text: Optional[str], what: str) -> Optional[List[float]]:
    if text is None:
        return None
    try:
        return [float(p) for p in text.split(",") if p.strip()]
    except ValueError:
        raise DetectionError("INVALID_ARGUMENT", f"{what} {text!r}: expected comma-separated seconds")


def cmd_speakers(a: argparse.Namespace) -> int:
    inputs = list(a.camera) + ([a.turns] if a.turns else [])
    _check_output(a.output, inputs, a.overwrite)
    cams, refs = [], []
    for path in a.camera:
        doc, fp = _read_doc(path)
        cams.append(doc)
        refs.append((path, fp))
    files: Dict[str, Any] = {"cameras": refs}
    turns = None
    if a.turns:
        turns, tfp = _read_doc(a.turns)
        files["turns"] = (a.turns, tfp)
    out = make_speakers(cams, offsets=_floats(a.offsets, "--offsets"), turns=turns, turns_offset=a.turns_offset,
                        window=a.window, min_activity=a.min_activity, min_shot=a.min_shot, files=files)
    rep = validate_speakers(out)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "speakers failed validation; nothing written", rep.errors[:10])
    _write_json(a.output, out)
    lines = [f"{out['counts']['ranges']} range(s), {out['counts']['switches']} camera switch(es) ({out['method']['name']}); "
             f"wrote {a.output}", f"  multicam --switch \"{out['switch']}\""]
    for r in out["ranges"]:
        who = f"  {r['speaker']}" if r["speaker"] else ""
        lines.append(f"  {r['start']:9.3f}-{r['end']:9.3f}s  camera {r['camera']}  {r['track'] or '-'}{who}")
    _emit(dict(out, ok=True, speakers_file=os.path.abspath(a.output)), a.json, "\n".join(lines))
    return 0


def cmd_pii(a: argparse.Namespace) -> int:
    _check_output(a.output, [a.text] + ([a.faces] if a.faces else []), a.overwrite)
    text, tfp = _read_doc(a.text)
    files: Dict[str, Any] = {"text": (a.text, tfp)}
    faces = None
    if a.faces:
        faces, ffp = _read_doc(a.faces)
        files["faces"] = (a.faces, ffp)
    out = make_pii(text, faces=faces, locales=a.locale, pad=a.pad, files=files)
    rep = validate_pii(out)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "pii failed validation; nothing written", rep.errors[:10])
    _write_json(a.output, out)
    lines = [f"{out['counts']['regions']} region(s) to confirm ({', '.join(out['parameters']['locales'])}); wrote {a.output}"]
    for r in out["regions"]:
        t = "image" if r["t"] is None else f"{r['t']:9.3f}s"
        x = r["redact"]
        lines.append(f"  {r['id']}  {t}  {r['category']:<13}  x={x['x']} y={x['y']} w={x['width']} h={x['height']}  "
                     f"{r['excerpt'] or ''}")
    _emit(dict(out, ok=True, pii_file=os.path.abspath(a.output)), a.json, "\n".join(lines))
    return 0


def cmd_rank(a: argparse.Namespace) -> int:
    _check_output(a.output, [a.faces], a.overwrite)
    faces, fp = _read_doc(a.faces)
    out = make_rank(faces, all_faces=a.all_faces, top=a.top, file=a.faces, fingerprint=fp)
    rep = validate_rank(out)
    if not rep.ok:
        raise DetectionError("INVALID_RESULT", "rank failed validation; nothing written", rep.errors[:10])
    _write_json(a.output, out)
    lines = [f"{out['counts']['ranked']} frame(s) ranked, {out['counts']['without_face']} without a face; wrote {a.output}"]
    for f in out["frames"]:
        if f["rank"] is None:
            continue
        b = f["faces"][f["best_face"]]["parts"]
        t = "image" if f["t"] is None else f"{f['t']:9.3f}s"
        lines.append(f"  #{f['rank']:<3} {t}  score {f['score']:.3f}  eyes {b['eyes_open']}  sharp {b['sharpness']}  "
                     f"frontal {b['frontal']}  size {b['size']}")
    _emit(dict(out, ok=True, rank_file=os.path.abspath(a.output)), a.json, "\n".join(lines))
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    doc = load_json(a.file)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    rep = VALIDATORS.get(schema, validate_detections)(doc)
    out = dict(rep.to_dict(), schema=schema if schema in VALIDATORS else None)
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
    f.add_argument("--attributes", action="store_true", help="also measure each face's size, pose (yaw, roll, frontal), "
                                                             "sharpness, mouth contrast and eyes open (heuristics, ADR-005, "
                                                             "ADR-007)")

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

    sh = sub.add_parser("shots", help="shot boundaries (cuts, gradual) over every frame of a video (PySceneDetect)")
    sh.add_argument("input", help="a video (every frame via ffmpeg-skill look)")
    sh.add_argument("--engine", choices=SHOT_ENGINES, default=DEFAULT_SHOT_ENGINE,
                    help=f"{DEFAULT_SHOT_ENGINE} (default) or transnetv2 (optional, needs transnetv2.onnx)")
    sh.add_argument("--threshold", type=_positive, help="cut threshold (PySceneDetect content score, default 27; "
                                                        "TransNetV2 probability, default 0.5)")
    sh.add_argument("--min-shot-frames", type=int, help="shortest shot in frames (default 15)")
    sh.add_argument("--width", type=int, default=SHOT_WIDTH, help=f"width of the decoded frames (default {SHOT_WIDTH})")
    sh.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES_SHOTS,
                    help=f"refuse a video with more frames (default {DEFAULT_MAX_FRAMES_SHOTS})")
    sh.add_argument("--models-dir", help="directory with transnetv2.onnx (--engine transnetv2; default: $DETECTION_MODELS_DIR)")
    sh.add_argument("--model", help="model file name inside the models directory (--engine transnetv2)")
    sh.add_argument("--ffmpeg-skill", help="ffmpeg-skill checkout (default: $DETECTION_FFMPEG_SKILL_DIR, ...)")
    sh.add_argument("--timeout", type=_positive, default=DEFAULT_TIMEOUT, help=f"seconds for the engine (default {DEFAULT_TIMEOUT:g})")
    sh.add_argument("-o", "--output", help="write the shots JSON here")
    sh.add_argument("--overwrite", action="store_true")
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_shots)

    rf = sub.add_parser("reframe", help="a crop window per sample time for a target aspect, from face tracks (heuristic)")
    rf.add_argument("tracks", help="face tracks (detection faces -> detection tracks)")
    rf.add_argument("--aspect", default="9:16", help="target aspect W:H (default 9:16; e.g. 1:1, 4:5)")
    rf.add_argument("--shots", help="shots JSON (detection shots): the subject may change and the window jump only at a boundary")
    rf.add_argument("--objects", help="object tracks JSON: fallback subject when no face is in view")
    rf.add_argument("--object-classes", action="append", help="object classes used as a subject (default person)")
    rf.add_argument("--dead-zone", type=_nonneg, help="no move while the subject is this fraction of the window width from "
                                                      "the centre (default 0.1)")
    rf.add_argument("--max-speed", type=_positive, help="speed limit inside a shot, window widths per second (default 0.6)")
    rf.add_argument("-o", "--output", required=True, help="write the reframe JSON here")
    rf.add_argument("--overwrite", action="store_true")
    rf.add_argument("--json", action="store_true")
    rf.set_defaults(func=cmd_reframe)

    sp = sub.add_parser("speakers", help="active speaker per time range and the best camera for it: a switch list for "
                                         "ffmpeg-skill multicam (heuristic)")
    sp.add_argument("--camera", action="append", required=True,
                    help="face tracks of one camera (detection faces --attributes -> detection tracks); camera 0 first, "
                         "repeat for each camera, in multicam's input order")
    sp.add_argument("--offsets", help="seconds per camera, comma-separated, as multicam --offsets-only reports them "
                                      "(reference time = camera time + offset; default all 0)")
    sp.add_argument("--turns", help="diarization-skill turns JSON: switch at each turn, the speaker matched to a face")
    sp.add_argument("--turns-offset", type=float, default=0.0, help="seconds added to the turns' times (default 0: "
                                                                     "already on camera 0's timeline)")
    sp.add_argument("--window", type=_positive, help="seconds of mouth contrast per activity measurement (default 0.6)")
    sp.add_argument("--min-activity", type=_nonneg, help="mouth activity a speaking face needs (default 0.02)")
    sp.add_argument("--min-shot", type=_nonneg, help="shortest range in seconds; shorter ones are folded in (default 1.0)")
    sp.add_argument("-o", "--output", required=True, help="write the speakers JSON here")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_speakers)

    pi = sub.add_parser("pii", help="personal and patient data in OCR text (and faces): redact-ready regions, each a proposal "
                                    "to confirm (patterns per locale)")
    pi.add_argument("text", help="text detections (detection text, --level line)")
    pi.add_argument("--faces", help="face detections of the same file (detection faces): each face is a region to confirm")
    pi.add_argument("--locale", action="append", help="pattern profiles, comma-separated and/or repeated: intl (default), "
                                                      "en-US, en-GB, ja-JP")
    pi.add_argument("--pad", type=_unit, help="padding of a redact rectangle, fraction of the box per side (default 0.1)")
    pi.add_argument("-o", "--output", required=True, help="write the pii JSON here")
    pi.add_argument("--overwrite", action="store_true")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=cmd_pii)

    rk = sub.add_parser("rank", help="rank frames for a thumbnail or take: eyes open, sharp, frontal, large faces (heuristic)")
    rk.add_argument("faces", help="face detections with attributes (detection faces --attributes)")
    rk.add_argument("--top", type=int, help="keep only the best N frames")
    rk.add_argument("--all-faces", action="store_true", help="score a frame by the mean of all its faces (group shots), "
                                                             "not its best face")
    rk.add_argument("-o", "--output", required=True, help="write the rank JSON here")
    rk.add_argument("--overwrite", action="store_true")
    rk.add_argument("--json", action="store_true")
    rk.set_defaults(func=cmd_rank)

    c = sub.add_parser("check", help="validate a detections, tracks, shots, reframe, speakers, pii or rank document")
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
