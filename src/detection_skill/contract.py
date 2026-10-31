"""Machine-readable contract (`detection contract --json`), in the shape the AI Video Production OS registry
reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict, List

from . import __version__
from .detections import DETECTIONS_SCHEMA
from .engines import DEFAULT_ENGINE, engine_ids
from .frames import ENV_DIRS, REQUIRED_TOOLS, VERSION_RANGE
from .models import ENV_MODELS_DIR, ENV_TESSDATA_DIR, FACE_MODEL_DEFAULT, KNOWN_MODELS, KNOWN_TESSDATA
from .service import DEFAULT_EVERY, DEFAULT_MAX_FRAMES, DEFAULT_MIN_SCORE, DEFAULT_TIMEOUT
from .tracks import METHOD, TRACKS_SCHEMA

SKILL_ID = "detection"
CONTRACT_VERSION = "1.0"

_SAMPLING = ["[--every S | --at t,... | --scenes]", "[--max-frames N]"]
_COMMON = ["[--min-score X]", "[--ffmpeg-skill <dir>]", f"[--timeout {DEFAULT_TIMEOUT:g}]", "[-o <det.json>]", "[--overwrite]",
           "--json"]
TOOLS: List[Dict[str, Any]] = [
    {"tool_id": "detection/faces", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Face boxes (YuNet) at sampled times of a video, or in an image: time, box in pixels and normalized, "
                    "score, five landmarks, engine, model and model sha256. A face box is never an identity",
     "argv": ["faces", "<video|image>"] + _SAMPLING + ["[--models-dir <dir>]", "[--model <file.onnx>]"] + _COMMON},
    {"tool_id": "detection/text", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Text lines (Tesseract, ja/en, vertical Japanese with jpn_vert) with their boxes and OCR text, from a "
                    "video's sampled frames or an image (a slide, a screenshot)",
     "argv": ["text", "<video|image>"] + _SAMPLING + ["[--lang jpn+eng]", "[--vertical]", "[--psm N]", "[--level line|word]",
                                                      "[--tessdata-dir <dir>]"] + _COMMON},
    {"tool_id": "detection/objects", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Object boxes (YOLOX, COCO's 80 classes) with class and score, optionally only some classes",
     "argv": ["objects", "<video|image>"] + _SAMPLING + ["[--classes person,...]", "[--models-dir <dir>]",
                                                         "[--model <file.onnx>]"] + _COMMON},
    {"tool_id": "detection/tracks", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Link a video's boxes over time by IoU (or centre distance) into track ids and redact-ready rectangles (union per span, "
                    "padded, even pixels) for ffmpeg-skill redact. A documented heuristic, not identity",
     "argv": ["tracks", "<det.json>", "-o <tracks.json>", "[--min-iou 0.3]", "[--max-shift 1.0]", "[--max-gap S]", "[--pad 0.15]", "[--extend S]",
              "[--window S]", "[--min-hits 1]", "[--overwrite]", "--json"]},
    {"tool_id": "detection/check", "role": "validation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": f"Validate a detections ({DETECTIONS_SCHEMA}) or tracks ({TRACKS_SCHEMA}) document",
     "argv": ["check", "<det.json|tracks.json>", "--json"]},
]

ERRORS = {
    "INPUT_NOT_FOUND": {"exit_code": 2, "retryable": False},
    "INVALID_ARGUMENT": {"exit_code": 2, "retryable": False},
    "UNSUPPORTED_MEDIA": {"exit_code": 2, "retryable": False},
    "INVALID_DETECTIONS": {"exit_code": 2, "retryable": False},
    "OUTPUT_IS_INPUT": {"exit_code": 2, "retryable": False},
    "OUTPUT_EXISTS": {"exit_code": 2, "retryable": False},
    "FFMPEG_SKILL_NOT_FOUND": {"exit_code": 3, "retryable": False},
    "ENGINE_UNAVAILABLE": {"exit_code": 3, "retryable": False},
    "MODEL_NOT_FOUND": {"exit_code": 3, "retryable": False},
    "LANGUAGE_NOT_FOUND": {"exit_code": 3, "retryable": False},
    "FRAME_EXTRACTION_FAILED": {"exit_code": 4, "retryable": False},
    "ENGINE_FAILED": {"exit_code": 4, "retryable": False},
    "ENGINE_TIMEOUT": {"exit_code": 4, "retryable": True},
    "INVALID_RESULT": {"exit_code": 4, "retryable": False},
}


def contract() -> Dict[str, Any]:
    located = ["--models-dir", f"${ENV_MODELS_DIR}"]
    return {
        "skill_id": SKILL_ID,
        "id": SKILL_ID,
        "skill": "detection-skill",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "provides": [{"id": "vision.detect_faces", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/faces"},
                     {"id": "vision.detect_text", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/text"},
                     {"id": "vision.detect_objects", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/objects"},
                     {"id": "vision.track_boxes", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/tracks"}],
        "dependencies": [
            {"id": "ffmpeg-skill", "kind": "skill", "required": True, "version_range": VERSION_RANGE,
             "for": ["detection/faces", "detection/text", "detection/objects"], "only_for": "video input (an image needs none)",
             "tools": list(REQUIRED_TOOLS), "located_by": ["--ffmpeg-skill"] + [f"${e}" for e in ENV_DIRS]
             + ["~/.claude/skills/ffmpeg-skill", "./vendor/ffmpeg-skill", "../ffmpeg-skill"]},
            {"id": "opencv-python-headless", "kind": "python-package", "required": False, "for": ["detection/faces", "detection/objects"],
             "version_range": ">=4.8", "install": "pip install 'detection-skill[faces]'"},
            {"id": "onnxruntime", "kind": "python-package", "required": False, "for": ["detection/objects"],
             "version_range": ">=1.16", "install": "pip install 'detection-skill[objects]'"},
            {"id": "tesseract", "kind": "binary", "required": False, "for": ["detection/text"], "version_range": ">=5.0 <6.0",
             "located_by": ["$DETECTION_TESSERACT", "PATH"]},
            {"id": FACE_MODEL_DEFAULT, "kind": "model-weights", "required": False, "for": ["detection/faces"],
             "version_range": "any YuNet face_detection_yunet_*.onnx (sha256 recorded in every result)", "license": "MIT",
             "located_by": located, "downloaded_at_run_time": False},
            {"id": "yolox_*.onnx", "kind": "model-weights", "required": False, "for": ["detection/objects"],
             "version_range": "any YOLOX COCO ONNX export (sha256 recorded in every result)", "license": "Apache-2.0",
             "located_by": located, "downloaded_at_run_time": False},
            {"id": "tessdata_best jpn/jpn_vert/eng", "kind": "model-weights", "required": False, "for": ["detection/text"],
             "version_range": "tessdata 4.0+ LSTM (tessdata_best recommended; sha256 recorded in every result)",
             "license": "Apache-2.0", "located_by": ["--tessdata-dir", f"${ENV_TESSDATA_DIR}", "tesseract's own tessdata"],
             "downloaded_at_run_time": False},
        ],
        "not_provided": ["face recognition or identifying anyone (a face box is never an identity)",
                         "decoding video (frames come from ffmpeg-skill look / scenes)",
                         "blurring or redacting (ffmpeg-skill redact, from the tracks rectangles)",
                         "downloading model files (a one-time manual step, see README)", "network access of any kind at run time",
                         "slide-change detection (slide-sync-skill)", "segmentation masks or pose", "OCR text correction"],
        "tools": TOOLS,
        "detections": {"schema": DETECTIONS_SCHEMA, "schema_file": "schemas/detections.schema.json", "provenance": "OBSERVED",
                       "box": "[x, y, w, h] in source pixels, top-left origin; box_norm divided by frame_size",
                       "time": "t in seconds on the source timeline (null for an image)",
                       "per_detection_provenance": ["engine", "model", "model_sha256"]},
        "tracks": {"schema": TRACKS_SCHEMA, "schema_file": "schemas/tracks.schema.json", "method": METHOD, "heuristic": True,
                   "identity": False},
        "engines": {"default": dict(DEFAULT_ENGINE), "registered": engine_ids(),
                    "known_model_sha256": {k: v["file"] for k, v in KNOWN_MODELS.items()},
                    "known_tessdata_sha256": {k: v["lang"] + " (" + v["variant"] + ")" for k, v in KNOWN_TESSDATA.items()}},
        "defaults": {"every": DEFAULT_EVERY, "max_frames": DEFAULT_MAX_FRAMES, "min_score": dict(DEFAULT_MIN_SCORE),
                     "lang": "jpn+eng"},
        "execution": {"canonical_invocation": ["detection", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess, own process group, killed on --timeout",
                      "frames": "ffmpeg-skill look --at T --no-timecode --width <source width>, PNG in a private temp dir",
                      "input_is_read_only": True, "overwrites": "never unless --overwrite; -o equal to an input is refused"},
        "errors": ERRORS,
    }
