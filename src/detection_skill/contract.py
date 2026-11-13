"""Machine-readable contract (`detection contract --json`), in the shape the AI Video Production OS registry
reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict, List

from . import __version__
from .detections import DETECTIONS_SCHEMA
from .engines import (DEFAULT_ENGINE, DEFAULT_SHOT_ENGINE, ENGINES_BY_KIND, OBJECT_ENGINES, OPTIONAL_ENGINES, SHOT_ENGINES,
                      TEXT_ENGINES, engine_ids)
from .frames import ENV_DIRS, REQUIRED_TOOLS, VERSION_RANGE
from .models import KNOWN_MODELS, KNOWN_TESSDATA
from .pii import CATEGORIES as PII_CATEGORIES, DEFAULT_LOCALES as PII_DEFAULT_LOCALES, PII_SCHEMA, RULES as PII_RULES
from .rank import RANK_SCHEMA, WEIGHTS as RANK_WEIGHTS
from .reframe import METHOD as REFRAME_METHOD, REFRAME_SCHEMA
from .service import DEFAULT_EVERY, DEFAULT_MAX_FRAMES, DEFAULT_MIN_SCORE, DEFAULT_TIMEOUT
from .shots import DEFAULT_MAX_FRAMES_SHOTS, SHOT_WIDTH, SHOTS_SCHEMA
from .speakers import DEFAULTS as SPEAKER_DEFAULTS, METHODS as SPEAKER_METHODS, SPEAKERS_SCHEMA, TURNS_SCHEMA
from .tracks import METHOD, TRACKS_SCHEMA

SKILL_ID = "detection"
CONTRACT_VERSION = "1.0"

_SAMPLING = ["[--every S | --at t,... | --scenes]", "[--max-frames N]"]
_COMMON = ["[--min-score X]", "[--ffmpeg-skill <dir>]", f"[--timeout {DEFAULT_TIMEOUT:g}]", "[-o <det.json>]", "[--overwrite]",
           "--json"]
TOOLS: List[Dict[str, Any]] = [
    {"tool_id": "detection/faces", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Face boxes (YuNet) at sampled times of a video, or in an image: time, box in pixels and normalized, "
                    "score, five landmarks, engine, model and model sha256; with --attributes also each face's measured "
                    "size, pose (yaw, roll, frontal), sharpness, mouth contrast and eyes open (heuristics). A face box is never an identity",
     "argv": ["faces", "<video|image>"] + _SAMPLING + ["[--models-dir <dir>]", "[--model <file.onnx>]", "[--attributes]"] + _COMMON},
    {"tool_id": "detection/text", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Text lines (Tesseract, ja/en, vertical Japanese with jpn_vert) with their boxes and OCR text, from a "
                    "video's sampled frames or an image (a slide, a screenshot); --engine ppocrv6 reads them with PP-OCRv6 "
                    "(optional, ONNX, line level) instead",
     "argv": ["text", "<video|image>"] + _SAMPLING + [f"[--engine {'|'.join(TEXT_ENGINES)}]", "[--lang jpn+eng]", "[--vertical]",
                                                      "[--psm N]", "[--level line|word]", "[--tessdata-dir <dir>]",
                                                      "[--models-dir <dir>]"] + _COMMON},
    {"tool_id": "detection/objects", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Object boxes (YOLOX, COCO's 80 classes) with class and score, optionally only some classes; "
                    "--engine rfdetr detects them with RF-DETR Small / Medium (optional, an ONNX export) instead",
     "argv": ["objects", "<video|image>"] + _SAMPLING + [f"[--engine {'|'.join(OBJECT_ENGINES)}]", "[--classes person,...]",
                                                         "[--models-dir <dir>]", "[--model <file.onnx>]"] + _COMMON},
    {"tool_id": "detection/tracks", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Link a video's boxes over time by IoU (or centre distance) into track ids and redact-ready rectangles (union per span, "
                    "padded, even pixels) for ffmpeg-skill redact. A documented heuristic, not identity",
     "argv": ["tracks", "<det.json>", "-o <tracks.json>", "[--min-iou 0.3]", "[--max-shift 1.0]", "[--max-gap S]", "[--pad 0.15]", "[--extend S]",
              "[--window S]", "[--min-hits 1]", "[--overwrite]", "--json"]},
    {"tool_id": "detection/shots", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Shot boundaries over every frame of a video (frames from ffmpeg-skill look): each shot's start/end in seconds "
                    "and frames, the boundary kind (cut / gradual), a score and the engine. PySceneDetect (content + adaptive "
                    "detectors, a blend test for dissolves) by default; TransNetV2 optional",
     "argv": ["shots", "<video>", f"[--engine {'|'.join(SHOT_ENGINES)}]", "[--threshold X]", "[--min-shot-frames 15]",
              f"[--width {SHOT_WIDTH}]", f"[--max-frames {DEFAULT_MAX_FRAMES_SHOTS}]", "[--models-dir <dir>]", "[--model <file.onnx>]",
              "[--ffmpeg-skill <dir>]", f"[--timeout {DEFAULT_TIMEOUT:g}]", "[-o <shots.json>]", "[--overwrite]", "--json"]},
    {"tool_id": "detection/reframe", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "A crop window per sample time for a target aspect (9:16, 1:1, 4:5) from face tracks (object tracks as a "
                    "fallback), smoothed with a dead zone and a speed limit, switching subject and jumping only at a shot "
                    "boundary; plus one window per shot. A documented heuristic; it crops nothing",
     "argv": ["reframe", "<face-tracks.json>", "-o <reframe.json>", "[--aspect 9:16]", "[--shots <shots.json>]",
              "[--objects <object-tracks.json>]", "[--object-classes person]", "[--dead-zone 0.1]", "[--max-speed 0.6]",
              "[--overwrite]", "--json"]},
    {"tool_id": "detection/speakers", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Active speaker for multicam switching: from each camera's face tracks (faces --attributes) and, when given, "
                    "diarization-skill turns, which face speaks in each time range (mouth motion, or turns matched to faces) and "
                    "which camera shows it best (size, frontal angle, sharpness), as a switch list for ffmpeg-skill multicam "
                    "--switch. A documented heuristic; it cuts nothing",
     "argv": ["speakers", "--camera <cam0-face-tracks.json>", "--camera <cam1-face-tracks.json>", "-o <speakers.json>",
              "[--offsets 0,S,...]", "[--turns <turns.json>]", "[--turns-offset 0]", f"[--window {SPEAKER_DEFAULTS['window']:g}]",
              f"[--min-activity {SPEAKER_DEFAULTS['min_activity']:g}]", f"[--min-shot {SPEAKER_DEFAULTS['min_shot']:g}]",
              "[--overwrite]", "--json"]},
    {"tool_id": "detection/pii", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Personal and patient data in slides and frames: OCR text lines checked for names, patient ids, dates of birth, "
                    "phone numbers and addresses (keyword and format patterns per locale: intl by default, en-US, en-GB, ja-JP "
                    "opt-in) and, when given, faces; each hit a redact-ready rectangle with its category, rule and reason, "
                    "every one a proposal to confirm (CONFIRM). It removes nothing",
     "argv": ["pii", "<text-det.json>", "-o <pii.json>", "[--faces <face-det.json>]", "[--locale intl,...]", "[--pad 0.1]",
              "[--overwrite]", "--json"]},
    {"tool_id": "detection/rank", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Rank a video's sampled frames (or images) for a thumbnail or a take by their faces' measured attributes: "
                    "eyes open, sharpness (blur), frontal angle and size, each part and the weighted score listed. A documented "
                    "heuristic; it suggests, a person or thumbnail-skill picks",
     "argv": ["rank", "<face-det.json>", "-o <rank.json>", "[--top N]", "[--all-faces]", "[--overwrite]", "--json"]},
    {"tool_id": "detection/check", "role": "validation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": f"Validate a detections ({DETECTIONS_SCHEMA}), tracks ({TRACKS_SCHEMA}), shots ({SHOTS_SCHEMA}), reframe "
                    f"({REFRAME_SCHEMA}), speakers ({SPEAKERS_SCHEMA}), pii ({PII_SCHEMA}) or rank ({RANK_SCHEMA}) document",
     "argv": ["check", "<det.json|tracks.json|shots.json|reframe.json|speakers.json|pii.json|rank.json>", "--json"]},
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
    return {
        "skill_id": SKILL_ID,
        "id": SKILL_ID,
        "skill": {"id": SKILL_ID, "package": "detection-skill", "version": __version__},
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "provides": [{"id": "vision.detect_faces", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/faces"},
                     {"id": "vision.detect_text", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/text"},
                     {"id": "vision.detect_objects", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/objects"},
                     {"id": "vision.track_boxes", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/tracks"},
                     {"id": "vision.detect_shots", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/shots"},
                     {"id": "vision.reframe_path", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/reframe"},
                     {"id": "vision.active_speaker", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/speakers"},
                     {"id": "vision.find_personal_data", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/pii"},
                     {"id": "vision.rank_face_frames", "lifecycle": "EXPERIMENTAL", "tool_id": "detection/rank"}],
        # Skill dependencies only (other Skills this one invokes), each with skill_id and version_range: the OS contract
        # shape (Velisa registry/capability_contract.schema.json, ADR-018). Python packages, binaries and model files are
        # not Skills: `doctor` reports each with its version, licence and install line, and `engines` names the engines
        # (as transcription-skill does). docs/decisions.md ADR-009.
        "dependencies": [
            {"skill_id": "ffmpeg-skill", "id": "ffmpeg-skill", "kind": "skill", "required": True, "version_range": VERSION_RANGE,
             "for": ["detection/faces", "detection/text", "detection/objects", "detection/shots"],
             "only_for": "video input (an image needs none)",
             "tools": list(REQUIRED_TOOLS), "located_by": ["--ffmpeg-skill"] + [f"${e}" for e in ENV_DIRS]
             + ["~/.claude/skills/ffmpeg-skill", "./vendor/ffmpeg-skill", "../ffmpeg-skill"]},
        ],
        "not_provided": ["face recognition or identifying anyone (a face box is never an identity)",
                         "decoding video (frames come from ffmpeg-skill look / scenes)",
                         "blurring or redacting (ffmpeg-skill redact, from the tracks rectangles)",
                         "downloading model files (a one-time manual step, see README)", "network access of any kind at run time",
                         "exporting RF-DETR to ONNX (a one-time manual step with the rfdetr package and PyTorch, see README)",
                         "slide-change detection (slide-sync-skill)", "segmentation masks or pose", "OCR text correction",
                         "cropping or applying a reframe path (video-editing-skill, ffmpeg-skill crop)",
                         "cutting between cameras or aligning them by audio (ffmpeg-skill multicam, from the speakers switch list)",
                         "speaker diarization (diarization-skill turns, read when given)",
                         "deciding or removing personal data on its own (every pii region is a CONFIRM proposal; "
                         "ffmpeg-skill redact applies the confirmed ones)", "named-entity recognition (pii is a keyword and format check)"],
        "tools": TOOLS,
        "detections": {"schema": DETECTIONS_SCHEMA, "schema_file": "schemas/detections.schema.json", "provenance": "OBSERVED",
                       "box": "[x, y, w, h] in source pixels, top-left origin; box_norm divided by frame_size",
                       "time": "t in seconds on the source timeline (null for an image)",
                       "per_detection_provenance": ["engine", "model", "model_sha256"]},
        "tracks": {"schema": TRACKS_SCHEMA, "schema_file": "schemas/tracks.schema.json", "method": METHOD, "heuristic": True,
                   "identity": False},
        "shots": {"schema": SHOTS_SCHEMA, "schema_file": "schemas/shots.schema.json", "provenance": "OBSERVED",
                  "frames": "every frame, ffmpeg-skill look --at floor_ms(k / frame_rate)",
                  "shot": "[start_frame, end_frame] inclusive; start = start_frame / rate, end = (end_frame + 1) / rate",
                  "boundary_kinds": ["cut", "gradual"], "engines": list(SHOT_ENGINES), "default_engine": DEFAULT_SHOT_ENGINE},
        "reframe": {"schema": REFRAME_SCHEMA, "schema_file": "schemas/reframe.schema.json", "method": REFRAME_METHOD,
                    "heuristic": True, "identity": False, "applies_crop": False},
        "speakers": {"schema": SPEAKERS_SCHEMA, "schema_file": "schemas/speakers.schema.json", "methods": list(SPEAKER_METHODS),
                     "heuristic": True, "identity": False, "cuts_media": False, "switch_format": "ffmpeg-skill multicam --switch",
                     "timeline": "camera 0 (reference); reference time = camera time + offset (multicam offsets_seconds)"},
        "pii": {"schema": PII_SCHEMA, "schema_file": "schemas/pii.schema.json", "method": "patterns", "heuristic": True,
                "identity": False, "decision": "CONFIRM", "removes": False, "categories": list(PII_CATEGORIES),
                "default_locales": list(PII_DEFAULT_LOCALES),
                "locales": {loc: [{"rule": r[0], "category": r[1], "means": r[3]} for r in rules] for loc, rules in PII_RULES.items()}},
        "rank": {"schema": RANK_SCHEMA, "schema_file": "schemas/face-rank.schema.json", "method": "face-quality",
                 "heuristic": True, "identity": False, "weights": dict(RANK_WEIGHTS),
                 "attributes": "detection faces --attributes: eyes_open, sharpness, frontal, size"},
        "engines": {"default": dict(DEFAULT_ENGINE, shot=DEFAULT_SHOT_ENGINE), "registered": engine_ids(),
                    "by_kind": dict({k: list(v) for k, v in ENGINES_BY_KIND.items()}, shot=list(SHOT_ENGINES)),
                    "optional": list(OPTIONAL_ENGINES),
                    "known_model_sha256": {k: v["file"] for k, v in KNOWN_MODELS.items()},
                    "known_tessdata_sha256": {k: v["lang"] + " (" + v["variant"] + ")" for k, v in KNOWN_TESSDATA.items()}},
        "defaults": {"every": DEFAULT_EVERY, "max_frames": DEFAULT_MAX_FRAMES, "min_score": dict(DEFAULT_MIN_SCORE),
                     "lang": "jpn+eng", "shots_width": SHOT_WIDTH, "shots_max_frames": DEFAULT_MAX_FRAMES_SHOTS},
        "execution": {"canonical_invocation": ["detection", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess, own process group, killed on --timeout",
                      "frames": "ffmpeg-skill look --at T --no-timecode --width <source width>, PNG in a private temp dir",
                      "input_is_read_only": True, "overwrites": "never unless --overwrite; -o equal to an input is refused"},
        "errors": ERRORS,
    }
