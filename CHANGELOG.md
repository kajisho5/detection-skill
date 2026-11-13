# Changelog

## 0.2.3 - 2026-11-13

Contract shape only; no command, option, output document, tool, capability id or default changed. `contract_version` stays 1.0.

- The contract follows the OS contract shape (Velisa `registry/capability_contract.schema.json`, ADR-009): `skill` is `{id, package, version}` instead of the package name as a string, and `dependencies` lists only ffmpeg-skill, with `skill_id` added (`id`, `kind`, `version_range` and the rest stay). OpenCV, PySceneDetect, ONNX Runtime, Tesseract and the model files leave `dependencies`; `doctor` reports each with its version, licence and install line, as before.

## 0.2.2 - 2026-11-12

Fixes; no command, option, output key or default changed.

- First run of `heavy-tests.yml` on the real files: all 8 real-model tests pass, no defect in PP-OCRv6 or RF-DETR pre- or post-processing, class mapping or box coordinates. Measured: PP-OCRv6 medium English / Japanese CER 0.000 on rendered slides; RF-DETR Small person / cat / cup IoU 0.97 / 0.97 / 0.98 (README "Measured here", ADR-008).
- The RF-DETR `export` commands (doctor, contract `get`, `MODEL_NOT_FOUND` detail, README) check `$DETECTION_MODELS_DIR` first, before PyTorch is installed.
- `heavy-tests.yml` fails on any skipped test explicitly; doc comments say which RF-DETR file is picked when neither preferred name is there.

## 0.2.1 - 2026-11-12

Fixes and tests; no command, option, output key or default changed.

- `tests/test_real_models.py` and the manual workflow `.github/workflows/heavy-tests.yml` (`workflow_dispatch` only, never on a pull request): PP-OCRv6 medium (the files fetched with the download commands `doctor` prints, sha256 checked), RF-DETR Small (exported with the commands `doctor` prints; PyTorch only in the export job) and YuNet on real model files: rendered English and Japanese slides (CER, line boxes), the three fixture photos and a 1280x720 frame with two of them at known places (IoU), the portrait's face and landmarks. They skip without the files; `DETECTION_REQUIRE_REAL_MODELS=1` (set by the workflow) makes a missing piece fail. The integration job lets only these skips through.
- The commands `doctor` prints (PP-OCRv6 `download`, RF-DETR `export`, the same in the contract's `get` and in `MODEL_NOT_FOUND` details) stop when `$DETECTION_MODELS_DIR` is unset, instead of downloading into the current directory or copying to `/`; the README runs them as one `&&` chain, so a failed `cd` downloads nothing.
- SKILL.md: run the optional engines when `doctor --json` shows `optional_ready.text_ppocrv6` / `objects_rfdetr` true (the files and ONNX Runtime), not only the file check.
- Licence notes: RF-DETR Nano and Large are Apache-2.0 too (the skill still looks for Small and Medium only); Atto, Femto, Pico, XL and 2XL are under the Roboflow PML 1.0.

## 0.2.0 - 2026-11-12

Everything below since 0.1.0 (roadmap steps 8-13). Additive: no command, option, output key or default was removed.

- Optional second engines (roadmap step 13, ADR-008), both through ONNX Runtime with no new package (the `[objects]` extra), both opt-in with `--engine`; the defaults stay Tesseract and YOLOX:
  - `detection text --engine ppocrv6`: PP-OCRv6 (Apache-2.0) text detection + recognition from the official ONNX exports (`PP-OCRv6_medium_det.onnx`, `PP-OCRv6_medium_rec.onnx` and the recognizer's `inference.yml` as `PP-OCRv6_medium_rec.yml`, or the small tier), PaddleOCR's pipeline pre- and post-processing (DB detector, CTC recognizer). One detection per line: the axis-aligned box around the detected quadrilateral, the text, the recognition score, `lang: multi`; every file's sha256 and role in `engine.files`. `--lang`, `--vertical`, `--psm`, `--tessdata-dir` and `--level word` are refused with it. `text` gains `--models-dir`.
  - `detection objects --engine rfdetr`: RF-DETR Small / Medium (Apache-2.0) from an ONNX export the user makes once with `rfdetr[onnx]` (PyTorch for the export only, never at run time), rfdetr's ONNX decoding (a sigmoid per class, the best query/class pairs, no NMS), COCO's sparse ids mapped to the same 80 names as YOLOX. A file not named `rfdetr-small.onnx` / `rfdetr-medium.onnx` runs with `model_license: unknown (not checked)` and a warning.
  - `doctor` adds the optional checks `ppocrv6_models` (with `download`, the pinned commands) and `rfdetr_model` (with `export`), and `optional_ready.text_ppocrv6` / `objects_rfdetr`; they never fail doctor. A run without the files exits 3 (`MODEL_NOT_FOUND`) with the same commands as its detail. The contract adds `--engine` to `detection/text` and `detection/objects`, two optional model-weights dependencies, `engines.by_kind` and `engines.optional`; the six PP-OCRv6 medium / small files join `engines.known_model_sha256`.
  - Tested without the real models: the pieces with stub sessions and the whole engines (worker, ONNX Runtime, document) with tiny synthetic ONNX models written by `tests/tiny_onnx.py`. The real PP-OCRv6 and RF-DETR models are not measured.

- `eyes_open` joins `detection faces --attributes` (roadmap step 12, ADR-007): the share of edges across the eye in both eye regions around YuNet's landmarks, rescaled to 0-1 (about 0.8 open, 0.2 closed); null when the head is rolled over 25 degrees. A heuristic.
- `detection rank` (roadmap step 12, ADR-007): face detections with attributes -> `detection-skill/face-rank@1` (`schemas/face-rank.schema.json`): frames best first by 0.35 eyes open + 0.25 sharpness + 0.20 frontal + 0.20 size (each part listed, missing parts named), `--top N`, `--all-faces` for group shots; frames without a face last. The contract adds `vision.rank_face_frames` (EXPERIMENTAL); `doctor` adds `ready.rank`.

- `detection pii` (roadmap step 11, ADR-006): text detections (+ face detections with `--faces`) -> `detection-skill/pii@1` (`schemas/pii.schema.json`): names, patient ids, dates of birth, phone numbers and addresses found by keyword and format patterns per locale (`intl` by default; `en-US`, `en-GB`, `ja-JP` opt-in, every rule listed in the contract), and faces, each as the line's (face's) box padded to an even-pixel redact rectangle with its rule and reason; joined into redact spans over time. Every region is `CONFIRM`; nothing is removed, the value is masked in `excerpt`. The contract adds `vision.find_personal_data` (EXPERIMENTAL); `doctor` adds `ready.pii`.

- `detection faces --attributes` (ADR-005): each face also gets measured `attributes`: `size` (box height / frame height), `yaw_deg`, `roll_deg`, `frontal` (from the five landmarks), `sharpness` (Laplacian variance at 96x96) and `mouth_contrast`. Heuristics, off by default; `detection tracks` keeps them per box, and the validators check their ranges.
- `detection speakers` (roadmap step 10, ADR-005): face tracks from two or more cameras (+ diarization-skill turns with `--turns`) -> `detection-skill/speakers@1` (`schemas/speakers.schema.json`): per time range on camera 0's timeline the speaking face (mouth motion, or turns matched to faces), the camera with its best view (size, frontal angle, sharpness) and the reason, and `switch`, the ranges as ffmpeg-skill `multicam --switch`. `--offsets` takes multicam's `offsets_seconds`. The contract adds `vision.active_speaker` (EXPERIMENTAL); `doctor` adds `ready.speakers`. Measured on a two-camera fixture with alternating speakers: with turns the switch list is the turns; with mouth motion alone every switch is within 0.12 s.

- `detection shots` (roadmap step 8, ADR-003): shot boundaries over every frame of a video, as `detection-skill/shots@1` (`schemas/shots.schema.json`): each shot's first/last frame and start/end seconds, the boundary kind (`cut` / `gradual`), a 0-1 score and the engine. Frames come from ffmpeg-skill `look --at` (frame k at its time rounded down to the millisecond). Default engine PySceneDetect (BSD-3-Clause, ContentDetector + AdaptiveDetector) with a blend test for dissolves and fades; optional TransNetV2 (MIT) as `transnetv2.onnx` through ONNX Runtime, which `doctor` reports `MISSING` (optional, under `notes` / `optional_ready`) without failing doctor or the default op. New extra `[shots]` (`scenedetect>=0.6.7,<0.7`), also in `[all]`.
- `detection reframe` (roadmap step 9, ADR-004): face tracks (+ object tracks with `--objects`, + shots with `--shots`) -> `detection-skill/reframe@1` (`schemas/reframe.schema.json`): a crop window per sample time for `--aspect` (9:16, 1:1, 4:5, ...), following the subject with a dead zone and a speed limit, switching subject and jumping only at a shot boundary, every switch with its reason, and one fixed window per shot. A documented heuristic; it crops nothing.
- `detection check` validates shots and reframe documents; the contract adds `vision.detect_shots` and `vision.reframe_path` (EXPERIMENTAL); `doctor` adds `scenedetect` and the optional `transnetv2_model`, and `ready.shots` / `ready.reframe`.
- Measured on the CI fixtures: every hard cut within one frame (on its frame) and both dissolves found on a 225-frame lavfi fixture; a face walking across the frame stays inside the 9:16 / 1:1 / 4:5 window in every sampled frame.

- Accept ffmpeg-skill 3.x: the dependency range is `>=2.0.0 <4.0.0` (contract, `detection doctor`, frame extraction). The tools used (`probe`, `scenes`, `look`) did not change in 3.0.0; its `cut.py` changes do not reach this skill (ADR-002).
- ffmpeg-skill 3.x needs Python 3.10+, and its scripts run under this skill's interpreter, so on Python 3.9 a 3.x checkout is refused with `FFMPEG_SKILL_NOT_FOUND` ("needs Python 3.10 or later") and `doctor` reports it `INCOMPATIBLE` with the fix. This skill itself keeps Python 3.9 with an ffmpeg-skill 2.x checkout.
- CI: the integration job runs against ffmpeg-skill `v2.24.1` and `v3.0.0` (tags, not `main`).

## 0.1.0 - 2026-10-30

First release (roadmap steps 2-6).

- `detection faces`, `detection text`, `detection objects`: a video (frames from ffmpeg-skill `look`, sampled `--every S`, `--at t,...` or `--scenes`) or an image -> detections (`detection-skill/detections@1`, `schemas/detections.schema.json`, `docs/detections.md`). Each box carries the frame time, the box in source pixels and normalized, the score, the label (OCR text and language for text, COCO class for objects, null for faces), and the engine, model and model sha256.
  - Faces: YuNet (`face_detection_yunet_2026may.onnx`, MIT) via OpenCV `FaceDetectorYN`, a second downscaled pass for large faces, five landmarks. A face box is never an identity: the validator rejects labels and identity keys on faces.
  - Text: the `tesseract` binary with `jpn` / `jpn_vert` / `eng` (tessdata_best recognised by sha256), line or word level, vertical Japanese with `--vertical` (jpn_vert, psm 5).
  - Objects: YOLOX ONNX via ONNX Runtime (COCO 80 classes, `--classes` filter).
  - Engines run in a worker subprocess in their own process group with `--timeout`. Model files come from `--models-dir` / `DETECTION_MODELS_DIR` and `--tessdata-dir` / `DETECTION_TESSDATA_DIR`; nothing is downloaded at run time.
- `detection tracks`: greedy IoU linking of a video's boxes into `track_nnn` ids with redact-ready rectangles (padded union per span, even pixels, extended by one sampling step) for ffmpeg-skill `redact` (`detection-skill/tracks@1`). Documented as a heuristic, not identity.
- `detection check` (detections and tracks validators), `detection contract` (OS registry shape: four EXPERIMENTAL `provides`, dependencies with version ranges, ffmpeg-skill `>=2.0.0 <3.0.0`), `detection doctor` (MISSING / INCOMPLETE / INCOMPATIBLE with install lines and model sources; exit 3).
- Measured on the test fixtures: YuNet recall 5/5; YOLOX-Nano 3/3; Tesseract CER ja 0.019 (apt tessdata) / 0.057 (tessdata_best), en 0.000, vertical ja 0.000. Not measured: real event footage.
