# Changelog

## Unreleased

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
