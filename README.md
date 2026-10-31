<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/app-icon-dark.png">
    <img src="brand/app-icon.png" width="128" alt="detection-skill app icon">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/detection-skill-wordmark-white.svg">
    <img src="brand/logo/detection-skill-wordmark-black.svg" height="48" alt="detection-skill">
  </picture>
</p>

<p align="center"><b>Faces, text and objects, as boxes in time</b></p>


Faces, text and objects in video frames, as measured boxes in time, for the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS) (department 認識 / Recognition). Each box carries the frame time, the box in pixels and normalized, a score, and the engine, model and model-file sha256 that produced it. Boxes over time become redact-ready rectangles for [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) `redact`.

Local, offline, CPU first ([ADR-001](docs/decisions.md)): faces with [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) through OpenCV, text with [Tesseract 5](https://github.com/tesseract-ocr/tesseract) and [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) `jpn` / `jpn_vert` / `eng` (Apache-2.0), objects with [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) through ONNX Runtime. Every engine is optional. Frames of a video come from ffmpeg-skill `look` and `scenes`; this skill never decodes video and never downloads anything. A face box is never an identity.

```bash
pip install -e '.[all]'                       # the skill + OpenCV + ONNX Runtime (faces and objects)
export DETECTION_MODELS_DIR=~/models/detection # YuNet and YOLOX .onnx files (see below)
detection doctor                              # what is missing, and how to get it
detection faces talk.mp4 --every 0.5 -o faces.json
detection tracks faces.json -o tracks.json    # redact rectangles per face track
detection text slide.png -o text.json         # jpn+eng by default
detection objects hall.mp4 --every 2 --classes person,laptop -o objects.json
```

`tracks` and `check` need only the Python standard library (`pip install -e .`, Python >= 3.9).

## Install, per engine

| Engine | Install | Model files (a directory you point to; never downloaded by the skill) |
|---|---|---|
| Faces: YuNet | `pip install -e '.[faces]'` (opencv-python-headless) | `face_detection_yunet_2026may.onnx` from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), **MIT**. A Git LFS file: download it from the GitHub page, or `git lfs pull` in a clone. Put it in `$DETECTION_MODELS_DIR` |
| Text: Tesseract 5 | the `tesseract` binary: `apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert`, `brew install tesseract tesseract-lang`, or `choco install tesseract` | `jpn.traineddata`, `jpn_vert.traineddata`, `eng.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best), **Apache-2.0**, in a directory set as `--tessdata-dir` / `$DETECTION_TESSDATA_DIR`. Without it, Tesseract's own data is used (on Ubuntu, the `tessdata` variant from apt) and every result says which variant it was |
| Objects: YOLOX | `pip install -e '.[objects]'` (onnxruntime, opencv-python-headless) | a YOLOX ONNX file from the [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases) (e.g. `yolox_s.onnx`, or `yolox_nano.onnx` for speed), **Apache-2.0**, in `$DETECTION_MODELS_DIR`. COCO's 80 classes |
| Frames of a video | [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) 2.x or 3.x and FFmpeg (3.x needs Python 3.10+ and FFmpeg 6.1+) | `git clone https://github.com/kajisho5/ffmpeg-skill ~/.claude/skills/ffmpeg-skill`, or `--ffmpeg-skill` / `$DETECTION_FFMPEG_SKILL_DIR` |

Model files are identified by sha256, measured on every run and written into every result. The skill knows these files (`detection contract --json`, `engines.known_model_sha256`):

| File | sha256 |
|---|---|
| `face_detection_yunet_2026may.onnx` | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |
| `yolox_nano.onnx` (YOLOX release 0.1.1rc0) | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| tessdata_best `jpn.traineddata` | `36bdf9ac823f5911e624c30d0553e890b8abc7c31a65b3ef14da943658c40b79` |
| tessdata_best `jpn_vert.traineddata` | `1258be6eb2a9851f18043234ad18cca13ed32690bfff62b335c898bbea371548` |
| tessdata_best `eng.traineddata` | `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba` |

Another file (for example `yolox_s.onnx`) still runs: its measured sha256 is recorded and the result carries a warning that the skill has not seen that file before. `doctor` reports `MISSING` with the source for each missing file and `INCOMPLETE` for a Git LFS pointer.

## Commands

| Command | What it does |
|---|---|
| `faces <video\|image> [--every S \| --at t,... \| --scenes] [--min-score 0.6] [--models-dir DIR] [--model FILE] [-o det.json]` | Face boxes with five landmarks ([`detection-skill/detections@1`](docs/detections.md)) |
| `text <video\|image> [...] [--lang jpn+eng] [--vertical] [--psm N] [--level line\|word] [--tessdata-dir DIR]` | Text lines (or words) with their OCR text |
| `objects <video\|image> [...] [--classes person,...] [--min-score 0.3]` | Object boxes with COCO class names |
| `tracks <det.json> -o tracks.json [--min-iou 0.3] [--max-shift 1.0] [--max-gap S] [--pad 0.15] [--extend S] [--window S] [--min-hits 1]` | Boxes linked over time by IoU into track ids, with even-pixel redact rectangles per span (a heuristic, [docs/detections.md](docs/detections.md#tracks-a-heuristic)) |
| `check <det.json\|tracks.json>` | Validate a detections or tracks document |
| `contract --json` | Machine-readable contract in the OS registry's shape |
| `doctor --json [--models-dir DIR] [--tessdata-dir DIR] [--ffmpeg-skill DIR]` | Python, ffmpeg-skill, OpenCV, ONNX Runtime, the model files, Tesseract and its languages; `MISSING` entries carry install lines |

Common options: `--max-frames 2000` (refuses a bigger sampling plan), `--timeout 3600` (seconds for the engine; the worker process group is killed past it), `--ffmpeg-skill DIR`, `--overwrite`, `--json`. A video without a sampling option is sampled once a second. Every command prints one JSON document with `--json`. Exit codes: 0 ok, 2 invalid input or arguments, 3 engine / model file / tessdata / ffmpeg-skill missing, 4 the engine or frame extraction failed or timed out. Outputs never overwrite an existing file without `--overwrite`, and `-o` equal to an input is refused.

## How it works

- **Frames.** For a video: ffmpeg-skill `probe` (duration, size, rotation), `scenes` for `--scenes`, then `look --at T ... --no-timecode --width <source width> --json`, which writes PNG frames into a private temporary directory (deleted afterwards). An image is its own single frame.
- **Engines.** The engine runs in a worker subprocess (`python -m detection_skill.engines.worker`) in its own process group, killed on `--timeout`. Model files are found and hashed before any frame is made, so a missing model fails fast with exit 3.
- **Boxes.** Engine boxes are scaled from the extracted frame to the source size, clipped to the frame, and written with `box_norm`, the score and the provenance. The document is validated before it is returned or written.
- **Tracks.** Greedy IoU linking between consecutive samples (with a centre-distance fallback for motion faster than the sampling); redact rectangles are the padded union of a track's boxes, snapped to even pixels, extended by one sampling step on each side. Documented as a heuristic, not identity.

## Measured here

On the fixtures in `tests/` (2026-10-30, CPU, this repository's tests):

| What | Result |
|---|---|
| YuNet 2026may, 5 faces about 23 px to 276 px wide in one 1920x1080 frame | recall 5/5, precision 5/5 at IoU 0.5 (IoU 0.76-0.90 against hand-drawn boxes) |
| YuNet on a face moving 60 px/s, sampled every 0.2 s through ffmpeg-skill look | 20/20 frames with the face, 1 track |
| YOLOX-Nano: person, cat, cup in three photos | 3/3 found at IoU >= 0.5 (0.97, 0.89, 0.87) |
| Tesseract 5.3.4, rendered slide text (Noto Sans CJK, 44 px) | Japanese CER 0.019 (apt `tessdata`) / 0.057 (`tessdata_best`); English CER 0.000 with both; vertical Japanese (`jpn_vert`) CER 0.000 |

These are small synthetic fixtures that prove the plumbing and give a floor, not a benchmark. Real event footage (motion blur, small faces, low light, stylised slide fonts) will do worse; measure on your own material before relying on it. On these fixtures `tessdata_best` was not better than apt's `tessdata` for Japanese (it misread 第二章), so the ADR's follow-up (a Japanese CER comparison on real slides, and RapidOCR) stays open.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests inject a test-only engine (`tests/fake_engine.py`, never registered) and a fake ffmpeg-skill (`tests/fake_ffmpeg_skill`, the real tools' JSON shapes, PNG frames written with the standard library), so they run anywhere without engines, models or FFmpeg. They cover the schema and its validator, sampling, scaling and clipping, the worker and its timeout kill, tracks, the CLI, doctor and the contract. Tests that need more skip with the reason: frames through a real ffmpeg-skill checkout (`FFMPEG_SKILL_DIR` or `../ffmpeg-skill`) and FFmpeg; the OS registry conformance checks (`AI_VIDEO_PRODUCTION_OS_DIR` or `../AI-video-production-OS`); the real engines (`DETECTION_MODELS_DIR` with the YuNet / YOLOX files and the Python packages; `tesseract` with `jpn`, `jpn_vert`, `eng`, plus Pillow and a Japanese font, `DETECTION_TEST_FONT`, to render the OCR fixture). CER limits: `DETECTION_CER_JA` (default 0.10), `DETECTION_CER_EN` (default 0.05). CI's integration job installs everything (the two model files are fetched by the job and checked against the sha256 above) and fails if any test skips.

Fixture images: `tests/fixtures/astronaut.jpg` (NASA portrait of Eileen Collins, public domain), `chelsea.jpg` and `coffee.jpg` (CC0), as shipped in scikit-image's `data` module.

## License

The code is MIT (see `LICENSE`). Model files are not part of this repository and keep their own licences: YuNet (MIT), Tesseract and tessdata (Apache-2.0), YOLOX (Apache-2.0).
