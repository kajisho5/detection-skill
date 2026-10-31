# Architecture Decision Records

## ADR-001 Engines, one per task: YuNet (faces), Tesseract (OCR), YOLOX (objects)

Checked 2026-10-30 against the upstream repositories listed under Sources.

### Requirements

Detection runs on event and conference footage, mostly to find what must be blurred: faces of attendees who did not consent, name badges, and screens. The engines have to meet these:

- **Commercial use.** The licence allows commercial use of both the code and the weights.
- **Offline, CPU first.** Weights are local files and a GPU is optional.
- **Japanese OCR.** OCR covers Japanese, including vertical text, as well as English.
- **One OCR provider.** stepcap and slide-sync use the same OCR provider. stepcap already runs Tesseract.

### Faces

| Candidate | Licence | Notes | Verdict |
|---|---|---|---|
| **YuNet (OpenCV Zoo `face_detection_yunet_2026may.onnx`)** | MIT. "All files in this directory are licensed under MIT License." | Runs through OpenCV's `FaceDetectorYN` on CPU. The 2026may model takes a dynamic input size. Detects faces of about 10 px to 300 px | **Chosen** |
| MediaPipe BlazeFace | Not verified for this ADR | Pulls in another runtime (mediapipe) | Not now |
| InsightFace (SCRFD/RetinaFace weights) | Non-commercial (roadmap note) | – | Rejected: licence |

### OCR

| Candidate | Licence | Notes | Verdict |
|---|---|---|---|
| **Tesseract 5 + `tessdata_best`** | Apache-2.0. "All data in the repository are licensed under the Apache-2.0 License." | Ships `jpn` and `jpn_vert`. It is a system binary, already used by stepcap | **Chosen.** One OCR provider for detection, slide-sync and stepcap |
| RapidOCR (PaddleOCR models in ONNX) | Code is Apache-2.0. The models come from PaddleOCR, which RapidOCR says is Apache-2.0; Baidu holds the copyright | Its default languages are Chinese and English. Japanese needs a model from its model list, which was not verified | **Second backend, later.** Accuracy on Japanese screen text may beat Tesseract; measure before switching |

### Objects

| Candidate | Licence | Notes | Verdict |
|---|---|---|---|
| **YOLOX (Megvii)** | Apache-2.0 | The repo has an ONNX export and an ONNXRuntime demo, and publishes COCO weights in its GitHub Releases | **Chosen** |
| RT-DETR / RT-DETRv2 | Code is Apache-2.0. The licence of the weights is not stated on the repo page | – | Not now. Revisit if the weight licence is confirmed |
| Ultralytics YOLO | AGPL-3.0 (roadmap note) | – | Rejected: licence |

### Decision

- **Engines and runtimes.**
  - Faces: YuNet through `opencv-python-headless`.
  - OCR: the `tesseract` binary with `jpn`, `jpn_vert` and `eng` from `tessdata_best`.
  - Objects: YOLOX through `onnxruntime`.
  - Each engine is optional. `doctor` reports a missing engine or model file with its install line, and an op whose engine is missing exits 3.
- **Model files.** Model files live in a directory the user points to. YuNet and YOLOX ONNX files are named with their sha256. Nothing is downloaded at run time.
- **Frames.** Frames come from ffmpeg-skill (`look` at sample points, or `scenes` for change points). The skill does not decode video itself.
- **Output.** Boxes are measured facts: the frame time, the box, the score, the engine and the model hash. Linking boxes into tracks (step 6) is a separate, documented heuristic. A face box is never an identity.

### Follow-ups

- **RapidOCR backend.** Add it after a Japanese CER comparison against Tesseract on rendered slides.
- **RT-DETR.** Reconsider it if its weight licence is confirmed.

### Sources

- [OpenCV Zoo: face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
- [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## ADR-002 Dependency floors clear every published advisory

### Context

A supply-chain audit (Trail of Bits supply-chain-risk-auditor, checked against OSV on 2026-10-30) found two floors that allowed vulnerable releases:

- `opencv-python-headless>=4.8` allowed 4.8.0.74, whose wheels bundle a libwebp with the heap overflow CVE-2023-4863 (GHSA-jh2j-j4j9-crg3, HIGH; PYSEC-2023-184). The faces and objects engines decode the frames and images this skill is given through OpenCV.
- `numpy` had no floor at all, so any release could satisfy it, including ones before 1.22.0 with known advisories (e.g. GHSA-fpfv-jqm9-f5jm on 1.21.x).

### Decision

- `opencv-python-headless>=4.8.1.78`, the first wheel with a fixed libwebp.
- `numpy>=1.22`. 1.22.0 has no published advisory.
- Both support Python 3.9, the oldest version this skill declares and tests, so no environment marker is needed.

### Consequences

Pip already installs newer releases than either floor; the change only stops an old environment from satisfying the extras. onnxruntime's floor (`>=1.16`) had no advisory in the audit and is unchanged.


## ADR-002 ffmpeg-skill 3.x accepted (`>=2.0.0 <4.0.0`)

Checked 2026-10-31 against ffmpeg-skill v3.0.0 (CHANGELOG "3.0.0", `docs/contract.md` "What 3.0 changed", `contract --json` `removed`).

### Context

ffmpeg-skill 3.0.0 removed Python 3.9 and FFmpeg 5.x, changed `cut.py`'s defaults and the meaning of its `precision` / `keyframe_snapped`, and added `probe`'s `timecode`. `contract_version` stays 2.0 and no tool, argument or JSON key was removed. This skill uses `probe` (duration, size, video stream), `scenes` and `look --at --no-timecode --width`; it never calls `cut`.

### Decision

- **Range `>=2.0.0 <4.0.0`.** Nothing this skill reads changed in 3.0.0, so 3.x is accepted unchanged, with no 2.x compatibility flags (there is no `cut` call to pass them to). `frames.SUPPORTED_MAJORS = (2, 3)` and `frames.VERSION_RANGE` are the one source for the contract, `doctor` and frame extraction.
- **Python 3.9 stays, with ffmpeg-skill 2.x.** ffmpeg-skill's scripts run under this skill's own interpreter (`sys.executable`), and 3.x requires Python 3.10. On Python 3.9 a 3.x checkout is refused before any frame is extracted (`FFMPEG_SKILL_NOT_FOUND`, exit 3, naming both fixes) and `doctor` reports it `INCOMPATIBLE`. `requires-python` stays `>=3.9`: images need no ffmpeg-skill at all, and ffmpeg-skill 2.x still runs on 3.9.
- **An FFmpeg older than 6.1** is refused by ffmpeg-skill 3.x itself (`missing_tool`, `ffmpeg_unsupported`); it surfaces here as `FRAME_EXTRACTION_FAILED` with ffmpeg-skill's message, and `doctor` as `INCOMPLETE` (ffmpeg-skill's own doctor says `ok: false`).
- **CI** runs the integration job (every real engine, no skips) against the tags `v2.24.1` and `v3.0.0`, so both ends of the range are exercised; the unit jobs keep Python 3.9 / 3.12 on the fake ffmpeg-skill.
