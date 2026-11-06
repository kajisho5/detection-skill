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

## ADR-003 Shot boundaries: PySceneDetect by default, TransNetV2 optional

Checked 2026-11-06 against the upstream repositories listed under Sources (roadmap step 8).

### Context

The agent's edit view (video-production-agent step 14) and qc-skill's cut checks need the measured shot list: every shot's first and last frame, whether it starts with a hard cut or a gradual transition (dissolve, fade), and how sure the detector was. ffmpeg-skill `scenes` (FFmpeg `scdet`) gives a quick list of cut times; it is not frame-indexed and does not see dissolves. The ADR-001 requirements still hold: commercial use of code and weights, offline, CPU first, and frames come from ffmpeg-skill.

### Candidates

| Candidate | Licence | Notes | Verdict |
|---|---|---|---|
| **PySceneDetect 0.6.x (`scenedetect`)** | BSD-3-Clause (`License-Expression: BSD-3-Clause` in the 0.7.2 wheel, `License: BSD-3-Clause` in 0.6.7.1) | ContentDetector (HSV + edge change between consecutive frames) and AdaptiveDetector (the same score against a rolling average) are frame accurate on hard cuts; no model file. Its detectors take frames as arrays, so the skill feeds them the frames ffmpeg-skill decoded and never uses its video backends | **Chosen as the default** |
| **TransNetV2** (Souček & Lokoč 2020, [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)) | Code: MIT (`LICENSE`: "MIT License, Copyright (c) 2020 Tomáš Souček"). Weights: shipped in the same repository (`inference/transnetv2-weights/`, Git LFS) under that same LICENSE; no separate weight licence or use restriction is stated in the README or the inference README | A network trained for shot transitions, including gradual ones. Upstream inference is TensorFlow 2.1 (a PyTorch port converts the same weights); here it runs as an ONNX export through ONNX Runtime, already the objects engine's runtime. Its training data (ClipShots, IACC.3) carry their own terms, which this ADR did not review; the weights' licence is what governs their use | **Optional engine.** No weights ship with the skill and none are downloaded |
| PySceneDetect 0.7.x | BSD-3-Clause | Requires Python 3.10 and depends on `opencv-python` (not `-headless`, which would install a second OpenCV next to the faces/objects one). It also bundles a TransNetV2 ONNX detector | Not now: the range stays `>=0.6.7 <0.7` |
| ffmpeg-skill `scenes` (scdet) | (ffmpeg-skill) | Cut times only, no frame index, no gradual transitions | Stays ffmpeg-skill's quick cut list; not replaced |

### Decision

- **`detection shots <video>`** writes `detection-skill/shots@1` (`schemas/shots.schema.json`, `docs/detections.md`): boundaries (`frame` = first frame of the next shot, `t`, `kind` cut / gradual, `score` 0-1, the transition span for gradual, raw `metric`) and contiguous shots that cover every frame (`start_frame`..`end_frame` inclusive, `start` / `end` seconds, `starts_with`, `score`, `engine`).
- **Frames.** Every frame, through ffmpeg-skill `look --at T --no-timecode --width 320`. `look` seeks with `-ss T` before the input, which returns the first frame at or after T, so frame k is asked for at its time rounded down to the millisecond (`frames.seek_time`; measured on 25 and 29.97 fps H.264 with FFmpeg 6.1: rounding to the nearest millisecond returned the next frame for 29.97 fps). The exact rate comes from probe's `r_frame_rate` (its `fps` is rounded). One `look` call per frame costs about 0.07 s, so `--max-frames` defaults to 20000 (about 11 minutes at 30 fps).
- **Default engine: PySceneDetect** ContentDetector (threshold 27) and AdaptiveDetector (ratio 3.0, min content 15), `min_scene_len` 15 frames; their cuts are merged (one frame apart = one cut).
- **Gradual transitions with the default engine: a blend test, added here** (PySceneDetect has no dissolve detector). A frame passes when the frames h either side (h = 0.16 s, 4 at 25 fps) differ (mean absolute BGR difference >= 20 on a 64 px wide copy) and the frame is within 12 % of that difference from their average. A cross-dissolve or fade is a linear blend; motion is not (the average of two displaced frames is a double image). A run of passing frames (at least 3) is a gradual transition over [first - h, last + h]; a detector cut inside it (AdaptiveDetector fires where a dissolve starts or ends) is absorbed. A heuristic, documented as one: a slow, even lighting change can pass, a dissolve between two moving shots can fail.
- **Score.** 0-1, comparable within one engine only: a cut's content score over twice the threshold (or the adaptive ratio over twice its threshold, whichever is higher); a gradual's 1 - its best blend ratio; TransNetV2's network probability. The raw values are in `metric`.
- **TransNetV2** (`--engine transnetv2`): `transnetv2.onnx` (or `transnetv2*.onnx`) in the models directory, its sha256 recorded like every model file. The upstream recipe (48x27 RGB, 100-frame windows with step 50 and 25 frames of padding, the middle 50 predictions, threshold 0.5); a run of 3 or more frames above the threshold is gradual. **`doctor` reports it `MISSING` with where to get it, under `notes` and `optional_ready`, never in `problems` or `ready`, so it never fails doctor or the default op.** Asked for without the file, `shots --engine transnetv2` exits 3 (`MODEL_NOT_FOUND`).

### Consequences

- On the CI fixture (six lavfi sources, three hard cuts and two cross-dissolves, 225 frames) PySceneDetect finds every hard cut within one frame (on its frame when measured on 2026-11-06) and the blend test finds both dissolves inside their spans. Real footage (flashes, fast pans, slow dissolves between moving shots) will do worse; TransNetV2 is the upgrade path for gradual transitions.
- TransNetV2 is not measured in this repository: CI has no weights. The engine is unit-tested on its windowing and its turn of probabilities into boundaries only.
- `scenedetect` is a new optional extra (`[shots]`, also in `[all]`); it adds click, platformdirs and tqdm, all small.

### Sources

- [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (the wheels' metadata for 0.6.7.1 and 0.7.2)
- [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2): `LICENSE`, `README.md`, `inference/README.md`
- Souček, Lokoč: TransNet V2: An effective deep network architecture for fast shot transition detection, [arXiv:2008.04838](https://arxiv.org/abs/2008.04838)

## ADR-004 Subject path for reframing (`detection reframe`)

Roadmap step 9. The roadmap names OpenChatCut as a reference for the idea only; no code from it is used.

### Context

Vertical and square versions (9:16, 1:1, 4:5) of a 16:9 talk need a crop window that follows the speaker. Choosing the window is measurement; applying it (尺・画角合わせ) belongs to video-editing-skill and the engine. ffmpeg-skill has no time-varying crop yet (a proposal in Velisa's ROADMAP), so until it does, the consumer takes one window per shot.

### Decision

- **`detection reframe <face-tracks.json> --aspect 9:16 [--shots shots.json] [--objects object-tracks.json] -o reframe.json`** writes `detection-skill/reframe@1` (`schemas/reframe.schema.json`): one window per sample time of the face detections, the subject id and whether its box is inside the window, every subject switch with its reason, and one fixed window per shot.
- **Window.** The largest even-pixel window of the aspect inside the frame, the same size all through (no zoom).
- **Subject.** Per shot, the face track with the most presence in the rest of the shot (sum of sqrt(box area) over its boxes); kept until its track ends, then the next one present. Object tracks (default class `person`) are used only when no face is in view and only when `--objects` is given. With no subject the window holds still. A subject is a box track (`face/track_001`), never a person.
- **Smoothing.** Inside a shot the window centre follows the subject's box centre (interpolated between samples) with a dead zone (0.1 window width) and a speed limit (0.6 window widths per second); at a shot boundary from `detection shots` the window is placed on the new subject directly. The window is kept inside the frame. The validator rejects any move inside a shot larger than the speed limit allows (plus 2 px of even-pixel rounding).
- **Report what happened.** `subject_inside` is measured per sample and counted (`counts.subject_outside`, with a warning to raise `--max-speed`); the per-shot window carries the fraction of samples whose subject box it contains.

### Consequences

- A heuristic, like `tracks`: two people crossing can swap the track the window follows, and a subject that is not detected (turned away, too small) is not followed. Without `--shots` the whole video is one shot and the subject changes only when its track ends.
- Measured on the CI fixture (a face walking across a 960x540 frame at 105 px/s, then a cut): the face is inside the window in every sampled frame for 9:16, 1:1 and 4:5, and no move inside a shot exceeds the speed limit.

## ADR-005 Active speaker for multicam switching (`detection speakers`) and face attributes

Roadmap step 10 (field review 2026-11-05, Velisa ROADMAP third round).

### Context

A panel or an interview shot with two or more cameras is cut between them by who is talking. ffmpeg-skill `multicam` aligns the cameras by audio and cuts from a switch list (`START-END:CAM`, reference timeline); its `--switch energy` picks the loudest camera, which is a proxy, not who speaks. Deciding who speaks and which camera shows them best is measurement, and it belongs here; cutting stays with ffmpeg-skill, diarization (anonymous speaker turns) with diarization-skill.

### Decision

- **`detection faces --attributes`** adds measured attributes to every face (`engines/faceattrs.py`, on the full-size frame inside the YuNet engine): `size` (box height / frame height), `yaw_deg`, `roll_deg` and `frontal` (= cos yaw) from the five landmarks only (yaw from the nose tip's offset from the eyes' midpoint along the eye line, in half eye distances; no 3-D model, no pitch), `sharpness` (variance of the Laplacian of the face resized to 96x96 gray) and `mouth_contrast` (gray-level standard deviation around the two mouth corners / 128). Off by default, so existing documents do not change. `detection tracks` keeps them per box. All heuristics; the validator checks each range.
- **`detection speakers --camera <cam0 face tracks> --camera <cam1 face tracks> [--offsets 0,S] [--turns turns.json] -o speakers.json`** writes `detection-skill/speakers@1` (`schemas/speakers.schema.json`): contiguous ranges on camera 0's timeline, each with the camera, the speaking face track in it, the diarization speaker (with turns), the mean view quality and the reason, and `switch`, the ranges in `multicam --switch` form. Camera order and `--offsets` are multicam's (reference time = camera time + offset, as `multicam --offsets-only` reports `offsets_seconds`).
- **Mouth motion.** A face's activity at t is the standard deviation of its `mouth_contrast` over the samples within `--window` (0.6 s, at least three samples): talking opens and closes the mouth, silence keeps it still. Faces must be sampled densely (`--every 0.1` to `0.2`); a coarser plan gets a warning.
- **Best camera.** View quality per box = 0.5 x min(1, size / 0.5) + 0.3 x frontal + 0.2 x sharpness / (sharpness + 100). Without turns, at each step the faces with activity >= `--min-activity` (0.02) and >= half the highest are speaking, and the one with the best quality wins, so a close-up beats the wide shot of the same speaker without knowing it is the same person (nothing is matched across cameras). Nobody speaking: the camera holds; before anyone speaks, camera 0.
- **With diarization turns** (`diarization-skill/turns@1`, `--turns-offset` when it was measured on another clock): each anonymous speaker is matched, per camera, to the face track whose mean activity during that speaker's turns exceeds its activity during the other speakers' turns the most (`matches`, with both numbers). Each turn goes to the camera whose matched track is in view for at least half the turn with the best quality; the range starts at the turn's start, so the switch list follows the turns exactly. A speaker matched to no face in view: the camera holds.
- **Shortest shot.** Ranges shorter than `--min-shot` (1.0 s) are folded into the previous range, then neighbours on one camera are joined.

### Consequences

- Heuristics: mouth motion is not speech (chewing, laughing, a hand at the mouth), a speaker turned away or too small is not seen, and mouth contrast depends on lighting and resolution; with turns the decision is only as good as the diarization. Neither a track id nor a speaker id is an identity, and no face is matched across cameras.
- Without turns a switch lands up to about half a window from the true change (activity is measured over a centred window).
- Measured on the CI fixture (two cameras, 640x360, 12 s: a wide shot of two faces and a close-up of one; alternating speakers every 3 s, mouths drawn opening and closing with lavfi drawbox; YuNet sampled every 0.12 s): with turns the switch list is exactly the turns; with mouth motion alone every switch is within 0.12 s of the truth, and the close-up is chosen for the speaker it shows. Synthetic, not real footage; real panels (small faces, beards, masks, side angles) are unmeasured. The turns in that test are written from the fixture's construction in diarization-skill's `turns@1` shape, not produced by diarization-skill.

