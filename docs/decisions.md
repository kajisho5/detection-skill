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

## ADR-006 Personal and patient data in slides and frames (`detection pii`)

Roadmap step 11.

### Context

Medical and scientific talks show case slides; a patient id, a date of birth or a face in a clinical photo left on a slide must not be published. Finding them is measurement; deciding is a person's job, and removing them is ffmpeg-skill `redact`'s. The roadmap asks for patterns stated per locale and for every region to be a proposal, none removed silently.

### Decision

- **`detection pii <text-detections.json> [--faces face-detections.json] [--locale intl,...] -o pii.json`** writes `detection-skill/pii@1` (`schemas/pii.schema.json`): per OCR line that matches, one region with the category (`name`, `patient_id`, `date_of_birth`, `phone`, `address`; `face` for each face given with `--faces`), the rule ids, the reason, an `excerpt` (the keyword with the value masked: `DOB: ####-##-##`, the value itself is never copied), and a redact rectangle (the line's box, padded 0.1 per side, even pixels). `redact[]` joins regions of one category that overlap at consecutive sample times into spans (`start`, `end`; null for an image) in ffmpeg-skill `redact`'s `--x --y --width --height` convention.
- **Every region is a proposal.** `decision` is always `CONFIRM`; the validator rejects any other value and any region missing from `redact[]`. Nothing is removed by this skill; the agent asks a person, then runs `redact` on the confirmed spans.
- **Patterns per locale, international by default.** `intl` (default): English keywords (Patient ID / MRN / medical record / hospital number / case no., DOB / date of birth / born, Name / Patient, Mr / Mrs / Ms, Address, Tel / Phone) with ISO 8601, day-month-year, month-day-year and month-name dates, and E.164 phone numbers. Opt-in regional profiles: `en-US` (NANP phone formats, SSN, street addresses with ZIP), `en-GB` (NHS number, UK phones, postcodes), `ja-JP` (患者ID / 患者番号 / カルテ番号 / 診察券番号, 生年月日 with 和暦 or 西暦, 電話, 〒 and 住所, 氏名 / 患者名, 様 / 殿). Every rule is listed with what it means in `detection contract --json` (`pii.locales`).
- **Keyword first.** A date is a date of birth only after a birth keyword, a number a patient id only after an id keyword, a name only after a name keyword or an honorific; bare dates, years, percentages and statistics are left alone, so a results slide gives no region.
- **The whole line.** The region is the OCR line box, not a guessed sub-box of the matched characters: over-covering is the safe side for redaction. Line-level OCR (`detection text`'s default) is required for a keyword and its value to be seen together; word level gets a warning.

### Consequences

- A pattern check, not named-entity recognition: a bare name, a date of birth without its keyword, a handwritten or stylised label, or an OCR misread is not found. Every document says so (`method.note`), and an empty result carries a warning to review the slides anyway.
- Faces are all proposed when `--faces` is given; the skill cannot tell a clinical photo from the speaker, so the person confirming decides.
- Measured on rendered slides read by Tesseract (CI): a slide with `Patient ID: MRN 00482913` and `DOB: 1956-03-14` gives exactly those two regions (intl); a results slide with a publication date and statistics gives none; the Japanese slide (患者ID, 生年月日 in 和暦) gives both with `--locale ja-JP`. Synthetic slides; real case slides (photos of charts, scanned forms, small print) are unmeasured.

## ADR-007 Face attributes for frame choice (`eyes_open`, `detection rank`)

Roadmap step 12.

### Context

thumbnail-skill and take selection should not pick a frame where the face blinks or is blurred. Size, frontal angle and sharpness are already measured by `detection faces --attributes` (ADR-005); eyes open is not, and nothing ranks frames by them.

### Decision

- **`eyes_open`** joins the attributes of `faces --attributes` (`engines/faceattrs.py`): in each eye region (around YuNet's eye landmark, 0.56 x 0.28 eye distances, resized to 40x20 gray) r = mean|dI/dx| / (mean|dI/dx| + mean|dI/dy|), the share of edges across the eye: an open eye shows the iris and the corners, a closed one mostly the lid line along the eye. eyes_open = mean over both eyes of clamp((r - 0.25) / 0.2, 0, 1); null when the head is rolled more than 25 degrees or an eye region leaves the frame. A measured value from edge contrast, documented as a heuristic, not a blink classifier and not a model.
- **`detection rank <face-detections.json> [--top N] [--all-faces] -o rank.json`** writes `detection-skill/face-rank@1` (`schemas/face-rank.schema.json`): the frames in descending score, each face with its four parts in 0-1 and its score = 0.35 eyes open + 0.25 sharpness / (sharpness + 100) + 0.20 frontal + 0.20 min(1, size / 0.4). A frame scores by its best face, or with `--all-faces` by the mean of its faces (a group shot where one person blinks drops). A part not measured counts 0 and is listed in `missing`. Frames without a face come last with a null score. The validator checks the order, the ranks and that each score is the weighted sum of its parts.
- Blur is reported as `sharpness` (higher = sharper), the same measure ADR-005 uses for the camera choice.

### Consequences

- Heuristics: glasses, heavy make-up, side light and small faces (eye distance under about 12 px) disturb `eyes_open`; a squint or a look down can read as closed. The ranking suggests frames; a person or thumbnail-skill picks.
- Measured on the CI fixture (the astronaut portrait, and the same face with both eyes painted closed, a soft-edged patch of the cheek's colour with a lash line, placed with the landmarks; a 5 s video: open, closed, open blurred, open, closed; sampled every 0.5 s): every sharp open frame ranks above every other frame, every open frame's `eyes_open` (0.99-1.00) is above every closed frame's (0.30-0.35), and the blurred frames' sharpness is below every sharp frame's. Synthetic closed eyes, not real blinks; real footage is unmeasured.

## ADR-008 Optional second engines: PP-OCRv6 (text) and RF-DETR Small / Medium (objects)

Checked 2026-11-12 against the upstream pages listed under Sources. Revisits ADR-001's follow-ups (a second OCR backend; a transformer object detector once its weight licence is confirmed).

### Context

ADR-001 chose one engine per task and left Tesseract's Japanese screen-text accuracy open. Both new candidates run as ONNX files through ONNX Runtime, already the runtime of YOLOX and TransNetV2, so each can be an opt-in second engine without a new package. ADR-001's requirements still hold: commercial use of code and weights, offline, CPU first, frames from ffmpeg-skill.

### PP-OCRv6: what was verified

| Question | Answer | Where |
|---|---|---|
| Licence | Apache-2.0: `license: apache-2.0` on the model cards of `PP-OCRv6_medium_rec`, `PP-OCRv6_medium_det` and both `_onnx` repositories | the Hugging Face model cards |
| ONNX | Official exports: `PaddlePaddle/PP-OCRv6_medium_det_onnx` and `PaddlePaddle/PP-OCRv6_medium_rec_onnx` (the rec card links them as "ONNX Model"), each an `inference.onnx` (62,032,837 / 76,554,979 bytes) beside the Paddle `inference.json` / `inference.yml` it was converted from (paddle2onnx); each Paddle model card links its export as "ONNX Model". A `small` tier is published the same way (and a `tiny` one) | the repositories' file lists at revisions `6132380` (det) and `50c7eac` (rec) |
| Graph I/O | Read from the ONNX files themselves (the graph's input and output declarations at the end of each file, not the weights): det `x` float32 [N, 3, H, W] -> `fetch_name_0` float32 [N, 1, H, W], opset 14; rec `x` float32 [N, 3, 48, W] -> `fetch_name_0` float32 [N, T, 18710], opset 11. The Paddle programs end in `sigmoid` (`/DBHead/Head/`) and `softmax` axis 2 (`/MultiHead/CTCHead/`): the outputs are probabilities. Of the CTC + NRTR heads only CTC is in the inference graph | `inference.onnx`, `inference.json` |
| Languages | "medium/small tiers support 50 languages with a single unified model, including Simplified Chinese, Traditional Chinese, English, Japanese, and 46 Latin-script languages (tiny supports 49, excluding Japanese)". The rec cards say 50 languages too and their accuracy table has Printed EN and JP columns; the det cards say 48; the cards' `language` tags list only en and zh. The recognizer and its dictionary decide what can be read. The medium dictionary (`inference.yml`, `PostProcess.character_dict`) has 18708 entries, 86 hiragana and 94 katakana among them; 18708 + blank + space = 18710, the recognizer's output width | the PaddleOCR PP-OCRv6 page; `inference.yml` |
| Detection input | `DecodeImage` BGR; `DetResizeForTest` with the pipeline's `limit_side_len: 64`, `limit_type: min`, `max_side_limit: 4000` (each side then rounded to a multiple of 32); `NormalizeImage` scale 1/255, mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225), applied in the image's channel order (B, G, R); CHW | `inference.yml`; PaddleX `configs/pipelines/OCR.yaml`, `text_detection/processors.py` |
| Detection output | `DBPostProcess`: binary map at `thresh`, contours, each minimum-area rectangle scored by the mean probability inside it (`box_score_fast`), kept at `box_thresh`, grown by `unclip_ratio` (offset distance area x ratio / perimeter), shorter side >= 3 px before and >= 5 px after. The pipeline uses thresh 0.3, box_thresh 0.6, unclip_ratio 1.5 (the det module's own `inference.yml` has 0.2 / 0.45 / 1.4); max_candidates 3000 | PaddleX `OCR.yaml`, `text_detection/processors.py`, `inference.yml` |
| Recognition input | each box cut out by a perspective warp (`get_rotate_crop_image`, turned 90 degrees at height / width >= 1.5); `RecResizeImg` image_shape [3, 48, 320]: height 48, aspect kept, (x / 255 - 0.5) / 0.5, zero-padded on the right to max(320, 48 x w / h), at most 3200 | `inference.yml`; PaddleX `text_recognition/processors.py`, `crop_image_regions.py` |
| Recognition output | `CTCLabelDecode`: argmax per step, repeats collapsed, blank (index 0) dropped; classes = blank + `character_dict` + space; score = mean of the kept steps' probabilities | PaddleX `text_recognition/processors.py` |

### RF-DETR: what was verified

| Question | Answer | Where |
|---|---|---|
| Licence | The `rfdetr` package and "RF-DETR Nano, Small, Medium and Large detection" are Apache 2.0; Atto, Femto, Pico, XL and 2XL (`rfdetr[plus]`) are under the Roboflow Platform Model License (PML 1.0). `LICENSE` is Apache 2.0; PyPI `rfdetr` 1.12.0 (2026-11-06) declares Apache License 2.0 | README "License" at tag 1.12.0; `LICENSE`; PyPI |
| ONNX | No ONNX file of the official weights is published. `model.export()` (default `format="onnx"`, opset 17, batch 1, the model's fixed resolution) writes `{variant}.onnx`, e.g. `rfdetr-small.onnx`; it needs `rfdetr[onnx]` (PyTorch, `onnx`). Running the export needs only ONNX Runtime ("Advanced: Run ONNX Runtime directly"). So PyTorch is needed once, for the export, never at run time | `docs/exports/basics.md`, `docs/exports/onnx.md`, `src/rfdetr/detr.py` |
| Graph I/O | input `input` [1, 3, H, W]; outputs `dets` [1, Q, 4] (normalised cx, cy, w, h) and `labels` [1, Q, num_classes + 1] (raw logits), to be matched by name, not shape | `src/rfdetr/export/prepare.py`, `docs/exports/onnx.md` |
| Sizes | Small 512 x 512, Medium 576 x 576 (patch 16), 300 queries, num_select 300; COCO AP50:95 53.0 / 54.7 as published | `src/rfdetr/config.py`, README |
| Input | RGB, scaled to 0-1, resized to H x W (bilinear, half-pixel, no antialias), ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), NCHW float32 | `src/rfdetr/export/_runtime/preprocess.py` |
| Decoding | a sigmoid per class (multi-label, not softmax); the Q x C query/class pairs flattened and the best num_select kept (descending score, then ascending index), then the threshold; boxes times the image size, clipped; no NMS | `src/rfdetr/export/_runtime/decode.py`, `_topk.py` |
| Class map | the official COCO checkpoints keep COCO's sparse category ids: num_classes 90, 91 logit slots, slot k = category id k, no background slot to drop (`background_class_id=None`). `assets/coco_classes.py` names the 80 ids 1-90 with the same names as YOLOX's list | `docs/exports/onnx.md`, `src/rfdetr/assets/coco_classes.py`, `utilities/class_names.py` |

### Decision

- **Two registered, opt-in engines.** `text --engine ppocrv6` and `objects --engine rfdetr`. The defaults stay `tesseract` and `yolox`; installing a file never changes a default.
- **Runtime.** ONNX Runtime, numpy and OpenCV, all already in `[objects]`. No PaddleOCR, no polygon-clipping package (a rectangle's unclip is computed in closed form: the rectangle grown by the offset distance on each side, which is the minimum-area rectangle of PaddleOCR's rounded offset), no YAML package (a reader for the YAML subset of `inference.yml`, strict, checked to give PyYAML's result on the medium and small files), no PyTorch.
- **Model files are the user's.** PP-OCRv6: `PP-OCRv6_<tier>_det.onnx`, `PP-OCRv6_<tier>_rec.onnx`, `PP-OCRv6_<tier>_rec.yml` (the rec repository's `inference.yml`), medium then small; tiny is not looked for (no Japanese). RF-DETR: `rfdetr-small.onnx` or `rfdetr-medium.onnx` (or `--model`). The download commands (pinned to the revisions above) and the export commands (rfdetr 1.12.0) are printed by `doctor` (`ppocrv6_models.download`, `rfdetr_model.export`) and are the `detail` of the op's `MODEL_NOT_FOUND`. Both checks are optional: `notes` and `optional_ready` (`text_ppocrv6`, `objects_rfdetr`), never `problems`, `ready` or `ok`. Nothing is downloaded or exported by the skill.
- **Provenance.** The six PP-OCRv6 files (medium and small) are in the known-file table by sha256 (the `.onnx` from Hugging Face's LFS record, the `.yml` hashed here); every result lists each file with its role and the combined sha256, as for tessdata. An RF-DETR export has no fixed sha256: it runs with its measured sha256 and a warning. `model_license` is Apache-2.0 for `rfdetr-small.onnx` / `rfdetr-medium.onnx` and `unknown (not checked)`, with a warning naming the PML sizes, for any other file name.
- **Mapping onto `detections@1`, no schema change.** PP-OCRv6: one detection per line, `box` the axis-aligned box around the quadrilateral (over-covering a slanted line, the safe side for redaction), `label` the text, `score` the recognition score, `lang: multi`, `words` the whitespace-separated parts; `--lang`, `--vertical`, `--psm`, `--tessdata-dir` and `--level word` are refused with it. RF-DETR: `label` the COCO name, `score` the class sigmoid, `--classes` as for YOLOX; the 11 unnamed slots are dropped and a file with another slot count (a custom-trained model) is refused (`ENGINE_FAILED`).
- **Left out on purpose.** PaddleOCR's optional document orientation, unwarping and text-line orientation models (an upside-down line is misread); batching of recognizer crops (each line is run alone, so its zero padding differs from a batch's widest line).

### Consequences

- **Measured on the real files, on small fixtures only** (2026-11-12, the manual `heavy-tests.yml`, `tests/test_real_models.py`; the files fetched or exported with the commands `doctor` prints, PyTorch only in the export job). PP-OCRv6 medium on rendered 1280x720 slides (44 px): English and Japanese CER 0.000, 4/4 lines each, line-box IoU 0.72-0.95, about 2.2 s per slide with model load. RF-DETR Small (rfdetr 1.12.0 export): person / cat / cup at IoU 0.97 / 0.97 / 0.98, and 0.95 / 0.88 for the cat and cup pasted into a 1280x720 frame (the non-uniform resize maps back correctly), about 1.05 s per image. No pre- or post-processing defect showed. Still unmeasured: real footage and slides, vertical Japanese, a CER comparison with Tesseract on hard material. Unit tests keep covering the arithmetic with stub sessions and tiny synthetic ONNX models.
- PP-OCRv6 keeps a frame at full resolution unless its short side is under 64 px (a 1920 x 1080 frame enters the detector as 1920 x 1088).
- The contract is additive: `--engine` on `detection/text` and `detection/objects`, two optional model-weights dependencies, `engines.by_kind` and `engines.optional`; `contract_version` stays 1.0. The skill version becomes 0.2.0.

### Follow-ups

- Measure both on the real files: CER on rendered Japanese and English slides against Tesseract, recall on the object fixtures against YOLOX-Nano, seconds per frame. A default changes only with a new ADR.

### Sources

- [PaddlePaddle/PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) and [PaddlePaddle/PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) (model cards)
- [PaddlePaddle/PP-OCRv6_medium_rec_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b) and [PaddlePaddle/PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/tree/61323801669c338b7891481ec7bac61ce31b576a) (`inference.onnx`, `inference.json`, `inference.yml`)
- [PaddleOCR: PP-OCRv6 introduction](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) (languages)
- PaddleX: [`configs/pipelines/OCR.yaml`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/pipelines/OCR.yaml), [`text_detection/processors.py`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/text_detection/processors.py), [`text_recognition/processors.py`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/text_recognition/processors.py), [`crop_image_regions.py`](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/pipelines/components/common/crop_image_regions.py)
- [roboflow/rf-detr at 1.12.0](https://github.com/roboflow/rf-detr/tree/1.12.0): [README "License"](https://github.com/roboflow/rf-detr/blob/1.12.0/README.md#license), [`LICENSE`](https://github.com/roboflow/rf-detr/blob/1.12.0/LICENSE), [`docs/exports/onnx.md`](https://github.com/roboflow/rf-detr/blob/1.12.0/docs/exports/onnx.md), [`docs/exports/basics.md`](https://github.com/roboflow/rf-detr/blob/1.12.0/docs/exports/basics.md), [`src/rfdetr/config.py`](https://github.com/roboflow/rf-detr/blob/1.12.0/src/rfdetr/config.py), [`src/rfdetr/assets/coco_classes.py`](https://github.com/roboflow/rf-detr/blob/1.12.0/src/rfdetr/assets/coco_classes.py), [`src/rfdetr/export/_runtime/decode.py`](https://github.com/roboflow/rf-detr/blob/1.12.0/src/rfdetr/export/_runtime/decode.py), [`src/rfdetr/export/prepare.py`](https://github.com/roboflow/rf-detr/blob/1.12.0/src/rfdetr/export/prepare.py)
- [rfdetr 1.12.0 on PyPI](https://pypi.org/project/rfdetr/1.12.0/)

## ADR-009 The contract in the OS contract shape: `skill` as an object, Skill-only `dependencies`

2026-11-13.

### Context

Velisa's contract-shape check (`si contract-check`, Velisa ADR-018) validates every pinned contract against the OS schema, `registry/capability_contract.schema.json`. This contract broke it in two places:

- `skill` was the package name as a string; the schema wants an object (`{id, version}`).
- `dependencies[]` named every entry under `id` and mixed the one Skill (ffmpeg-skill) with Python packages, a binary and model files; the schema wants Skill dependencies, each with `skill_id` and `version_range`.

The schema stays as it is. The compliant contracts already show the convention: transcription-skill lists only Skills in `dependencies` and leaves its provider packages and programs to `doctor` and `engines`; nle-bridge-skill and publish-skill give `skill` as `{id, package, version}`.

### Decision

- `skill` is `{"id": "detection", "package": "detection-skill", "version": <version>}`. `skill_id` and `id` stay.
- `dependencies` lists ffmpeg-skill only. The entry gains `skill_id`; its `id`, `kind`, `version_range`, `tools` and `located_by` stay, so a reader of `dependencies[].id` keeps working.
- OpenCV, PySceneDetect, ONNX Runtime, Tesseract and the model files leave `dependencies`. `doctor` already reports each one with its version or file, licence and install line (the PP-OCRv6 `download` and RF-DETR `export` commands included), and `engines` names the engines.

### Consequences

- `detection contract --json` passes `si contract-check` against the OS schema with no baseline entry.
- No command, option, output document, tool, capability id or error changed. `contract_version` stays 1.0: by the OS versioning rules (Velisa docs/VERSIONING.md section 3) a bump is for capability shape, and nothing a dependent calls changed. The skill version becomes 0.2.3.
- An orchestrator that compares `dependencies` or `skill` with a pinned copy sees a change and has to re-pin. video-production-agent does not read either field (it reads `skill_id`), and neither is in its fatal drift keys.
