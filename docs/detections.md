# Detections (`detection-skill/detections@1`), tracks (`detection-skill/tracks@1`), shots (`detection-skill/shots@1`), reframe (`detection-skill/reframe@1`) and speakers (`detection-skill/speakers@1`)

JSON Schemas: `schemas/detections.schema.json`, `schemas/tracks.schema.json`, `schemas/shots.schema.json`, `schemas/reframe.schema.json`. The validators in `src/detection_skill/detections.py`, `tracks.py`, `shots.py` and `reframe.py` are the executable form; `detection check <file>` runs the right one by the document's `schema`.

## Detections

One document per run of `detection faces`, `detection text` or `detection objects`, for one input file.

| field | meaning |
|---|---|
| `schema` | `detection-skill/detections@1` |
| `kind` | `face`, `text` or `object` |
| `source` | `filename` (bare name), `fingerprint` (`sha256:<hex>` of the file bytes), `size_bytes`, `media_type` (`video` / `image`), `duration` (seconds from ffmpeg-skill `probe`; null for an image), `width`, `height` (display size, rotation applied), `fps` (video) |
| `frame_size` | `{width, height}`, equal to the source size: every box is in these pixels |
| `sampling` | `method` (`every` / `at` / `scenes` / `image`), `every` (seconds, for `every`), `times` (the frame times, in order; empty for an image), `scene_points` (for `scenes`: `[{scene, start, end, t}]`, `t` the scene's midpoint), `frames_tool` (`ffmpeg-skill/look`), `ffmpeg_skill_version`, `extracted_size` (size of the PNG frames `look` wrote) |
| `engine` | `id` (`yunet` / `tesseract` / `yolox`), `kind`, `version` (OpenCV / Tesseract / ONNX Runtime version), `runtime`, `model`, `model_sha256`, `model_license`, `engine_license`, `files` (`[{file, sha256, size_bytes, known, ...}]`, every model file used; `known` says whether the sha256 is in the skill's table), `execution_mode` (`local`), `engine_seconds`, `details` |
| `parameters` | `min_score`; text: `lang`, `vertical`, `psm`, `level`; objects: `classes` (null = all) |
| `frames` | `[{index, t, detections: [...]}]`, strictly increasing `t` (null for an image) |
| `counts` | `frames`, `frames_with_detections`, `detections` |
| `skill`, `skill_version`, `tool`, `created_at`, `processing_seconds`, `warnings` | provenance of the run |

A detection:

| field | meaning |
|---|---|
| `t` | the frame time in seconds on the source timeline (the frame's own `t`); null for an image |
| `kind` | as the document |
| `box` | `[x, y, w, h]` in source pixels, top-left origin, clipped to the frame, 2 decimals |
| `box_norm` | `box` divided by `frame_size` (x and w by width, y and h by height), 5 decimals |
| `score` | 0..1. Faces: YuNet's score. Objects: objectness x class probability. Text: mean Tesseract word confidence / 100 |
| `label` | face: **null** (a face box is never an identity). text: the OCR text of the line (or word). object: the COCO class name |
| `lang` | text only: the Tesseract languages used (`jpn+eng`, `jpn_vert+eng`, ...) |
| `words` | text only: number of words in the line |
| `landmarks` | face only: five points (right eye, left eye, nose tip, right and left mouth corner) in source pixels |
| `attributes` | face only, with `faces --attributes` (ADR-005): `size` (box height / frame height), `yaw_deg`, `roll_deg`, `frontal` (cos yaw; from the landmarks), `sharpness` (Laplacian variance of the face at 96x96), `mouth_contrast` (gray-level spread around the mouth corners / 128); each null when it could not be measured. Heuristics |
| `engine`, `model`, `model_sha256` | provenance on every box, equal to the document's `engine` |

For OCR, `model_sha256` is the sha256 over the sorted `file=sha256` lines of the traineddata files used; each file's own sha256 is in `engine.files`, with `variant: tessdata_best` when it matches the skill's table and `unknown (...)` otherwise.

Rules the validator checks: frames in time order inside the duration; boxes inside the frame with positive size; `box_norm = box / frame_size`; scores in [0, 1] and not below `parameters.min_score`; face labels null and no identity keys (`identity`, `name`, `person_id`, `embedding`, ...) on a face; non-empty labels on text and objects; per-box provenance equal to the document's engine; counts; no absolute path, command line or credential anywhere (OCR text in `label` is exempt from the path check: it is what the frame shows).

### How boxes are measured

- **Frames.** A video's frames come from ffmpeg-skill: `probe` (duration, size, rotation), `scenes` (for `--scenes`), and `look --at T ... --no-timecode --width <source width> --json`, which writes one PNG per time into a private temporary directory that is deleted afterwards. The skill never decodes video itself. If `look` returns frames of another size (an odd source height is rounded to even), boxes are scaled back to the source size and a warning says so.
- **Times.** `--every S` samples 0, S, 2S, ... up to one frame before the end. `--at` takes the given times (rounded to milliseconds; a time past the end is refused). `--scenes` takes each scene's midpoint, away from the cut.
- **Faces.** YuNet at the frame's own size, plus a second pass downscaled to 640 px on the long side when the frame is larger than 960 px (YuNet finds faces of about 10-300 px), merged by NMS (IoU 0.3).
- **Text.** One `tesseract` run per frame (`--oem 1`, TSV output), words grouped into lines by Tesseract's block/paragraph/line numbers; no space is kept between two Japanese characters. `--vertical` uses `jpn_vert` and page segmentation mode 5.
- **Objects.** YOLOX's ONNXRuntime recipe: letterbox to the model's input size (pad 114, BGR, 0-255), grid decode with strides 8/16/32, score = objectness x class probability, class-aware NMS (IoU 0.45).

## Tracks (a heuristic)

`detection tracks det.json -o tracks.json` links a video's boxes over time. It is a heuristic and is documented as one: a track id says "these boxes overlap from one sample to the next", not who or what it is. Two people who cross can swap ids; a face that leaves and comes back gets a new id. For a subject that moves across the frame, the union over the whole track is a wide band: use `--window` (for example 1-2 s) to get one tighter rectangle per span. Nothing was measured between samples.

- **Linking.** Frame by frame in time order, each box joins the live track whose last box overlaps it most (IoU >= `--min-iou`, default 0.3), greedily, one box per track per frame. When consecutive boxes do not overlap enough (motion faster than the sampling: a 46 px face moving 60 px/s sampled every 0.5 s), a box whose centre moved at most `--max-shift` (default 1.0) times the last box's size (the square root of its area), with an area within 4x, still links; overlap wins over distance. `--max-shift 0` links by IoU only. In a crowd the distance fallback can link neighbouring faces; sample more densely instead of raising it. A track is live while its last box is at most `--max-gap` seconds old (default 2.5 sampling steps, at least 0.5 s). Objects link only within one class. Unlinked boxes start new tracks.
- **Redact rectangles.** For each track (or each `--window` seconds of it), the union of its boxes, padded by `--pad` (default 0.15) of the union's width/height on each side, snapped outward to even pixels (ffmpeg-skill `redact` needs even width and height) and clipped to the frame, over `[first box - extend, last box + extend]` clipped to the media. `--extend` defaults to one sampling step, because the object may be on screen just before the first sample that saw it and just after the last. Feed `x`, `y`, `width`, `height` to `ffmpeg-skill redact` on the segment `[start, end]` (cut it out first; redact covers a whole clip).
- `--min-hits N` drops tracks with fewer than N boxes (default 1: for redaction, a single sighting still counts).

| field | meaning |
|---|---|
| `schema` | `detection-skill/tracks@1` |
| `kind`, `source`, `frame_size` | copied from the detections document |
| `detections` | the input: `file` (bare name), `fingerprint` (sha256 of the file bytes), `tool`, `engine`, `model`, `model_sha256`, `sampling_method`, `sampling_step` |
| `method` | `name: greedy-iou`, `heuristic: true`, `identity: false`, `note` |
| `parameters` | `min_iou`, `max_shift`, `max_gap`, `pad`, `extend`, `window`, `min_hits`, `same_label` |
| `tracks` | `[{id: track_001, label, start, end, hits, mean_score, max_score, boxes: [{t, box, score}], redact: [{start, end, x, y, width, height}]}]`; face tracks have `label: null`, text tracks the most frequent OCR text |
| `counts` | `tracks`, `boxes_linked`, `boxes_dropped` (in tracks under `--min-hits`), `redact_rects` |

## Shots

`detection shots video.mp4 -o shots.json` measures shot boundaries over **every frame** (ADR-003). It is the frame-accurate shot list; ffmpeg-skill `scenes` stays the quick cut list.

- **Frames.** ffmpeg-skill `probe` gives the exact frame rate (`r_frame_rate`) and the frame count (`nb_frames`, else duration x rate). Frame k is asked for with `look --at T --no-timecode --width 320`, T its time rounded down to the millisecond: `look` returns the first frame at or after T. About 0.07 s per frame; `--max-frames` (default 20000) refuses a longer video.
- **Default engine: PySceneDetect** (`pyscenedetect`). ContentDetector (`--threshold`, default 27) and AdaptiveDetector find hard cuts; a cut is reported on the first frame of the new shot. A blend test finds gradual transitions (dissolves, fades): a run of frames that are each the average of the frames 0.16 s either side, while those two differ. A detector cut inside such a run is the same transition and is absorbed. Boundaries closer than `--min-shot-frames` (default 15) keep the stronger one.
- **Optional engine: TransNetV2** (`--engine transnetv2`) with `transnetv2.onnx` in the models directory; `doctor` reports it `MISSING` (optional) otherwise and the default engine is unaffected.

| field | meaning |
|---|---|
| `schema` | `detection-skill/shots@1` |
| `source` | as in detections (`media_type: video`, `fps` the exact rate) |
| `frames` | `method: every_frame`, `frame_rate`, `frame_count`, `seek_rule`, `frames_tool` (`ffmpeg-skill/look`), `ffmpeg_skill_version`, `extracted_size` |
| `engine` | `id` (`pyscenedetect` / `transnetv2`), `kind: shot`, `version`, `runtime`, `model`, `model_sha256` (null for PySceneDetect: no model file), `model_license`, `engine_license`, `files`, `execution_mode`, `engine_seconds`, `details` |
| `parameters` | the engine's: `threshold`, `adaptive_threshold`, `min_content_val`, `min_shot_frames`, `blend_half_window`, `blend_far_min`, `blend_ratio_max`, `blend_width` (PySceneDetect); `threshold`, `input_size` (TransNetV2) |
| `boundaries` | `[{index, frame, t, kind, score, transition, engine, metric}]`: `frame` is the first frame of the next shot (for gradual, the middle of the transition), `t` = frame / rate, `kind` `cut` or `gradual`, `score` 0-1 (comparable within one engine), `transition` null for a cut and `{start_frame, end_frame, start, end, frames}` (the blended frames, inclusive) for gradual, `metric` the raw values (`content_val`, `adaptive_ratio`, `detectors`; `blend_ratio_min`, `blend_far_max`; `probability`) |
| `shots` | `[{id: shot_001, index, start_frame, end_frame, frames, start, end, duration, starts_with, score, engine}]`, contiguous, covering every frame; `end_frame` inclusive, `start` = start_frame / rate, `end` = (end_frame + 1) / rate (the duration for the last), `starts_with` `start` / `cut` / `gradual`, `score` of the starting boundary (null for the first) |
| `counts` | `frames`, `shots`, `boundaries`, `cuts`, `gradual` |

## Reframe (a heuristic)

`detection reframe faces-tracks.json --aspect 9:16 --shots shots.json -o reframe.json` gives a crop window per sample time (ADR-004). It chooses windows; it crops nothing (video-editing-skill and the engine apply them).

- **Window.** The largest even-pixel window of `--aspect` (`9:16`, `1:1`, `4:5`, any `W:H`) inside the frame, the same size throughout.
- **Subject.** Per shot (the whole video without `--shots`), the face track with the most presence in the rest of the shot; kept until its track ends, then the next one in view. `--objects object-tracks.json` adds object tracks (`--object-classes`, default `person`) as the fallback when no face is in view. Nobody in view: the window holds. Every change is in `switches` with its reason.
- **Smoothing.** Inside a shot the window centre follows the subject's box centre with a dead zone (`--dead-zone`, 0.1 window width) and a speed limit (`--max-speed`, 0.6 window widths per second): it never jumps inside a shot. At a shot boundary it is placed on the new subject directly. It is kept inside the frame.
- **One window per shot.** `shots[]` gives one fixed window per shot for an engine without a time-varying crop (ffmpeg-skill today), with the fraction of samples whose subject box it contains.

| field | meaning |
|---|---|
| `schema` | `detection-skill/reframe@1` |
| `source`, `frame_size` | copied from the face tracks |
| `aspect`, `window` | `{name, ratio}`, `{width, height}` (even pixels) |
| `inputs` | `faces`, `objects`, `shots`: `file` (bare name), `fingerprint` (sha256 of the file bytes), `tool`, and `kind` / `tracks` / `sampling_step` or `engine` / `shots` |
| `method` | `name: subject-path`, `heuristic: true`, `identity: false`, `note` |
| `parameters` | `dead_zone`, `max_speed` (fractions of the window width), `dead_zone_px`, `max_speed_px_per_s`, `object_classes` |
| `windows` | `[{t, shot, x, y, width, height, subject, subject_box, subject_inside}]`: `subject` is `face/track_nnn`, `object/track_nnn` or null; `subject_inside` whether its box lies inside the window |
| `switches` | `[{t, shot, from, to, reason}]` |
| `shots` | `[{index, start, end, subject, x, y, width, height, subject_inside_fraction}]` |
| `counts` | `windows`, `switches`, `shots`, `subject_outside` |

## Speakers (a heuristic)

`detection speakers --camera cam0-face-tracks.json --camera cam1-face-tracks.json [--offsets 0,S] [--turns turns.json] -o speakers.json` gives, per time range on camera 0's timeline, the speaking face and the camera that shows it best, as a switch list for ffmpeg-skill `multicam --switch` (ADR-005). It cuts nothing.

- **Input.** One face tracks document per camera from `faces --every 0.12 --attributes` -> `tracks`, in multicam's input order (camera 0 is the reference). `--offsets` is multicam's `offsets_seconds` (reference time = camera time + offset).
- **Mouth motion.** A face's activity is the spread of its `mouth_contrast` over `--window` seconds (0.6). Without `--turns`, the faces with activity >= `--min-activity` (0.02) and at least half the highest are speaking; the one with the best view quality (size, frontal angle, sharpness) is shown. Nobody speaking: the camera holds.
- **Turns.** With diarization-skill turns, each anonymous speaker is matched per camera to the face whose mouth moves most during their turns (`matches`), and each turn goes to the best camera showing that face, switching at the turn's start.
- **Shortest shot.** Ranges shorter than `--min-shot` (1.0 s) are folded into the one before.

| field | meaning |
|---|---|
| `schema` | `detection-skill/speakers@1` |
| `source` | camera 0's source |
| `cameras` | `[{index, file, fingerprint, source {filename, fingerprint, duration}, offset, tracks, sampling_step, attributes}]` |
| `turns` | `{file, fingerprint, tool, offset, speakers, turns}` or null |
| `method` | `name: mouth-motion` or `turns+mouth-motion`, `heuristic: true`, `identity: false`, `note` |
| `parameters` | `window`, `min_activity`, `min_shot`, `step` (the grid, the finest sampling step), `share` (0.5) |
| `matches` | with turns: `[{speaker, camera, track, score, activity_in_turns, activity_elsewhere}]` |
| `ranges` | `[{start, end, camera, track, speaker, quality, reason}]`, contiguous from 0 to camera 0's duration |
| `switch` | the ranges as `START-END:CAM,...` for `multicam --switch` |
| `counts` | `cameras`, `ranges`, `switches`, `matches` |

