# Detections (`detection-skill/detections@1`) and tracks (`detection-skill/tracks@1`)

JSON Schemas: `schemas/detections.schema.json`, `schemas/tracks.schema.json`. The validators in `src/detection_skill/detections.py` and `src/detection_skill/tracks.py` are the executable form; `detection check <file>` runs the right one by the document's `schema`.

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
