---
name: detection-skill
description: Find faces, text (OCR, Japanese and English, including vertical Japanese) and objects in a video's frames or in an image, as measured boxes with time, pixel box, score and model hash, and link them over time into redact-ready rectangles. Use when asked to find (and then blur) faces, name badges, screens or other things in event footage, read text on slides, screenshots or signs, check where people or objects are in a shot, or plan a reframe or crop around them. Local and offline (YuNet, Tesseract, YOLOX on CPU); frames come from ffmpeg-skill. It never identifies anyone.
---

# detection-skill

Run `detection <command> ... --json` (installed by `pip install -e .`; engines by `pip install -e '.[faces]'`, `'.[objects]'`, and the `tesseract` binary for text). Every command prints one JSON document. `detection doctor --json` says which engines are ready and, if not, what to install and where to get the model files (it never downloads anything).

| Task | Command |
|---|---|
| Faces once a second | `detection faces talk.mp4 --every 1 -o faces.json --json` |
| Faces at chosen times / once per scene | `--at 12.5,40` / `--scenes` |
| Text on a slide or screenshot | `detection text slide.png -o text.json --json` (default `--lang jpn+eng`) |
| English only / vertical Japanese | `--lang eng` / `--vertical` |
| Sparse text in a video frame (signs, badges) | `detection text hall.mp4 --every 2 --psm 11 -o text.json` |
| People and laptops | `detection objects hall.mp4 --every 2 --classes person,laptop -o objects.json --json` |
| Boxes over time, rectangles to blur | `detection tracks faces.json -o tracks.json --json` |
| Check a file | `detection check faces.json --json` |
| What this skill declares / needs | `detection contract --json`, `detection doctor --json` |

Model files: `--models-dir` or `DETECTION_MODELS_DIR` (YuNet `face_detection_yunet_2026may.onnx`, a YOLOX `yolox_*.onnx`); Tesseract data: `--tessdata-dir` or `DETECTION_TESSDATA_DIR`, else tesseract's own. ffmpeg-skill: `--ffmpeg-skill`, `DETECTION_FFMPEG_SKILL_DIR`, or `~/.claude/skills/ffmpeg-skill`. An image input needs no ffmpeg-skill.

## When to use it

- **Blur faces or badges** in event footage: `faces` (or `text` / `objects`) -> `tracks` -> for each `tracks[].redact` rectangle, cut `[start, end]` and run ffmpeg-skill `redact --x --y --width --height` on that segment. Sample densely enough for motion (`--every 0.2` to `0.5` for people walking) and keep `--extend` so the blur starts before and ends after the samples that saw the face. For someone walking across the frame add `--window 1` to `tracks`, so each second gets its own rectangle instead of one wide band.
- **Slide or screen text** for slide-sync, stepcap, chapters or search: `text` on the slide image or on video frames at slide changes (`--scenes`, or times from slide-sync).
- **Where things are** for reframing or a crop: `objects --classes person` or `faces`, then the agent decides the crop.

## Report it honestly

- **A face box is not a person.** Say "a face was detected at 12.0 s", never who it is. `label` is always null for faces and the skill has no recognition. Track ids (`track_001`) are not identities either: they say boxes overlapped from one sample to the next; two people crossing can swap ids, and a person leaving and returning gets a new id.
- **Only the sampled frames were looked at.** Say how frames were sampled (`sampling.method`, `every`, the number of frames). Something on screen between samples can be missed; a face that is only visible for 0.3 s can fall between one-second samples. For redaction, say that the result needs a human check of the rendered video before it is published.
- **It is a detector with misses and false alarms.** Report counts and scores, not certainty. Small (under about 10 px), blurred, turned-away or partly hidden faces are often missed; the default `--min-score` (faces 0.6, objects 0.3, text 0.3) trades misses for false alarms. If you lowered or raised it, say so.
- **OCR text is as read, not corrected.** Quote `label` as it is; do not fix spelling silently. Mention the language setting and that Japanese and small or stylised text have more errors. If `engine.files[].variant` is not `tessdata_best`, say the data variant differs from the one the ADR recommends.
- **Missing engine = say so.** Exit 3 means an engine, model file, tessdata language or ffmpeg-skill is missing; relay `doctor`'s install line. Never report "no faces" when the run did not happen.
- **Provenance.** Each result names the engine, the model, and the model file's sha256 on every box; keep `detections.json` with any redaction made from it.

Exit codes: `0` ok, `2` invalid input or arguments, `3` an engine, model file, tessdata or ffmpeg-skill missing (see `doctor`), `4` the engine or frame extraction failed or timed out.

## Not this skill's job

Recognising or naming people, decoding video (ffmpeg-skill `look` / `scenes`), blurring (ffmpeg-skill `redact`), slide-change detection (slide-sync-skill), segmentation masks or pose, correcting OCR text, downloading model files (a one-time manual step, see README).
