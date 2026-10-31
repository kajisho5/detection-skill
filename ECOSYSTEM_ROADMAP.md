# Roadmap — detection-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Early (0.1.0). FACES (YuNet), TEXT (Tesseract, ja/en, vertical) and OBJECTS (YOLOX) write `detection-skill/detections@1` from ffmpeg-skill frames or an image; TRACKS links boxes into redact-ready rectangles (a heuristic). Contract and doctor in the OS shape. Measured on synthetic fixtures only; real event footage is not measured yet.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 顔検出 | partial | `detection faces` (0.1.0, YuNet); fixture recall 5/5, real footage unmeasured |
| 文字検出(OCR) | partial | `detection text` (0.1.0, Tesseract jpn/jpn_vert/eng; images accepted for slide-sync / stepcap); stepcap and slide-sync not wired yet |
| 物体検出 | partial | `detection objects` (0.1.0, YOLOX ONNX, COCO 80) |

## Wiring

Frames come from ffmpeg-skill `probe` / `scenes` / `look --no-timecode --width <source>` (dependency `>=2.0.0 <4.0.0`: 3.x accepted since [#7](https://github.com/kajisho5/detection-skill/pull/7), see ADR-002). `detection tracks` writes even-pixel rectangles with time spans in ffmpeg-skill `redact`'s `--x --y --width --height` convention; the agent-side plan (detection -> cut -> redact) is step 7. Not wired yet: video-editing reframing, slide-sync / stepcap OCR.

## Steps

- [x] **1. Engine ADR per task** ([docs/decisions.md](docs/decisions.md) ADR-001: YuNet / Tesseract / YOLOX) — commercial-use filter, CPU first. To verify: faces YuNet / MediaPipe (InsightFace weights non-commercial); OCR Tesseract / PaddleOCR / RapidOCR; objects YOLOX / RT-DETR (Ultralytics is AGPL)
  - Done when: ADR; ecosystem.json engines
- [x] **2. Skeleton + detection schema + contract/doctor** — frame sampling via ffmpeg-skill look (0.1.0: `schemas/detections.schema.json`, `detection contract` / `doctor` / `check`; publishes_contract, lifecycle_declared, dependency_version_ranges, no_unsafe_shell_out and doctor_status PASS in `tests/`)
  - Done when: OS static checks PASS
- [x] **3. FACES op** (0.1.0: YuNet 2026may; 5 faces of 23-276 px in one frame, recall 5/5 at IoU 0.5)
  - Done when: fixture recall
- [x] **4. TEXT (OCR) op, ja/en** (0.1.0: rendered slide text CER ja 0.019 apt tessdata / 0.057 tessdata_best, en 0.000, vertical ja 0.000; limits 0.10 / 0.05 in `tests/`)
  - Done when: CER threshold on rendered text
- [x] **5. OBJECTS op** (0.1.0: YOLOX-Nano; person / cat / cup fixtures 3/3 at IoU 0.5)
  - Done when: fixture
- [x] **6. Tracks** — link boxes over time into redact-ready rectangles (0.1.0: `detection tracks`, greedy IoU, documented heuristic; a real face moving 60 px/s sampled every 0.2 s keeps one id)
  - Done when: stable ids on a moving box
- [ ] **7. SKILL.md, CI; (agent side) blur faces -> redact** (skill side done in 0.1.0, checked in [#6](https://github.com/kajisho5/detection-skill/pull/6): `SKILL.md` documents every CLI command (`faces`, `text`, `objects`, `tracks`, `check`, `contract`, `doctor`) and the detection -> ffmpeg-skill `redact` recipe; `.github/workflows/tests.yml` (unit: Linux / macOS / Windows x Python 3.9 / 3.12 plus `detection contract --json`; integration: every real engine (YuNet, Tesseract jpn / jpn_vert / eng, YOLOX-Nano, models pinned by sha256) + ffmpeg-skill + the OS checkout, no test may skip, `doctor`); the OS checkout now names `kajisho5/Velisa`. The agent-side plan is pending in video-production-agent)
  - Done when: plan: detection -> ffmpeg-skill/redact

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
