# Roadmap — detection-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold. Engines chosen (ADR-001: YuNet faces, Tesseract OCR, YOLOX objects; all commercial-use, offline, CPU); no code yet.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 顔検出 | missing |  |
| 文字検出(OCR) | missing | stepcap already runs OCR; slide-sync needs it -- one provider for all |
| 物体検出 | missing |  |

## Wiring

None. Boxes feed ffmpeg-skill redact / crop / fit and video-editing reframing; frames come from ffmpeg-skill look, sample points from scenes.

## Steps

- [x] **1. Engine ADR per task** ([docs/decisions.md](docs/decisions.md) ADR-001: YuNet / Tesseract / YOLOX) — commercial-use filter, CPU first. To verify: faces YuNet / MediaPipe (InsightFace weights non-commercial); OCR Tesseract / PaddleOCR / RapidOCR; objects YOLOX / RT-DETR (Ultralytics is AGPL)
  - Done when: ADR; ecosystem.json engines
- [ ] **2. Skeleton + detection schema + contract/doctor** — frame sampling via ffmpeg-skill look
  - Done when: OS static checks PASS
- [ ] **3. FACES op**
  - Done when: fixture recall
- [ ] **4. TEXT (OCR) op, ja/en**
  - Done when: CER threshold on rendered text
- [ ] **5. OBJECTS op**
  - Done when: fixture
- [ ] **6. Tracks** — link boxes over time into redact-ready rectangles
  - Done when: stable ids on a moving box
- [ ] **7. SKILL.md, CI; (agent side) blur faces -> redact**
  - Done when: plan: detection -> ffmpeg-skill/redact

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
