"""detection-skill tests.

Unit tests use a test-only engine (tests/fake_engine.py) injected as an object, and a fake ffmpeg-skill
(tests/fake_ffmpeg_skill: probe / look / scenes with the real tools' JSON shapes, frames written as PNG with the
standard library), so they need no engine, no model file and no ffmpeg.

Tests that need more skip with the reason, and run where it is available:
- ffmpeg on PATH + a real ffmpeg-skill checkout (FFMPEG_SKILL_DIR or ../ffmpeg-skill): frames through `look`/`scenes`.
- the AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS): registry conformance.
- real engines: faces need opencv + YuNet in DETECTION_MODELS_DIR; objects need onnxruntime + opencv + a yolox_*.onnx
  in DETECTION_MODELS_DIR; text needs tesseract with jpn/jpn_vert/eng (DETECTION_TESSDATA_DIR or tesseract's own),
  Pillow and a Japanese font to render the fixture; shots need scenedetect + opencv + ffmpeg + ffmpeg-skill (fixtures
  rendered with ffmpeg's lavfi sources); reframe on real footage needs all of faces and shots.

Fixture images (tests/fixtures): astronaut.jpg (NASA portrait of Eileen Collins, public domain), chelsea.jpg (CC0),
coffee.jpg (CC0), all as shipped in scikit-image's data module. Hand-annotated boxes are below (ASTRONAUT_FACE etc.).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_skill import __version__, cli  # noqa: E402
from detection_skill.contract import ERRORS, TOOLS, contract  # noqa: E402
from detection_skill.detections import DETECTIONS_SCHEMA, REQUIRED, validate_detections  # noqa: E402
from detection_skill.doctor import doctor  # noqa: E402
from detection_skill.engines import get_engine  # noqa: E402
from detection_skill.engines.base import EngineRequest  # noqa: E402
from detection_skill.engines.boxes import iou, nms, union  # noqa: E402
from detection_skill.engines.tesseract_engine import group, join_words, parse_tsv, resolve_langs  # noqa: E402
from detection_skill.errors import DetectionError  # noqa: E402
from detection_skill import frames  # noqa: E402
from detection_skill.frames import last_time, plan_at, plan_every  # noqa: E402
from detection_skill.metrics import cer, match_boxes  # noqa: E402
from detection_skill.models import KNOWN_MODELS, find_tesseract, inspect_models, tessdata_dir, tessdata_langs  # noqa: E402
from detection_skill.service import detect, run_engine, run_process_group  # noqa: E402
from detection_skill.tracks import TRACKS_SCHEMA, make_tracks, redact_rect, validate_tracks  # noqa: E402
from detection_skill.shots import SHOTS_SCHEMA, detect_shots, validate_shots  # noqa: E402
from detection_skill.reframe import REFRAME_SCHEMA, make_reframe, parse_aspect, validate_reframe, window_size  # noqa: E402
from detection_skill.pii import PII_SCHEMA, find, make_pii, parse_locales, validate_pii  # noqa: E402
from detection_skill.speakers import SPEAKERS_SCHEMA, make_speakers, switch_list, validate_speakers  # noqa: E402
from detection_skill.engines.faceattrs import pose  # noqa: E402
from detection_skill.engines.shotcore import (blend_runs, merge_boundaries, transnet_boundaries,  # noqa: E402
                                              transnet_windows)
from detection_skill.frames import seek_time  # noqa: E402
from fake_engine import MODEL_SHA, ScriptedEngine, ScriptedShotEngine, png_size, write_png  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
FAKE_FFMPEG_SKILL = str(ROOT / "tests" / "fake_ffmpeg_skill")
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _sibling(env: str, name: str):
    for cand in (os.environ.get(env), str(ROOT.parent / name)):
        if cand and os.path.isdir(cand):
            return Path(cand)
    return None


FFMPEG_SKILL_DIR = _sibling("FFMPEG_SKILL_DIR", "ffmpeg-skill")
if FFMPEG_SKILL_DIR and not (FFMPEG_SKILL_DIR / "scripts" / "look.py").is_file():
    FFMPEG_SKILL_DIR = None
OS_DIR = _sibling("AI_VIDEO_PRODUCTION_OS_DIR", "AI-video-production-OS")
MODELS_DIR = os.environ.get("DETECTION_MODELS_DIR")
HAVE_CV2 = importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None
HAVE_ORT = importlib.util.find_spec("onnxruntime") is not None
HAVE_PIL = importlib.util.find_spec("PIL") is not None
HAVE_SCENEDETECT = importlib.util.find_spec("scenedetect") is not None
REAL_VIDEO = HAVE_FFMPEG and FFMPEG_SKILL_DIR is not None
REAL_VIDEO_WHY = "needs ffmpeg on PATH and an ffmpeg-skill checkout (FFMPEG_SKILL_DIR or ../ffmpeg-skill)"

# hand-annotated boxes [x, y, w, h] in the fixture images (512x512 / 451x300 / 600x400)
ASTRONAUT_FACE = [176, 62, 92, 118]
ASTRONAUT_PERSON = [15, 15, 355, 497]
CHELSEA_CAT = [0, 0, 400, 300]
COFFEE_CUP = [172, 18, 238, 290]


def _models_file(pred):
    if not MODELS_DIR or not os.path.isdir(MODELS_DIR):
        return None
    for n in sorted(os.listdir(MODELS_DIR)):
        if pred(n):
            return n
    return None


YUNET_FILE = _models_file(lambda n: n.startswith("face_detection_yunet_") and n.endswith(".onnx"))
YOLOX_FILE = _models_file(lambda n: n.startswith("yolox_") and n.endswith(".onnx"))
TESSERACT = find_tesseract()
TESS_LANGS = tessdata_langs(tessdata_dir(TESSERACT)) if TESSERACT else []
HAVE_TESS = bool(TESSERACT) and {"jpn", "jpn_vert", "eng"} <= set(TESS_LANGS)
FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                   "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/Library/Fonts/Arial Unicode.ttf", "C:/Windows/Fonts/msgothic.ttc"]
JP_FONT = next((f for f in [os.environ.get("DETECTION_TEST_FONT")] + FONT_CANDIDATES if f and os.path.isfile(f)), None)


def run_cli(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, json.loads(buf.getvalue())


def fake_video(tmp, name="clip.mp4", **kw):
    """A 'video' for the fake ffmpeg-skill: a JSON file with the facts its probe/look/scenes report."""
    v = {"duration": 4.0, "width": 640, "height": 360, "fps": 25.0}
    v.update(kw)
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(v, fh)
    return p


def moving_box(t, w, h):
    """One face-like box moving right 20 px per second (in a 640-wide frame), plus a fixed second box."""
    s = w / 640.0
    return [{"box": [(100 + 20 * t) * s, 80 * s, 60 * s, 70 * s], "score": 0.9, "landmarks": [[(130 + 20 * t) * s, 110 * s]] * 5},
            {"box": [400 * s, 200 * s, 50 * s, 50 * s], "score": 0.7}]


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="detection-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)


# ---- detections schema ---------------------------------------------------------------------------
class DetectionsTests(TmpCase):
    def video_doc(self, **kw):
        eng = ScriptedEngine("face", moving_box)
        return detect("face", fake_video(self.tmp), every=1.0, engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL, **kw)

    def test_video_document_validates_and_carries_provenance(self):
        doc = self.video_doc()
        self.assertEqual(doc["schema"], DETECTIONS_SCHEMA)
        self.assertTrue(validate_detections(doc).ok, validate_detections(doc).errors)
        self.assertEqual(doc["sampling"]["times"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(doc["sampling"]["frames_tool"], "ffmpeg-skill/look")
        self.assertEqual(doc["frame_size"], {"width": 640, "height": 360})
        d = doc["frames"][1]["detections"][0]
        self.assertEqual((d["t"], d["kind"], d["label"], d["engine"], d["model_sha256"]), (1.0, "face", None, "scripted", MODEL_SHA))
        self.assertEqual(d["box"], [120.0, 80.0, 60.0, 70.0])
        self.assertEqual(d["box_norm"], [0.1875, 0.22222, 0.09375, 0.19444])
        self.assertEqual(doc["counts"], {"frames": 4, "frames_with_detections": 4, "detections": 8})
        self.assertEqual(doc["source"]["fingerprint"][:7], "sha256:")

    def test_min_score_filters_and_is_recorded(self):
        doc = self.video_doc(min_score=0.8)
        self.assertEqual(doc["counts"]["detections"], 4)
        self.assertEqual(doc["parameters"]["min_score"], 0.8)

    def test_json_schema_file_matches_validator(self):
        schema = json.loads((ROOT / "schemas" / "detections.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(REQUIRED))
        self.assertEqual(schema["properties"]["schema"]["const"], DETECTIONS_SCHEMA)
        det = schema["$defs"]["detection"]
        self.assertEqual(set(det["required"]), {"t", "kind", "box", "box_norm", "score", "label", "engine", "model", "model_sha256"})
        tracks = json.loads((ROOT / "schemas" / "tracks.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(tracks["properties"]["schema"]["const"], TRACKS_SCHEMA)
        from detection_skill import reframe as rf_mod, shots as shots_mod
        from detection_skill import pii as pii_mod, speakers as sp_mod
        for name, const, required in (("shots", SHOTS_SCHEMA, shots_mod.REQUIRED), ("reframe", REFRAME_SCHEMA, rf_mod.REQUIRED),
                                      ("speakers", SPEAKERS_SCHEMA, sp_mod.REQUIRED), ("pii", PII_SCHEMA, pii_mod.REQUIRED)):
            sch = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
            self.assertEqual(sch["properties"]["schema"]["const"], const)
            self.assertEqual(set(sch["required"]), set(required))

    def test_validator_rejects(self):
        base = self.video_doc()
        cases = {
            "face label": lambda d: d["frames"][0]["detections"][0].__setitem__("label", "Alice"),
            "identity key": lambda d: d["frames"][0]["detections"][0].__setitem__("person_id", "p1"),
            "outside": lambda d: d["frames"][0]["detections"][0].__setitem__("box", [600.0, 80.0, 60.0, 70.0]),
            "norm": lambda d: d["frames"][0]["detections"][0].__setitem__("box_norm", [0.5, 0.5, 0.1, 0.1]),
            "below min": lambda d: d["frames"][0]["detections"][0].__setitem__("score", 0.1),
            "order": lambda d: d["frames"][1].__setitem__("t", 0.0),
            "counts": lambda d: d["counts"].__setitem__("detections", 99),
            "abs path": lambda d: d["source"].__setitem__("filename", "/home/user/clip.mp4"),
            "credential": lambda d: d["warnings"].append("token hf_abcdefghijklmnopqrstuvwxyz"),
            "command key": lambda d: d.__setitem__("argv", ["x"]),
            "sha": lambda d: d["engine"].__setitem__("model_sha256", "nope"),
            "provenance": lambda d: d["frames"][0]["detections"][0].__setitem__("model_sha256", "0" * 64),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(base)
            mutate(d)
            self.assertFalse(validate_detections(d).ok, name)

    def test_ocr_text_that_looks_like_a_path_is_not_a_leak(self):
        eng = ScriptedEngine("text", lambda t, w, h: [{"box": [10, 10, 100, 20], "score": 0.9, "label": "/usr/bin/env python3",
                                                      "lang": "eng"}])
        img = os.path.join(self.tmp, "slide.png")
        write_png(img, 320, 200)
        doc = detect("text", img, engine=eng, lang="eng")
        self.assertEqual(doc["frames"][0]["detections"][0]["label"], "/usr/bin/env python3")


# ---- service: images, sampling, frames ----------------------------------------------------------
class ServiceTests(TmpCase):
    def test_image_is_one_frame_with_null_time(self):
        img = os.path.join(self.tmp, "slide.png")
        write_png(img, 800, 600)
        eng = ScriptedEngine("text", lambda t, w, h: [{"box": [10, 20, 300, 40], "score": 0.95, "label": "本日 の 発表", "lang": "jpn+eng"}])
        doc = detect("text", img, engine=eng)
        self.assertEqual(doc["source"]["media_type"], "image")
        self.assertIsNone(doc["source"]["duration"])
        self.assertEqual((doc["sampling"]["method"], doc["frames"][0]["t"]), ("image", None))
        self.assertEqual(doc["parameters"]["lang"], "jpn+eng")
        self.assertEqual(eng.requests[0].params["psm"], 3)
        self.assertTrue(validate_detections(doc).ok)
        with self.assertRaises(DetectionError) as cm:
            detect("text", img, every=1.0, engine=eng)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")

    def test_vertical_uses_jpn_vert_and_psm5(self):
        self.assertEqual(resolve_langs("jpn+eng", True), ["jpn_vert", "eng"])
        self.assertEqual(resolve_langs("eng", True), ["jpn_vert", "eng"])
        self.assertEqual(resolve_langs(None, False), ["jpn", "eng"])
        img = os.path.join(self.tmp, "v.png")
        write_png(img, 200, 600)
        eng = ScriptedEngine("text")
        doc = detect("text", img, engine=eng, vertical=True)
        self.assertEqual((eng.requests[0].params["langs"], eng.requests[0].params["psm"]), (["jpn_vert", "eng"], 5))
        self.assertEqual(doc["parameters"]["lang"], "jpn_vert+eng")
        with self.assertRaises(DetectionError):
            resolve_langs("jpn; rm -rf", False)

    def test_boxes_are_clipped_and_scaled_from_the_extracted_frame(self):
        # odd source height: look's scale=W:-2 gives 640x360 for a 640x361 source -> boxes scaled back
        video = fake_video(self.tmp, height=361)
        eng = ScriptedEngine("object", lambda t, w, h: [{"box": [-20, 300, 100, 100], "score": 0.9, "label": "person"}])
        doc = detect("object", video, at=[1.0], engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL)
        d = doc["frames"][0]["detections"][0]
        self.assertEqual(d["box"][0], 0.0)
        self.assertLessEqual(d["box"][1] + d["box"][3], 361)
        self.assertEqual(doc["sampling"]["extracted_size"], {"width": 640, "height": 360})
        self.assertTrue(any("scaled" in w for w in doc["warnings"]))

    def test_rotation_swaps_the_frame_size(self):
        video = fake_video(self.tmp, width=640, height=360, rotation=90)
        doc = detect("face", video, at=[0.5], engine=ScriptedEngine("face"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(doc["frame_size"], {"width": 360, "height": 640})

    def test_sampling_plans(self):
        self.assertEqual(plan_every(1.5, 4.0, 25.0), [0.0, 1.5, 3.0])
        self.assertEqual(last_time(4.0, 25.0), 3.96)
        self.assertEqual(plan_every(1.0, 3.0, 25.0), [0.0, 1.0, 2.0])
        self.assertEqual(plan_at([3.0, 1.0, 1.0, 4.0], 4.0, 25.0), [1.0, 3.0, 3.96])
        with self.assertRaises(DetectionError):
            plan_at([5.0], 4.0, 25.0)
        with self.assertRaises(DetectionError):
            plan_every(0, 4.0, 25.0)

    def test_scenes_sample_each_scene_midpoint(self):
        video = fake_video(self.tmp, duration=10.0, scenes=[[0, 4], [4, 10]])
        doc = detect("face", video, scenes=True, engine=ScriptedEngine("face"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(doc["sampling"]["times"], [2.0, 7.0])
        self.assertEqual([p["scene"] for p in doc["sampling"]["scene_points"]], [0, 1])

    def test_look_is_called_at_source_width_without_timecode(self):
        log = os.path.join(self.tmp, "calls.log")
        os.environ["FAKE_FFMPEG_SKILL_LOG"] = log
        self.addCleanup(os.environ.pop, "FAKE_FFMPEG_SKILL_LOG", None)
        video = fake_video(self.tmp, width=1920, height=1080, duration=100.0)
        detect("face", video, every=1.0, engine=ScriptedEngine("face"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        calls = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines()]
        looks = [c for c in calls if c[0].endswith("look.py")]
        self.assertEqual(len(looks), 3)  # 100 frames in batches of 40
        for c in looks:
            self.assertIn("--no-timecode", c)
            self.assertEqual(c[c.index("--width") + 1], "1920")
            self.assertIn("--json", c)

    def test_max_frames_and_argument_errors(self):
        video = fake_video(self.tmp, duration=100.0)
        eng = ScriptedEngine("face")
        with self.assertRaises(DetectionError) as cm:
            detect("face", video, every=0.1, max_frames=50, engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        for kw in ({"every": 1.0, "scenes": True}, {"min_score": 1.5}, {"classes": ["person"]}):
            with self.assertRaises(DetectionError):
                detect("face", video, engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL, **kw)
        with self.assertRaises(DetectionError) as cm:
            detect("face", os.path.join(self.tmp, "missing.mp4"), engine=eng)
        self.assertEqual(cm.exception.code, "INPUT_NOT_FOUND")
        with self.assertRaises(DetectionError) as cm:
            detect("object", video, classes=["person,unicorn"], engine=ScriptedEngine("object"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            detect("text", video, engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL)  # a face engine asked for text
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")

    def test_ffmpeg_skill_missing_or_incompatible_exits_3(self):
        video = fake_video(self.tmp)
        old = {k: os.environ.pop(k, None) for k in ("DETECTION_FFMPEG_SKILL_DIR", "FFMPEG_SKILL_DIR")}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) for k, v in old.items() if v is not None])
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        home = os.environ.get("HOME")
        os.environ["HOME"] = self.tmp
        self.addCleanup(lambda: os.environ.__setitem__("HOME", home) if home else os.environ.pop("HOME", None))
        with self.assertRaises(DetectionError) as cm:
            detect("face", video, engine=ScriptedEngine("face"), ffmpeg_skill=os.path.join(self.tmp, "nope"))
        self.assertEqual((cm.exception.code, ERRORS[cm.exception.code]["exit_code"]), ("FFMPEG_SKILL_NOT_FOUND", 3))
        for major in ("1", "4"):
            other = os.path.join(self.tmp, f"ffmpeg-skill-{major}")
            shutil.copytree(FAKE_FFMPEG_SKILL, other)
            Path(other, "package.json").write_text(f'{{"version": "{major}.0.0"}}', encoding="utf-8")
            with self.assertRaises(DetectionError) as cm:
                detect("face", video, engine=ScriptedEngine("face"), ffmpeg_skill=other)
            self.assertEqual(cm.exception.code, "FFMPEG_SKILL_NOT_FOUND")
            self.assertIn(">=2.0.0 <4.0.0", cm.exception.message)
            self.assertEqual(doctor(ffmpeg_skill=other)["checks"]["ffmpeg_skill"]["status"], "INCOMPATIBLE")

    def test_ffmpeg_skill_3x_is_accepted_and_needs_python_3_10(self):
        video = fake_video(self.tmp)
        v3 = os.path.join(self.tmp, "ffmpeg-skill-3")
        shutil.copytree(FAKE_FFMPEG_SKILL, v3)
        Path(v3, "package.json").write_text('{"version": "3.0.0"}', encoding="utf-8")
        skill = frames.locate(v3)
        self.assertIsNone(frames.python_floor(skill, (3, 10)))
        self.assertIsNone(frames.python_floor(skill, (3, 12, 1)))
        self.assertEqual(frames.python_floor(skill, (3, 9)), "3.10")
        self.assertIsNone(frames.python_floor(frames.locate(FAKE_FFMPEG_SKILL), (3, 9)))  # 2.x keeps Python 3.9
        if sys.version_info >= (3, 10):
            doc = detect("face", video, engine=ScriptedEngine("face"), ffmpeg_skill=v3)
            self.assertEqual(doc["sampling"]["ffmpeg_skill_version"], "3.0.0")
            self.assertTrue(validate_detections(doc).ok)
            self.assertEqual(doctor(ffmpeg_skill=v3)["checks"]["ffmpeg_skill"]["status"], "OK")
        else:
            with self.assertRaises(DetectionError) as cm:
                detect("face", video, engine=ScriptedEngine("face"), ffmpeg_skill=v3)
            self.assertEqual(cm.exception.code, "FFMPEG_SKILL_NOT_FOUND")
            self.assertIn("Python 3.10", cm.exception.message)
            check = doctor(ffmpeg_skill=v3)["checks"]["ffmpeg_skill"]
            self.assertEqual(check["status"], "INCOMPATIBLE")
            self.assertIn("ffmpeg-skill 2.x", check["install"])

    def test_engine_timeout_and_failure(self):
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        with self.assertRaises(DetectionError) as cm:
            detect("face", img, engine=ScriptedEngine("face", delay=2.0), timeout=0.2)
        self.assertEqual(cm.exception.code, "ENGINE_TIMEOUT")
        with self.assertRaises(DetectionError) as cm:
            detect("face", img, engine=ScriptedEngine("face", fail=RuntimeError("boom")))
        self.assertEqual(cm.exception.code, "ENGINE_FAILED")

    def test_model_warnings_reach_the_document(self):
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        doc = detect("face", img, engine=ScriptedEngine("face", warnings=["model x: sha256 not known"]))
        self.assertIn("model x: sha256 not known", doc["warnings"])


# ---- worker subprocess ----------------------------------------------------------------------------
class WorkerTests(TmpCase):
    def test_process_group_is_killed_on_timeout(self):
        t0 = time.time()
        code, _ = run_process_group([sys.executable, "-c", "import time; time.sleep(30)"], 1.0)
        self.assertEqual(code, 124)
        self.assertLess(time.time() - t0, 15)

    def test_worker_reports_structured_errors(self):
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        req = EngineRequest(frames=[{"index": 0, "t": None, "path": img}], model_path=os.path.join(self.tmp, "missing.onnx"))
        with self.assertRaises(DetectionError) as cm:
            run_engine(get_engine("yunet"), req, 120, self.tmp)
        self.assertIn(cm.exception.code, ("ENGINE_UNAVAILABLE", "ENGINE_FAILED"))
        with self.assertRaises(DetectionError) as cm:
            get_engine("nope")
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")

    def test_real_engines_refuse_without_model_files(self):
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        empty = os.path.join(self.tmp, "models")
        os.mkdir(empty)
        for kind, cmd in (("face", "faces"), ("object", "objects")):
            rc, doc = run_cli(cmd, img, "--models-dir", empty, "--json")
            self.assertEqual(rc, 3, doc)
            self.assertIn(doc["error"]["code"], ("MODEL_NOT_FOUND", "ENGINE_UNAVAILABLE"))
        lfs = os.path.join(empty, "face_detection_yunet_2026may.onnx")
        Path(lfs).write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n", encoding="utf-8")
        self.assertEqual(inspect_models(empty)["face"]["status"], "INCOMPLETE")


# ---- tracks ---------------------------------------------------------------------------------------
class TracksTests(TmpCase):
    def det_doc(self, boxes=moving_box, kind="face", every=0.5, duration=4.0):
        return detect(kind, fake_video(self.tmp, duration=duration), every=every, engine=ScriptedEngine(kind, boxes),
                      ffmpeg_skill=FAKE_FFMPEG_SKILL)

    def test_moving_box_keeps_one_stable_id(self):
        doc = self.det_doc()
        tr = make_tracks(doc)
        self.assertTrue(validate_tracks(tr).ok, validate_tracks(tr).errors)
        self.assertEqual(tr["counts"]["tracks"], 2)
        moving = tr["tracks"][0]
        self.assertEqual((moving["id"], moving["hits"], moving["start"], moving["end"]), ("track_001", 8, 0.0, 3.5))
        self.assertIsNone(moving["label"])
        r = moving["redact"][0]
        self.assertEqual((r["start"], r["end"]), (0.0, 4.0))  # extended by one 0.5 s step, clipped to the duration
        u = union([b["box"] for b in moving["boxes"]])
        self.assertLessEqual(r["x"], u[0])
        self.assertGreaterEqual(r["x"] + r["width"], u[0] + u[2])
        self.assertEqual((r["width"] % 2, r["height"] % 2, r["x"] % 2, r["y"] % 2), (0, 0, 0, 0))
        self.assertTrue(tr["method"]["heuristic"])
        self.assertFalse(tr["method"]["identity"])

    def test_gap_and_label_split_tracks(self):
        def flicker(t, w, h):
            return [] if 1.0 <= t < 3.0 else [{"box": [100, 100, 50, 50], "score": 0.9, "label": "cup" if t < 2 else "person"}]
        doc = self.det_doc(flicker, kind="object")
        tr = make_tracks(doc)
        self.assertEqual([(t["label"], t["hits"]) for t in tr["tracks"]], [("cup", 2), ("person", 2)])
        tr2 = make_tracks(doc, max_gap=5.0)
        self.assertEqual(tr2["counts"]["tracks"], 2)  # labels differ: never linked
        same = self.det_doc(lambda t, w, h: [] if 1.0 <= t < 3.0 else [{"box": [100, 100, 50, 50], "score": 0.9}])
        self.assertEqual(make_tracks(same)["counts"]["tracks"], 2)
        self.assertEqual(make_tracks(same, max_gap=5.0)["counts"]["tracks"], 1)

    def test_window_min_hits_and_padding(self):
        doc = self.det_doc()
        tr = make_tracks(doc, window=1.0, min_hits=3, pad=0.0, extend=0.0)
        self.assertEqual(tr["counts"]["tracks"], 2)
        self.assertEqual(len(tr["tracks"][0]["redact"]), 4)
        self.assertEqual(tr["tracks"][0]["redact"][0]["start"], 0.0)
        self.assertEqual(tr["tracks"][0]["redact"][0]["end"], 0.5)
        self.assertEqual(make_tracks(doc, min_hits=99)["counts"]["tracks"], 0)
        self.assertEqual(redact_rect([[0, 0, 11, 11]], 0.0, 101, 101), {"x": 0, "y": 0, "width": 12, "height": 12})
        self.assertEqual(redact_rect([[95, 95, 6, 6]], 0.5, 101, 101), {"x": 92, "y": 92, "width": 8, "height": 8})

    def test_fast_motion_links_by_centre_distance_unless_disabled(self):
        def fast(t, w, h):
            return [{"box": [50 + 60 * t, 100, 46, 56], "score": 0.9}]  # 30 px per 0.5 s step: IoU ~0.2 between samples
        doc = self.det_doc(fast)
        self.assertEqual(make_tracks(doc)["counts"]["tracks"], 1)
        self.assertEqual(make_tracks(doc, max_shift=0)["counts"]["tracks"], 8)
        two = self.det_doc(lambda t, w, h: [{"box": [50 + 60 * t, 100, 46, 56], "score": 0.9},
                                            {"box": [400, 250, 46, 56], "score": 0.8}])
        self.assertEqual([t["hits"] for t in make_tracks(two)["tracks"]], [8, 8])

    def test_crossing_boxes_follow_overlap_not_identity(self):
        def crossing(t, w, h):
            return [{"box": [100 + 40 * t, 100, 60, 60], "score": 0.9}, {"box": [400 - 40 * t, 100, 60, 60], "score": 0.9}]
        tr = make_tracks(self.det_doc(crossing))
        self.assertTrue(validate_tracks(tr).ok)
        self.assertIn("not an identity", tr["method"]["note"])

    def test_refuses_images_and_invalid_documents(self):
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        with self.assertRaises(DetectionError) as cm:
            make_tracks(detect("face", img, engine=ScriptedEngine("face")))
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            make_tracks({"schema": "x"})
        self.assertEqual(cm.exception.code, "INVALID_DETECTIONS")

    def test_validator_rejects_odd_rects_and_face_labels(self):
        tr = make_tracks(self.det_doc())
        for mutate in (lambda d: d["tracks"][0]["redact"][0].__setitem__("width", 13),
                       lambda d: d["tracks"][0].__setitem__("label", "Alice"),
                       lambda d: d["method"].__setitem__("identity", True),
                       lambda d: d["tracks"][1].__setitem__("id", "track_001")):
            d = copy.deepcopy(tr)
            mutate(d)
            self.assertFalse(validate_tracks(d).ok)


# ---- shots ----------------------------------------------------------------------------------------
def scripted_cut_and_dissolve(n):
    return [{"frame": 30, "kind": "cut", "score": 0.9, "metric": {"content_val": 80.0}},
            {"frame": 60, "kind": "gradual", "score": 0.95, "start_frame": 55, "end_frame": 65, "metric": {}}]


class ShotsTests(TmpCase):
    def shots_doc(self, boundaries=scripted_cut_and_dissolve, **kw):
        video = fake_video(self.tmp, duration=4.0, fps=25.0)
        return detect_shots(video, engine=ScriptedShotEngine(boundaries), ffmpeg_skill=FAKE_FFMPEG_SKILL, **kw)

    def test_seek_time_lands_on_the_frame(self):
        self.assertEqual((seek_time(0, 25.0), seek_time(1, 25.0), seek_time(2, 30000 / 1001)), (0.0, 0.04, 0.066))
        for rate in (25.0, 30000 / 1001, 24000 / 1001, 50.0, 60000 / 1001):
            for k in range(1, 2000):
                t = seek_time(k, rate)
                self.assertLessEqual(t, k / rate + 1e-9)       # not past frame k ...
                self.assertGreater(t, (k - 1) / rate + 1e-9)    # ... and past frame k - 1: look returns frame k

    def test_every_frame_is_asked_for_at_the_reduced_width(self):
        log = os.path.join(self.tmp, "calls.log")
        os.environ["FAKE_FFMPEG_SKILL_LOG"] = log
        self.addCleanup(os.environ.pop, "FAKE_FFMPEG_SKILL_LOG", None)
        doc = self.shots_doc()
        looks = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines() if "look.py" in line]
        asked = [float(c[i + 1]) for c in looks for i, a in enumerate(c) if a == "--at"]
        self.assertEqual(asked, [seek_time(k, 25.0) for k in range(100)])
        self.assertTrue(all(c[c.index("--width") + 1] == "320" and "--no-timecode" in c for c in looks))
        self.assertEqual(doc["frames"]["frame_count"], 100)
        self.assertEqual(doc["frames"]["extracted_size"], {"width": 320, "height": 180})

    def test_document_shots_cover_every_frame(self):
        doc = self.shots_doc()
        rep = validate_shots(doc)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(doc["schema"], SHOTS_SCHEMA)
        self.assertEqual(doc["counts"], {"frames": 100, "shots": 3, "boundaries": 2, "cuts": 1, "gradual": 1})
        self.assertEqual([(s["start_frame"], s["end_frame"], s["starts_with"]) for s in doc["shots"]],
                         [(0, 29, "start"), (30, 59, "cut"), (60, 99, "gradual")])
        self.assertEqual([(s["start"], s["end"]) for s in doc["shots"]], [(0.0, 1.2), (1.2, 2.4), (2.4, 4.0)])
        g = doc["boundaries"][1]
        self.assertEqual((g["t"], g["transition"]["start_frame"], g["transition"]["end_frame"], g["transition"]["frames"]),
                         (2.4, 55, 65, 11))
        self.assertIsNone(doc["boundaries"][0]["transition"])
        self.assertTrue(all(s["engine"] == "scripted-shots" for s in doc["shots"]))
        self.assertIsNone(doc["engine"]["model_sha256"])

    def test_no_boundary_is_one_shot(self):
        doc = self.shots_doc(lambda n: [])
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in doc["shots"]], [(0, 99)])
        self.assertTrue(validate_shots(doc).ok)

    def test_validator_rejects(self):
        base = self.shots_doc()
        cases = {
            "gap": lambda d: d["shots"][1].__setitem__("start_frame", 31),
            "count": lambda d: d["counts"].__setitem__("cuts", 2),
            "kind": lambda d: d["boundaries"][0].__setitem__("kind", "wipe"),
            "score": lambda d: d["boundaries"][0].__setitem__("score", 1.5),
            "first score": lambda d: d["shots"][0].__setitem__("score", 0.5),
            "engine": lambda d: d["shots"][2].__setitem__("engine", "other"),
            "cut transition": lambda d: d["boundaries"][0].__setitem__("transition", {"start_frame": 1, "end_frame": 2}),
            "t": lambda d: d["boundaries"][0].__setitem__("t", 9.0),
            "end": lambda d: d["shots"][2].__setitem__("end_frame", 98),
            "abs path": lambda d: d["source"].__setitem__("filename", "/home/user/clip.mp4"),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(base)
            mutate(d)
            self.assertFalse(validate_shots(d).ok, name)

    def test_engine_frame_count_and_argument_errors(self):
        video = fake_video(self.tmp, duration=4.0)
        with self.assertRaises(DetectionError) as cm:
            detect_shots(video, engine=ScriptedShotEngine(frames_seen=99), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(cm.exception.code, "INVALID_RESULT")
        with self.assertRaises(DetectionError) as cm:
            detect_shots(video, engine=ScriptedShotEngine(), max_frames=50, ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        img = os.path.join(self.tmp, "a.png")
        write_png(img, 64, 64)
        with self.assertRaises(DetectionError) as cm:
            detect_shots(img, engine=ScriptedShotEngine())
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            detect_shots(video, engine=ScriptedEngine("face"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            detect_shots(video, engine_id="scdet")
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")

    def test_transnetv2_without_its_model_exits_3_and_doctor_says_missing_optional(self):
        video = fake_video(self.tmp)
        empty = os.path.join(self.tmp, "models")
        os.mkdir(empty)
        rc, doc = run_cli("shots", video, "--engine", "transnetv2", "--models-dir", empty, "--ffmpeg-skill", FAKE_FFMPEG_SKILL, "--json")
        self.assertEqual(rc, 3, doc)
        self.assertIn(doc["error"]["code"], ("MODEL_NOT_FOUND", "ENGINE_UNAVAILABLE"))
        d = doctor(models_dir=empty, ffmpeg_skill=FAKE_FFMPEG_SKILL)
        tn = d["checks"]["transnetv2_model"]
        self.assertEqual((tn["status"], tn["optional"]), ("MISSING", True))
        self.assertIn("TransNetV2", tn["install"])
        self.assertFalse(any(p.startswith("transnetv2_model") for p in d["problems"]))
        self.assertIn("transnetv2_model: MISSING (optional)", d["notes"])
        self.assertNotIn("shots_transnetv2", d["ready"])
        self.assertFalse(d["optional_ready"]["shots_transnetv2"])
        have = all(importlib.util.find_spec(m) is not None for m in ("scenedetect", "cv2", "numpy"))
        self.assertEqual(d["ready"]["shots"], have and d["checks"]["scenedetect"]["status"] == "OK")

    def test_blend_runs_and_merge(self):
        passing = [(f, 40.0, 0.05) for f in range(73, 88)] + [(100, 30.0, 0.1)]  # one long run, one lone frame
        g = blend_runs(passing, 4, min_core=3, frame_count=225)
        self.assertEqual([(b["frame"], b["start_frame"], b["end_frame"], b["kind"]) for b in g], [(80, 69, 91, "gradual")])
        self.assertEqual(blend_runs([(f, 40.0, 0.05) for f in range(2, 10)], 4), [])  # a fade-in at the start is no boundary
        cuts = [{"frame": 50, "kind": "cut", "score": 1.0}, {"frame": 51, "kind": "cut", "score": 0.4},
                {"frame": 71, "kind": "cut", "score": 0.5}, {"frame": 120, "kind": "cut", "score": 0.9},
                {"frame": 125, "kind": "cut", "score": 0.95}]
        out, dropped = merge_boundaries(cuts, g, min_shot_frames=15)
        self.assertEqual([(b["frame"], b["kind"]) for b in out], [(50, "cut"), (80, "gradual"), (125, "cut")])
        self.assertEqual(dropped, 3)  # 51: same cut; 71: inside the dissolve; 120: 5 frames from a stronger one

    def test_transnet_windows_and_boundaries(self):
        for n in (1, 49, 50, 51, 237):
            wins = transnet_windows(n)
            self.assertTrue(all(len(w) == 100 for w in wins))
            middle = [i for w in wins for i in w[25:75]]
            self.assertEqual(middle[:n], list(range(n)))
        single = [0.01] * 100
        single[39] = 0.97                     # a cut: the frame after the peak starts the next shot
        for k in range(60, 70):
            single[k] = 0.7                   # a dissolve
        b = transnet_boundaries(single, None, 0.5)
        self.assertEqual([(x["frame"], x["kind"]) for x in b], [(40, "cut"), (64, "gradual")])
        self.assertEqual((b[1]["start_frame"], b[1]["end_frame"]), (60, 69))


# ---- reframe --------------------------------------------------------------------------------------
def walker(speed=60.0, start=40.0, jump_at=None, jitter=0.0):
    """One face box crossing a 640x360 frame at `speed` px/s; after `jump_at` seconds it is elsewhere (a new shot)."""
    def boxes(t, w, h):
        if jump_at is not None and t >= jump_at:
            return [{"box": [520.0, 120.0, 40.0, 48.0], "score": 0.9}]
        x = start + speed * t + (jitter if int(round(t * 5)) % 2 else -jitter)
        return [{"box": [x, 100.0, 40.0, 48.0], "score": 0.9}]
    return boxes


def shots_for(tmp, video, cut_frames):
    return detect_shots(video, engine=ScriptedShotEngine(lambda n: [{"frame": f, "kind": "cut", "score": 1.0} for f in cut_frames]),
                        ffmpeg_skill=FAKE_FFMPEG_SKILL)


def speed_ok(doc):
    v = doc["parameters"]["max_speed_px_per_s"]
    ws = doc["windows"]
    return all(abs(b["x"] - a["x"]) <= v * (b["t"] - a["t"]) + 2 and abs(b["y"] - a["y"]) <= v * (b["t"] - a["t"]) + 2
               for a, b in zip(ws, ws[1:]) if a["shot"] == b["shot"])


class ReframeTests(TmpCase):
    def face_tracks(self, boxes, duration=8.0, kind="face", name="clip.mp4"):
        video = fake_video(self.tmp, name=name, duration=duration)
        det = detect(kind, video, every=0.2, engine=ScriptedEngine(kind, boxes), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        return video, make_tracks(det)

    def test_window_sizes(self):
        self.assertEqual(window_size(640, 360, parse_aspect("9:16")[1]), (202, 360))
        self.assertEqual(window_size(640, 360, parse_aspect("1:1")[1]), (360, 360))
        self.assertEqual(window_size(640, 360, parse_aspect("4:5")[1]), (288, 360))
        self.assertEqual(window_size(360, 640, parse_aspect("16:9")[1]), (360, 202))
        for bad in ("", "9", "0:16", "9:16:1", "abc"):
            with self.assertRaises(DetectionError):
                parse_aspect(bad)

    def test_walking_subject_stays_inside_without_jumps(self):
        _video, tr = self.face_tracks(walker(60.0))
        for aspect in ("9:16", "1:1", "4:5"):
            doc = make_reframe(tr, aspect=aspect)
            rep = validate_reframe(doc)
            self.assertTrue(rep.ok, rep.errors)
            self.assertEqual(doc["schema"], REFRAME_SCHEMA)
            self.assertEqual(doc["counts"]["windows"], 40)
            self.assertEqual(doc["counts"]["subject_outside"], 0, aspect)
            self.assertTrue(all(w["subject_inside"] for w in doc["windows"]), aspect)
            self.assertTrue(speed_ok(doc), aspect)
            self.assertEqual([s["to"] for s in doc["switches"]], ["face/track_001"])
            self.assertTrue(doc["method"]["heuristic"])
            self.assertFalse(doc["method"]["identity"])
        doc = make_reframe(tr, aspect="9:16")
        self.assertGreater(doc["windows"][-1]["x"], doc["windows"][0]["x"])  # it followed the walk
        self.assertEqual(len(doc["shots"]), 1)
        self.assertIn("no --shots", " ".join(doc["warnings"]))

    def test_speed_limit_binds_and_is_reported(self):
        _video, tr = self.face_tracks(walker(120.0))
        doc = make_reframe(tr, aspect="9:16", max_speed=0.1)  # 20 px/s for a 120 px/s walk
        self.assertTrue(validate_reframe(doc).ok)
        self.assertTrue(speed_ok(doc))
        self.assertGreater(doc["counts"]["subject_outside"], 0)
        self.assertTrue(any("raise --max-speed" in w for w in doc["warnings"]))

    def test_dead_zone_holds_still_on_jitter(self):
        _video, tr = self.face_tracks(lambda t, w, h: [{"box": [300.0 + (6 if int(round(t * 5)) % 2 else -6), 100.0, 40.0, 48.0],
                                                        "score": 0.9}])
        xs = {w["x"] for w in make_reframe(tr, aspect="9:16")["windows"]}
        self.assertEqual(len(xs), 1)
        self.assertGreater(len({w["x"] for w in make_reframe(tr, aspect="9:16", dead_zone=0.0)["windows"]}), 1)

    def test_shot_boundary_allows_the_jump_and_switches_subject(self):
        video, tr = self.face_tracks(walker(40.0, jump_at=4.0))
        shots = shots_for(self.tmp, video, [100])  # 4.0 s at 25 fps
        doc = make_reframe(tr, aspect="9:16", shots=shots)
        self.assertTrue(validate_reframe(doc).ok, validate_reframe(doc).errors)
        self.assertEqual([w["shot"] for w in doc["windows"]], [0] * 20 + [1] * 20)
        a, b = doc["windows"][19], doc["windows"][20]
        self.assertGreater(abs(b["x"] - a["x"]), doc["parameters"]["max_speed_px_per_s"] * 0.2 + 2)  # jumped at the cut
        self.assertTrue(speed_ok(doc))
        self.assertEqual([(s["shot"], s["from"], s["to"]) for s in doc["switches"]],
                         [(0, None, "face/track_001"), (1, "face/track_001", "face/track_002")])
        self.assertIn("shot boundary (cut)", doc["switches"][1]["reason"])
        self.assertEqual([s["subject"] for s in doc["shots"]], ["face/track_001", "face/track_002"])
        self.assertEqual(doc["shots"][1]["subject_inside_fraction"], 1.0)
        self.assertEqual(doc["inputs"]["shots"]["shots"], 2)

    def test_object_fallback_and_hold(self):
        video, faces = self.face_tracks(lambda t, w, h: [{"box": [100.0, 100.0, 40.0, 48.0], "score": 0.9}] if t < 4 else [])
        det = detect("object", video, every=0.2, engine=ScriptedEngine("object", lambda t, w, h: [
            {"box": [400.0, 60.0, 80.0, 250.0], "score": 0.8, "label": "person"},
            {"box": [10.0, 10.0, 60.0, 60.0], "score": 0.9, "label": "cup"}] if t >= 5 else []), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        objects = make_tracks(det)
        person = next(t["id"] for t in objects["tracks"] if t["label"] == "person")
        doc = make_reframe(faces, objects=objects, aspect="9:16")
        self.assertTrue(validate_reframe(doc).ok, validate_reframe(doc).errors)
        self.assertEqual([s["to"] for s in doc["switches"]], ["face/track_001", None, "object/" + person])  # not the cup
        self.assertIn("no face or object in view", doc["switches"][1]["reason"])
        self.assertIn("(no face in view)", doc["switches"][2]["reason"])
        held = [w["x"] for w in doc["windows"] if 4.0 <= w["t"] < 5.0]
        self.assertEqual(len(set(held)), 1)
        self.assertTrue(speed_ok(doc))
        no_obj = make_reframe(faces, aspect="9:16")
        self.assertEqual([s["to"] for s in no_obj["switches"]], ["face/track_001", None])

    def test_refuses_mismatched_inputs_and_validator_rejects(self):
        _v1, tr = self.face_tracks(walker())
        other_video = fake_video(self.tmp, name="other.mp4", duration=8.0, width=640, height=360, fps=24.0)
        other = shots_for(self.tmp, other_video, [])
        with self.assertRaises(DetectionError) as cm:
            make_reframe(tr, shots=other)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            make_reframe(tr, objects=tr)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        base = make_reframe(tr, aspect="9:16")
        cases = {
            "jump": lambda d: d["windows"][5].__setitem__("x", d["windows"][4]["x"] + 200),
            "odd": lambda d: d["windows"][0].__setitem__("x", 3),
            "outside": lambda d: d["windows"][0].__setitem__("x", 600),
            "size": lambda d: d["windows"][0].__setitem__("width", 100),
            "identity": lambda d: d["method"].__setitem__("identity", True),
            "subject": lambda d: d["windows"][0].__setitem__("subject", "Alice"),
            "counts": lambda d: d["counts"].__setitem__("subject_outside", 3),
            "reason": lambda d: d["switches"][0].__setitem__("reason", ""),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(base)
            mutate(d)
            self.assertFalse(validate_reframe(d).ok, name)

    def test_cli_shots_check_and_reframe(self):
        video, tr = self.face_tracks(walker(40.0, jump_at=4.0))
        shots = shots_for(self.tmp, video, [100])
        paths = {}
        for name, doc in (("tracks", tr), ("shots", shots)):
            paths[name] = os.path.join(self.tmp, f"{name}.json")
            Path(paths[name]).write_text(json.dumps(doc), encoding="utf-8")
        rc, out = run_cli("check", paths["shots"], "--json")
        self.assertEqual((rc, out["schema"]), (0, SHOTS_SCHEMA))
        rf = os.path.join(self.tmp, "reframe.json")
        rc, out = run_cli("reframe", paths["tracks"], "--shots", paths["shots"], "--aspect", "1:1", "-o", rf, "--json")
        self.assertEqual((rc, out["window"]), (0, {"width": 360, "height": 360}), out)
        self.assertTrue(out["inputs"]["faces"]["fingerprint"].startswith("sha256:"))
        self.assertEqual(out["inputs"]["shots"]["file"], "shots.json")
        rc, out = run_cli("check", rf, "--json")
        self.assertEqual((rc, out["schema"], out["ok"]), (0, REFRAME_SCHEMA, True))
        rc, out = run_cli("reframe", paths["tracks"], "-o", rf, "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_EXISTS"))
        rc, out = run_cli("reframe", paths["tracks"], "--shots", paths["shots"], "-o", paths["shots"], "--overwrite", "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_IS_INPUT"))
        rc, out = run_cli("reframe", paths["tracks"], "--aspect", "wide", "-o", os.path.join(self.tmp, "r2.json"), "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "INVALID_ARGUMENT"))


# ---- face attributes and active speaker ----------------------------------------------------------
TURNS_A = [(0.0, 3.0), (6.0, 9.0)]
TURNS_B = [(3.0, 6.0), (9.0, 12.0)]


def talker(x, spans, size, step=0.2, shift=0.0, sharpness=400.0, frontal=1.0, y=60.0):
    """One face box whose mouth_contrast alternates high/low from sample to sample while it speaks (spans, in its
    camera's time; `shift` moves the spans), and stays still otherwise."""
    def face(t, w, h):
        on = any(a + shift <= t < b + shift for a, b in spans) and int(round(t / step)) % 2 == 0
        return {"box": [x, y, 80.0, 100.0], "score": 0.9,
                "attributes": {"size": size, "yaw_deg": 0.0, "roll_deg": 0.0, "frontal": frontal, "sharpness": sharpness,
                               "mouth_contrast": 0.45 if on else 0.2}}
    return face


def camera(*faces):
    return lambda t, w, h: [f(t, w, h) for f in faces]


def turns_doc(spans_by_speaker):
    turns = sorted(({"start": a, "end": b, "speaker": s} for s, spans in spans_by_speaker.items() for a, b in spans),
                   key=lambda u: u["start"])
    return {"schema": "diarization-skill/turns@1", "tool": "diarization/diarize", "turns": turns}


class SpeakersTests(TmpCase):
    def cams(self, shift1=0.0):
        """camera 0: a wide shot of A (left) and B (right); camera 1: a close-up of B. Alternating turns, 12 s."""
        out = []
        for i, fn in enumerate((camera(talker(60.0, TURNS_A, 0.28), talker(420.0, TURNS_B, 0.28)),
                                camera(talker(260.0, TURNS_B, 0.55, shift=shift1)))):
            video = fake_video(self.tmp, name=f"cam{i}.mp4", duration=12.0, camera=i)
            det = detect("face", video, every=0.2, engine=ScriptedEngine("face", fn), ffmpeg_skill=FAKE_FFMPEG_SKILL,
                         attributes=True)
            out.append(make_tracks(det))
        return out

    def test_pose_from_landmarks(self):
        front = [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]]
        p = pose(front)
        self.assertEqual((p["yaw_deg"], p["roll_deg"], p["frontal"]), (0.0, 0.0, 1.0))
        turned = [[40, 50], [80, 50], [70, 70], [45, 90], [75, 90]]  # nose half-way to the left eye
        p = pose(turned)
        self.assertAlmostEqual(p["yaw_deg"], 30.0, places=1)
        self.assertAlmostEqual(p["frontal"], math.cos(math.radians(30)), places=3)
        tilted = [[40, 50], [80, 90], [60, 70], [45, 90], [75, 90]]
        self.assertAlmostEqual(pose(tilted)["roll_deg"], 45.0, places=1)
        self.assertEqual(pose(None)["frontal"], None)
        self.assertEqual(pose([[1, 1], [1, 1], [1, 1]])["yaw_deg"], None)

    def test_attributes_only_when_asked_and_kept_by_tracks(self):
        video = fake_video(self.tmp, duration=2.0)
        eng = ScriptedEngine("face", camera(talker(60.0, [(0, 2)], 0.3)))
        doc = detect("face", video, every=0.5, engine=eng, ffmpeg_skill=FAKE_FFMPEG_SKILL, attributes=True)
        self.assertTrue(validate_detections(doc).ok, validate_detections(doc).errors)
        self.assertEqual(doc["parameters"]["attributes"], True)
        self.assertTrue(eng.requests[0].params["attributes"])
        d = doc["frames"][0]["detections"][0]
        self.assertEqual(d["attributes"]["mouth_contrast"], 0.45)
        tr = make_tracks(doc)
        self.assertTrue(validate_tracks(tr).ok)
        self.assertEqual(tr["tracks"][0]["boxes"][0]["attributes"]["size"], 0.3)
        plain = detect("face", video, every=0.5, engine=ScriptedEngine("face", camera(talker(60.0, [(0, 2)], 0.3))),
                       ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertNotIn("attributes", plain["frames"][0]["detections"][0])
        self.assertEqual(plain["parameters"]["attributes"], False)
        with self.assertRaises(DetectionError) as cm:
            detect("text", video, every=0.5, engine=ScriptedEngine("text"), ffmpeg_skill=FAKE_FFMPEG_SKILL, attributes=True)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        for key, value in (("frontal", 1.5), ("mouth_contrast", -0.1), ("sharpness", "high"), ("smile", 0.5)):
            bad = copy.deepcopy(doc)
            bad["frames"][0]["detections"][0]["attributes"][key] = value
            self.assertFalse(validate_detections(bad).ok, key)
            badt = copy.deepcopy(tr)
            badt["tracks"][0]["boxes"][0]["attributes"][key] = value
            self.assertFalse(validate_tracks(badt).ok, key)

    def test_mouth_motion_switches_to_the_best_view_of_the_speaker(self):
        doc = make_speakers(self.cams())
        rep = validate_speakers(doc)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(doc["schema"], SPEAKERS_SCHEMA)
        self.assertEqual(doc["method"]["name"], "mouth-motion")
        self.assertEqual([r["camera"] for r in doc["ranges"]], [0, 1, 0, 1])  # B's close-up beats the wide shot
        for got, want in zip([r["start"] for r in doc["ranges"][1:]], [3.0, 6.0, 9.0]):
            self.assertLessEqual(abs(got - want), 0.5, doc["switch"])
        self.assertEqual(doc["switch"], switch_list(doc["ranges"]))
        self.assertEqual(doc["counts"]["switches"], 3)
        self.assertTrue(all(r["track"] for r in doc["ranges"]))
        self.assertIsNone(doc["turns"])
        self.assertFalse(doc["method"]["identity"])

    def test_turns_give_the_switch_list_on_the_turns(self):
        turns = turns_doc({"SPEAKER_00": TURNS_A, "SPEAKER_01": TURNS_B})
        cams = self.cams()
        doc = make_speakers(cams, turns=turns)
        self.assertTrue(validate_speakers(doc).ok, validate_speakers(doc).errors)
        self.assertEqual(doc["switch"], "0.000-3.000:0,3.000-6.000:1,6.000-9.000:0,9.000-12.000:1")
        self.assertEqual([r["speaker"] for r in doc["ranges"]], ["SPEAKER_00", "SPEAKER_01"] * 2)
        a_track = next(t["id"] for t in cams[0]["tracks"] if t["boxes"][0]["box"][0] < 100)
        m = {(x["speaker"], x["camera"]): x["track"] for x in doc["matches"]}
        self.assertEqual(m[("SPEAKER_00", 0)], a_track)
        self.assertNotEqual(m[("SPEAKER_01", 0)], a_track)
        self.assertEqual(m[("SPEAKER_01", 1)], cams[1]["tracks"][0]["id"])
        self.assertNotIn(("SPEAKER_00", 1), m)  # A is not in camera 1
        self.assertEqual(doc["turns"]["speakers"], ["SPEAKER_00", "SPEAKER_01"])
        # a 0.4 s interjection is folded into the shot around it unless --min-shot allows it
        inter = turns_doc({"SPEAKER_00": [(0.0, 3.0), (6.0, 7.0), (7.4, 9.0)], "SPEAKER_01": [(3.0, 6.0), (7.0, 7.4), (9.0, 12.0)]})
        self.assertEqual(make_speakers(cams, turns=inter)["switch"], "0.000-3.000:0,3.000-6.000:1,6.000-9.000:0,9.000-12.000:1")
        self.assertIn("7.000-7.400:1", make_speakers(cams, turns=inter, min_shot=0.2)["switch"])
        # a turns file on another clock is moved onto camera 0's with --turns-offset
        late = turns_doc({"SPEAKER_00": [(a + 1.0, b + 1.0) for a, b in TURNS_A], "SPEAKER_01": [(a + 1.0, b + 1.0) for a, b in TURNS_B]})
        self.assertEqual(make_speakers(cams, turns=late, turns_offset=-1.0)["switch"], doc["switch"])

    def test_offsets_put_cameras_on_the_reference_timeline(self):
        turns = turns_doc({"SPEAKER_00": TURNS_A, "SPEAKER_01": TURNS_B})
        cams = self.cams(shift1=-1.0)  # camera 1 started 1 s late: its clock reads 1 s less
        doc = make_speakers(cams, offsets=[0.0, 1.0], turns=turns)
        self.assertEqual(doc["switch"], "0.000-3.000:0,3.000-6.000:1,6.000-9.000:0,9.000-12.000:1")
        self.assertEqual(doc["cameras"][1]["offset"], 1.0)
        mouth = make_speakers(cams, offsets=[0.0, 1.0])
        for got, want in zip([r["start"] for r in mouth["ranges"][1:]], [3.0, 6.0, 9.0]):
            self.assertLessEqual(abs(got - want), 0.5, mouth["switch"])

    def test_refusals_and_validator(self):
        cams = self.cams()
        for kw, code in ((dict(cameras=cams[:1]), "INVALID_ARGUMENT"), (dict(cameras=[cams[0], cams[0]]), "INVALID_ARGUMENT"),
                         (dict(cameras=cams, offsets=[0.0]), "INVALID_ARGUMENT"), (dict(cameras=cams, window=0), "INVALID_ARGUMENT"),
                         (dict(cameras=cams, turns={"schema": "x"}), "INVALID_DETECTIONS"),
                         (dict(cameras=cams, turns=turns_doc({"Alice": [(0, 1)]})), "INVALID_DETECTIONS")):
            with self.assertRaises(DetectionError) as cm:
                make_speakers(**kw)
            self.assertEqual(cm.exception.code, code, kw.keys())
        video = fake_video(self.tmp, name="plain.mp4", duration=12.0)
        plain = make_tracks(detect("face", video, every=0.2, engine=ScriptedEngine("face", camera(talker(60.0, TURNS_A, 0.3))),
                                   ffmpeg_skill=FAKE_FFMPEG_SKILL))
        with self.assertRaises(DetectionError) as cm:
            make_speakers([plain, cams[1]])
        self.assertIn("--attributes", cm.exception.message)
        base = make_speakers(cams, turns=turns_doc({"SPEAKER_00": TURNS_A, "SPEAKER_01": TURNS_B}))
        cases = {
            "gap": lambda d: d["ranges"][1].__setitem__("start", 3.5),
            "switch": lambda d: d.__setitem__("switch", "0.000-12.000:0"),
            "camera": lambda d: d["ranges"][0].__setitem__("camera", 5),
            "identity": lambda d: d["method"].__setitem__("identity", True),
            "speaker": lambda d: d["ranges"][0].__setitem__("speaker", "Alice"),
            "counts": lambda d: d["counts"].__setitem__("switches", 9),
            "end": lambda d: d["ranges"][-1].__setitem__("end", 11.0),
            "turns": lambda d: d.__setitem__("turns", None),
            "reason": lambda d: d["ranges"][0].__setitem__("reason", ""),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(base)
            mutate(d)
            self.assertFalse(validate_speakers(d).ok, name)

    def test_cli_speakers_and_check(self):
        cams = self.cams()
        paths = []
        for i, doc in enumerate(cams):
            paths.append(os.path.join(self.tmp, f"cam{i}_tracks.json"))
            Path(paths[-1]).write_text(json.dumps(doc), encoding="utf-8")
        tp = os.path.join(self.tmp, "turns.json")
        Path(tp).write_text(json.dumps(turns_doc({"SPEAKER_00": TURNS_A, "SPEAKER_01": TURNS_B})), encoding="utf-8")
        out_p = os.path.join(self.tmp, "speakers.json")
        rc, out = run_cli("speakers", "--camera", paths[0], "--camera", paths[1], "--turns", tp, "--offsets", "0,0", "-o", out_p, "--json")
        self.assertEqual((rc, out["switch"]), (0, "0.000-3.000:0,3.000-6.000:1,6.000-9.000:0,9.000-12.000:1"), out)
        self.assertEqual([c["file"] for c in out["cameras"]], ["cam0_tracks.json", "cam1_tracks.json"])
        self.assertEqual(out["turns"]["file"], "turns.json")
        rc, out = run_cli("check", out_p, "--json")
        self.assertEqual((rc, out["schema"], out["ok"]), (0, SPEAKERS_SCHEMA, True))
        rc, out = run_cli("speakers", "--camera", paths[0], "--camera", paths[1], "-o", out_p, "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_EXISTS"))
        rc, out = run_cli("speakers", "--camera", paths[0], "--camera", paths[1], "--offsets", "0,x",
                          "-o", os.path.join(self.tmp, "s2.json"), "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "INVALID_ARGUMENT"))
        rc, out = run_cli("speakers", "--camera", paths[0], "--camera", paths[1], "-o", paths[1], "--overwrite", "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_IS_INPUT"))


# ---- personal and patient data ----------------------------------------------------------------------
def slide_lines(lines, h=720):
    """A text engine reading these lines off a 1280x720 slide, one per 100 px."""
    def boxes(t, w, hh):
        return [{"box": [80.0, 80.0 + 100 * i, 30.0 * len(s) * w / 1280.0 + 10, 40.0], "score": 0.9, "label": s, "lang": "eng"}
                for i, s in enumerate(lines)]
    return boxes


class PiiTests(TmpCase):
    def text_doc(self, lines, video=False, **kw):
        if video:
            media = fake_video(self.tmp, name=kw.pop("name", "talk.mp4"), duration=4.0, width=1280, height=720)
            return detect("text", media, every=1.0, engine=ScriptedEngine("text", slide_lines(lines)), ffmpeg_skill=FAKE_FFMPEG_SKILL,
                          lang="eng", **kw)
        img = os.path.join(self.tmp, kw.pop("name", "slide.png"))
        write_png(img, 1280, 720, (255, 255, 255))
        return detect("text", img, engine=ScriptedEngine("text", slide_lines(lines)), lang="eng", **kw)

    def test_patterns_per_locale(self):
        cases = [
            ("intl", "Patient ID: MRN 00482913", ["patient_id"]),
            ("intl", "DOB: 1956-03-14", ["date_of_birth"]),
            ("intl", "Date of birth 14 March 1956", ["date_of_birth"]),
            ("intl", "born on Mar 3, 1956", ["date_of_birth"]),
            ("intl", "Tel: +44 20 7946 0958", ["phone"]),
            ("intl", "Patient: John Smith", ["name"]),
            ("intl", "Mrs. Jane Doe, 71", ["name"]),
            ("intl", "Address: 12 Baker Street, London", ["address"]),
            ("intl", "Published 2024-05-01", []),
            ("intl", "Sensitivity 92% (n = 1203), 95% CI 1.2-3.4", []),
            ("intl", "Case 3: 64-year-old man", []),
            ("intl", "Call (555) 123-4567", []),          # NANP format only with en-US
            ("en-US", "Call (555) 123-4567", ["phone"]),
            ("en-US", "SSN: 123-45-6789", ["patient_id"]),
            ("en-US", "1600 Pennsylvania Avenue NW, Washington, DC 20500", ["address"]),
            ("en-GB", "NHS No: 943 476 5919", ["patient_id"]),
            ("en-GB", "SW1A 1AA", ["address"]),
            ("ja-JP", "患者ID: 0012345", ["patient_id"]),
            ("ja-JP", "生年月日: 昭和45年3月14日", ["date_of_birth"]),
            ("ja-JP", "生年月日 1970/03/14", ["date_of_birth"]),
            ("ja-JP", "電話 03-1234-5678", ["phone"]),
            ("ja-JP", "〒100-0005 東京都千代田区丸の内", ["address"]),
            ("ja-JP", "山田 太郎 様", ["name"]),
            ("ja-JP", "2024年5月1日 発表", []),
            ("intl", "患者ID: 0012345", []),              # the Japanese profile is opt-in
        ]
        for loc, text, want in cases:
            self.assertEqual(sorted({h["category"] for h in find(text, [loc])}), want, (loc, text))
        h = find("DOB: 1956-03-14", ["intl"])[0]
        self.assertEqual(h["excerpt"], "DOB: ####-##-##")
        self.assertNotIn("1956", h["excerpt"])
        self.assertEqual(parse_locales(None), ["intl"])
        self.assertEqual(parse_locales(["intl,ja-JP", "ja-JP"]), ["intl", "ja-JP"])
        with self.assertRaises(DetectionError):
            parse_locales(["xx-YY"])

    def test_slide_with_patient_id_and_birth_date_gives_both_regions(self):
        doc = make_pii(self.text_doc(["Case presentation", "Patient ID: MRN 00482913", "DOB: 1956-03-14", "Chief complaint: chest pain"]))
        rep = validate_pii(doc)
        self.assertTrue(rep.ok, rep.errors)
        self.assertEqual(doc["schema"], PII_SCHEMA)
        self.assertEqual([r["category"] for r in doc["regions"]], ["patient_id", "date_of_birth"])
        self.assertEqual([r["box"][1] for r in doc["regions"]], [180.0, 280.0])  # the two lines
        for r in doc["regions"]:
            self.assertEqual(r["decision"], "CONFIRM")
            self.assertTrue(r["reason"])
            x = r["redact"]
            self.assertTrue(x["x"] <= r["box"][0] and x["x"] + x["width"] >= r["box"][0] + r["box"][2])
            self.assertEqual((x["width"] % 2, x["height"] % 2), (0, 0))
            self.assertNotIn("00482913", json.dumps(r))
        self.assertEqual([(s["start"], s["end"]) for s in doc["redact"]], [(None, None)] * 2)
        self.assertEqual(doc["counts"]["confirm"], 2)
        self.assertEqual(doc["parameters"]["locales"], ["intl"])

    def test_slide_with_none_gives_none(self):
        doc = make_pii(self.text_doc(["Results", "Sensitivity 92% (n = 1203)", "Published 2024-05-01", "Follow-up: 12 months"]))
        self.assertTrue(validate_pii(doc).ok)
        self.assertEqual((doc["regions"], doc["redact"]), ([], []))
        self.assertTrue(any("pattern check" in w for w in doc["warnings"]))

    def test_video_spans_faces_and_locale(self):
        doc = make_pii(self.text_doc(["患者ID: 0012345", "Summary"], video=True), locales=["ja-JP"])
        self.assertTrue(validate_pii(doc).ok, validate_pii(doc).errors)
        self.assertEqual(len(doc["regions"]), 4)  # one line at each of 4 sample times
        self.assertEqual([(s["start"], s["end"], len(s["regions"])) for s in doc["redact"]], [(0.0, 4.0, 4)])
        text = self.text_doc(["Summary"], video=True, name="talk.mp4")
        media = fake_video(self.tmp, name="talk.mp4", duration=4.0, width=1280, height=720)
        faces = detect("face", media, every=1.0, engine=ScriptedEngine("face", lambda t, w, h: [{"box": [900.0, 100.0, 120.0, 150.0],
                                                                                                    "score": 0.9}]),
                       ffmpeg_skill=FAKE_FFMPEG_SKILL)
        doc = make_pii(text, faces=faces)
        self.assertEqual({r["category"] for r in doc["regions"]}, {"face"})
        self.assertTrue(all(r["decision"] == "CONFIRM" for r in doc["regions"]))
        self.assertEqual(doc["inputs"]["faces"]["kind"], "face")
        other = fake_video(self.tmp, name="other.mp4", duration=4.0, width=1280, height=720, fps=24.0)
        faces2 = detect("face", other, every=1.0, engine=ScriptedEngine("face"), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        with self.assertRaises(DetectionError) as cm:
            make_pii(text, faces=faces2)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(DetectionError) as cm:
            make_pii(faces)
        self.assertEqual(cm.exception.code, "INVALID_ARGUMENT")

    def test_validator_rejects_silent_removal(self):
        base = make_pii(self.text_doc(["Patient ID: MRN 00482913", "DOB: 1956-03-14"]))
        cases = {
            "decision": lambda d: d["regions"][0].__setitem__("decision", "REMOVED"),
            "dropped": lambda d: d["redact"].pop(),
            "reason": lambda d: d["regions"][0].__setitem__("reason", ""),
            "odd": lambda d: d["regions"][0]["redact"].__setitem__("width", 3),
            "category": lambda d: d["regions"][0].__setitem__("category", "secret"),
            "counts": lambda d: d["counts"].__setitem__("regions", 5),
            "locale": lambda d: d["parameters"].__setitem__("locales", ["xx"]),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(base)
            mutate(d)
            self.assertFalse(validate_pii(d).ok, name)

    def test_cli_pii_and_check(self):
        tp = os.path.join(self.tmp, "text.json")
        Path(tp).write_text(json.dumps(self.text_doc(["Patient ID: MRN 00482913", "DOB: 1956-03-14"])), encoding="utf-8")
        out_p = os.path.join(self.tmp, "pii.json")
        rc, out = run_cli("pii", tp, "-o", out_p, "--json")
        self.assertEqual((rc, out["counts"]["regions"]), (0, 2), out)
        self.assertEqual(out["inputs"]["text"]["file"], "text.json")
        rc, out = run_cli("check", out_p, "--json")
        self.assertEqual((rc, out["schema"], out["ok"]), (0, PII_SCHEMA, True))
        rc, out = run_cli("pii", tp, "--locale", "xx", "-o", os.path.join(self.tmp, "p2.json"), "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "INVALID_ARGUMENT"))
        rc, out = run_cli("pii", tp, "-o", tp, "--overwrite", "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_IS_INPUT"))


# ---- small pieces ---------------------------------------------------------------------------------
class PieceTests(unittest.TestCase):
    def test_boxes(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        kept = nms([{"box": [0, 0, 10, 10], "score": 0.5, "label": "a"}, {"box": [1, 0, 10, 10], "score": 0.9, "label": "b"}], 0.5)
        self.assertEqual([k["score"] for k in kept], [0.9])
        kept = nms([{"box": [0, 0, 10, 10], "score": 0.5, "label": "a"}, {"box": [1, 0, 10, 10], "score": 0.9, "label": "b"}], 0.5,
                   by_label=True)
        self.assertEqual(len(kept), 2)

    def test_metrics(self):
        m = match_boxes([[0, 0, 10, 10], [50, 50, 10, 10]], [[1, 1, 10, 10]])
        self.assertEqual((m["recall"], m["precision"], m["missed"]), (0.5, 1.0, [1]))
        self.assertEqual(cer("本日の発表", "本日 の 発表"), 0.0)
        self.assertAlmostEqual(cer("abcd", "abxd"), 0.25)
        self.assertEqual(cer("", ""), 0.0)

    def test_tesseract_tsv_lines_and_cjk_join(self):
        tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
               "1\t1\t0\t0\t0\t0\t0\t0\t800\t600\t-1\t\n"
               "5\t1\t1\t1\t1\t1\t10\t10\t20\t20\t90\t本日\n"
               "5\t1\t1\t1\t1\t2\t32\t12\t20\t20\t80\tの\n"
               "5\t1\t1\t1\t1\t3\t60\t10\t40\t22\t70\tTalk\n"
               "5\t1\t1\t1\t2\t1\t10\t50\t30\t20\t-1\t \n"
               "5\t1\t1\t1\t3\t1\t10\t90\t30\t20\t60\tHi\n")
        size, words = parse_tsv(tsv)
        self.assertEqual((size, len(words)), ((800, 600), 4))
        lines = group(words, "line")
        self.assertEqual([ln["label"] for ln in lines], ["本日の Talk", "Hi"])
        self.assertEqual(lines[0]["box"], [10.0, 10.0, 90.0, 22.0])
        self.assertEqual(lines[0]["score"], 0.8)
        self.assertEqual(len(group(words, "word")), 4)
        self.assertEqual(join_words(["縦", "書き", "の", "abc", "def"]), "縦書きの abc def")


# ---- CLI, contract, doctor ------------------------------------------------------------------------
class CliTests(TmpCase):
    def test_faces_tracks_check_round_trip(self):
        video = fake_video(self.tmp)
        det = os.path.join(self.tmp, "det.json")
        doc = detect("face", video, every=0.5, engine=ScriptedEngine("face", moving_box), ffmpeg_skill=FAKE_FFMPEG_SKILL)
        Path(det).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        rc, out = run_cli("check", det, "--json")
        self.assertEqual((rc, out["ok"], out["schema"]), (0, True, DETECTIONS_SCHEMA))
        tracks = os.path.join(self.tmp, "tracks.json")
        rc, out = run_cli("tracks", det, "-o", tracks, "--json")
        self.assertEqual((rc, out["counts"]["tracks"]), (0, 2))
        self.assertTrue(out["detections"]["fingerprint"].startswith("sha256:"))
        rc, out = run_cli("check", tracks, "--json")
        self.assertEqual((rc, out["schema"]), (0, TRACKS_SCHEMA))
        rc, out = run_cli("tracks", det, "-o", tracks, "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_EXISTS"))
        rc, out = run_cli("tracks", det, "-o", det, "--overwrite", "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "OUTPUT_IS_INPUT"))
        bad = copy.deepcopy(doc)
        bad["frames"][0]["detections"][0]["label"] = "Alice"
        Path(det).write_text(json.dumps(bad), encoding="utf-8")
        rc, out = run_cli("check", det, "--json")
        self.assertEqual((rc, out["ok"]), (2, False))

    def test_cli_argument_errors(self):
        rc, out = run_cli("faces", os.path.join(self.tmp, "missing.mp4"), "--json")
        self.assertEqual((rc, out["error"]["code"]), (2, "INPUT_NOT_FOUND"))
        rc, out = run_cli("check", os.path.join(self.tmp, "missing.json"), "--json")
        self.assertEqual(rc, 2)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["faces", "x.mp4", "--every", "1", "--scenes"])

    def test_doctor_reports_missing_with_install_lines(self):
        d = doctor(models_dir=os.path.join(self.tmp, "nope"), ffmpeg_skill=os.path.join(self.tmp, "nope2"))
        self.assertFalse(d["ok"])
        for k in ("face_model", "object_model"):
            self.assertEqual(d["checks"][k]["status"], "MISSING")
            self.assertIn("http", d["checks"][k]["install"])
        for k, v in d["checks"].items():
            if v["status"] != "OK":
                self.assertTrue(v.get("install"), k)
        rc, doc = run_cli("doctor", "--models-dir", os.path.join(self.tmp, "nope"), "--json")
        self.assertEqual((rc, doc["ok"]), (3, False))
        self.assertNotIn(self.tmp, json.dumps(doc))
        d = doctor(ffmpeg_skill=FAKE_FFMPEG_SKILL)
        self.assertEqual(d["checks"]["ffmpeg_skill"]["status"], "OK")


class ContractTests(unittest.TestCase):
    def test_os_registry_keys_and_provides(self):
        c = contract()
        for k in ("skill_id", "id", "skill", "version", "contract_version", "provides", "dependencies", "not_provided"):
            self.assertIn(k, c)
        self.assertEqual(c["version"], __version__)
        self.assertEqual({p["id"] for p in c["provides"]},
                         {"vision.detect_faces", "vision.detect_text", "vision.detect_objects", "vision.track_boxes",
                          "vision.detect_shots", "vision.reframe_path", "vision.active_speaker",
                          "vision.find_personal_data"})
        tool_ids = {t["tool_id"] for t in TOOLS}
        for p in c["provides"]:
            self.assertEqual(set(p), {"id", "lifecycle", "tool_id"})
            self.assertEqual(p["lifecycle"], "EXPERIMENTAL")
            self.assertIn(p["tool_id"], tool_ids)
        for d in c["dependencies"]:
            self.assertIn("version_range", d)
        ff = next(d for d in c["dependencies"] if d["id"] == "ffmpeg-skill")
        self.assertEqual(ff["version_range"], ">=2.0.0 <4.0.0")
        self.assertFalse(c["execution"]["network"])
        self.assertEqual({e["exit_code"] for e in ERRORS.values()}, {2, 3, 4})
        self.assertIn("face_detection_yunet_2026may.onnx", c["engines"]["known_model_sha256"].values())

    def test_version_comes_from_pyproject(self):
        import re
        want = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(__version__, want)

    def test_cli_contract(self):
        rc, doc = run_cli("contract", "--json")
        self.assertEqual((rc, doc["skill_id"]), (0, "detection"))

    def test_no_shell_no_download_no_video_decoding_in_the_skill(self):
        for f in (ROOT / "src" / "detection_skill").rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("shell=True", text, f.name)
            for banned in ("urllib.request", "requests.", "urlopen", "http.client", "VideoCapture", "hf_hub_download",
                           'which("ffmpeg")', 'which("ffprobe")', '["ffmpeg"', '["ffprobe"'):
                self.assertNotIn(banned, text, f"{f.name}: {banned}")

    def test_known_model_table_is_well_formed(self):
        for sha, info in KNOWN_MODELS.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertIn(info["kind"], ("face", "object"))


@unittest.skipUnless(OS_DIR and (OS_DIR / "registry" / "conformance.py").is_file(),
                     "needs an AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS)")
class ConformanceTests(unittest.TestCase):
    def test_registry_checks_pass(self):
        sys.path.insert(0, str(OS_DIR))
        try:
            from registry import conformance as conf
        finally:
            sys.path.remove(str(OS_DIR))
        results = conf.run_static_checks(contract())
        results.append(conf.check_no_unsafe_shell_out(str(ROOT / "src" / "detection_skill")))
        results.append(conf.check_doctor_status(lambda: doctor()))
        for r in results:
            print(f"\n  conformance {r.check}: {r.status} ({r.detail})", end="")
            self.assertEqual(r.status, "PASS", f"{r.check}: {r.detail}")


# ---- frames through the real ffmpeg-skill (fake engine) -------------------------------------------
@unittest.skipUnless(REAL_VIDEO, REAL_VIDEO_WHY)
class RealFramesTests(TmpCase):
    def test_look_frames_have_the_source_size_and_times(self):
        video = os.path.join(self.tmp, "src.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=3",
                        "-pix_fmt", "yuv420p", video], check=True)
        eng = ScriptedEngine("face", lambda t, w, h: [{"box": [w - 50, h - 50, 100, 100], "score": 0.9}])
        doc = detect("face", video, every=1.0, engine=eng, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        self.assertEqual(doc["sampling"]["times"], [0.0, 1.0, 2.0])
        self.assertEqual(doc["sampling"]["extracted_size"], {"width": 1920, "height": 1080})
        self.assertEqual(doc["frames"][0]["detections"][0]["box"], [1870.0, 1030.0, 50.0, 50.0])  # clipped at the edge
        real_version = json.loads(Path(FFMPEG_SKILL_DIR, "package.json").read_text(encoding="utf-8"))["version"]
        self.assertEqual(doc["sampling"]["ffmpeg_skill_version"], real_version)
        self.assertIn(real_version.split(".")[0], ("2", "3"))
        self.assertTrue(validate_detections(doc).ok)
        doc = detect("face", video, scenes=True, engine=eng, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        self.assertEqual(doc["sampling"]["method"], "scenes")


# ---- real engines ---------------------------------------------------------------------------------
def compose(cv2, np, width, height, placements, bg=128):
    """Paste (image, scale, x, y[, crop]) onto a gray canvas; returns the canvas."""
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    for img, s, x, y in placements:
        im = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def tbox(box, s, x, y, crop=(0, 0)):
    return [(box[0] - crop[0]) * s + x, (box[1] - crop[1]) * s + y, box[2] * s, box[3] * s]


@unittest.skipUnless(HAVE_CV2 and YUNET_FILE, "needs DETECTION_MODELS_DIR with face_detection_yunet_*.onnx and opencv-python-headless")
class RealFacesTests(TmpCase):
    def test_fixture_recall(self):
        import cv2
        import numpy as np
        img = cv2.imread(str(FIXTURES / "astronaut.jpg"))
        crop = img[30:210, 150:300]  # the face region, upscaled 3x below: a face larger than YuNet's ~300 px
        placements = [(img, 0.25, 40, 40), (img, 0.5, 220, 40), (img, 1.0, 520, 40), (crop, 3.0, 1100, 40), (img, 0.75, 40, 600)]
        canvas = compose(cv2, np, 1920, 1080, placements)
        truth = [tbox(ASTRONAUT_FACE, 0.25, 40, 40), tbox(ASTRONAUT_FACE, 0.5, 220, 40), tbox(ASTRONAUT_FACE, 1.0, 520, 40),
                 tbox(ASTRONAUT_FACE, 3.0, 1100, 40, crop=(150, 30)), tbox(ASTRONAUT_FACE, 0.75, 40, 600)]
        path = os.path.join(self.tmp, "faces.png")
        cv2.imwrite(path, canvas)
        doc = detect("face", path, models_dir=MODELS_DIR, model=YUNET_FILE)
        self.assertTrue(validate_detections(doc).ok)
        found = [d["box"] for d in doc["frames"][0]["detections"]]
        m = match_boxes(truth, found, 0.5)
        print(f"\n  YuNet fixture: recall {m['recall']:.2f} ({m['matched']}/{m['truth']}), precision {m['precision']:.2f}, "
              f"missed {m['missed']}, IoUs {[p['iou'] for p in m['pairs']]}", end="")
        self.assertGreaterEqual(m["recall"], 0.8)
        self.assertGreaterEqual(m["precision"], 0.8)
        d = doc["frames"][0]["detections"][0]
        self.assertIsNone(d["label"])
        self.assertEqual(len(d["landmarks"]), 5)
        self.assertEqual(doc["engine"]["model_sha256"], doc["engine"]["files"][0]["sha256"])
        self.assertEqual(len(doc["engine"]["model_sha256"]), 64)

    @unittest.skipUnless(REAL_VIDEO, REAL_VIDEO_WHY)
    def test_moving_face_keeps_one_track(self):
        video = os.path.join(self.tmp, "moving.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=960x540:r=25:d=4",
                        "-loop", "1", "-i", str(FIXTURES / "astronaut.jpg"), "-filter_complex",
                        "[1:v]scale=256:256[a];[0:v][a]overlay=x='40+t*60':y=140:shortest=1,format=yuv420p", video], check=True)
        doc = detect("face", video, every=0.2, models_dir=MODELS_DIR, model=YUNET_FILE, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        hits = sum(1 for f in doc["frames"] if f["detections"])
        tr = make_tracks(doc)
        self.assertTrue(validate_tracks(tr).ok)
        print(f"\n  moving face: {hits}/{len(doc['frames'])} frames with a face, {tr['counts']['tracks']} track(s)", end="")
        self.assertGreaterEqual(hits, len(doc["frames"]) - 1)
        self.assertEqual(tr["counts"]["tracks"], 1)
        r = tr["tracks"][0]["redact"][0]
        first, last = tr["tracks"][0]["boxes"][0]["box"], tr["tracks"][0]["boxes"][-1]["box"]
        self.assertLess(r["x"], first[0])
        self.assertGreater(r["x"] + r["width"], last[0] + last[2])


@unittest.skipUnless(HAVE_CV2 and HAVE_ORT and YOLOX_FILE, "needs DETECTION_MODELS_DIR with yolox_*.onnx, onnxruntime and opencv")
class RealObjectsTests(TmpCase):
    def test_fixture_recall(self):
        truth = {"astronaut.jpg": ("person", ASTRONAUT_PERSON), "chelsea.jpg": ("cat", CHELSEA_CAT), "coffee.jpg": ("cup", COFFEE_CUP)}
        matched = 0
        for name, (label, box) in truth.items():
            doc = detect("object", str(FIXTURES / name), models_dir=MODELS_DIR, model=YOLOX_FILE)
            self.assertTrue(validate_detections(doc).ok)
            dets = doc["frames"][0]["detections"]
            best = max((iou(box, d["box"]) for d in dets if d["label"] == label), default=0.0)
            print(f"\n  YOLOX {name}: {label} best IoU {best:.2f}; all: {[(d['label'], d['score']) for d in dets]}", end="")
            matched += best >= 0.5
        print(f"\n  YOLOX fixture: recall {matched}/3 ({YOLOX_FILE})", end="")
        self.assertEqual(matched, 3)

    def test_classes_filter(self):
        doc = detect("object", str(FIXTURES / "coffee.jpg"), models_dir=MODELS_DIR, model=YOLOX_FILE, classes=["person"])
        self.assertEqual(doc["counts"]["detections"], 0)
        self.assertEqual(doc["parameters"]["classes"], ["person"])


@unittest.skipUnless(HAVE_TESS, "needs tesseract with jpn, jpn_vert and eng traineddata")
@unittest.skipUnless(HAVE_PIL and JP_FONT, "needs Pillow and a Japanese font (DETECTION_TEST_FONT) to render the fixture")
class RealTextTests(TmpCase):
    JA = ["本日の発表内容について説明します", "第二章 映像制作の自動化", "会場は東京国際フォーラムです", "質疑応答は最後に行います"]
    EN = ["Welcome to the annual conference", "Speaker: Dr. Jane Smith", "Session 3 starts at 14:30", "Please silence your phones"]
    VERT = "縦書きの文章を読み取ります"
    CER_JA = float(os.environ.get("DETECTION_CER_JA", "0.10"))
    CER_EN = float(os.environ.get("DETECTION_CER_EN", "0.05"))

    def render(self, lines, size=(1280, 720), font_px=44, vertical=False):
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(JP_FONT, font_px)
        im = Image.new("RGB", size, "white")
        dr = ImageDraw.Draw(im)
        if vertical:
            for i, ch in enumerate(lines[0]):
                dr.text((size[0] // 2 - font_px // 2, 40 + i * int(font_px * 1.3)), ch, font=font, fill="black")
        else:
            for i, line in enumerate(lines):
                dr.text((80, 80 + i * int(font_px * 2.4)), line, font=font, fill="black")
        path = os.path.join(self.tmp, f"slide{len(os.listdir(self.tmp))}.png")
        im.save(path)
        return path

    def ocr(self, path, **kw):
        doc = detect("text", path, tessdata_dir=os.environ.get("DETECTION_TESSDATA_DIR"), **kw)
        self.assertTrue(validate_detections(doc).ok, validate_detections(doc).errors)
        return doc, "\n".join(d["label"] for fr in doc["frames"] for d in fr["detections"])

    def test_japanese_and_english_cer(self):
        for name, lines, lang, limit in (("ja", self.JA, "jpn+eng", self.CER_JA), ("en", self.EN, "eng", self.CER_EN)):
            doc, text = self.ocr(self.render(lines), lang=lang)
            score = cer("".join(lines), text)
            variant = doc["engine"]["model"]
            print(f"\n  Tesseract {name} ({variant}): CER {score:.4f} over {len(''.join(lines))} chars; lines {doc['counts']['detections']}",
                  end="")
            self.assertLessEqual(score, limit, f"{name}: {text!r}")
            self.assertEqual(doc["counts"]["detections"], len(lines))

    def test_vertical_japanese(self):
        doc, text = self.ocr(self.render([self.VERT], size=(400, 900), vertical=True), vertical=True)
        score = cer(self.VERT, text)
        print(f"\n  Tesseract jpn_vert: CER {score:.4f} ({text!r})", end="")
        self.assertLessEqual(score, self.CER_JA)
        box = doc["frames"][0]["detections"][0]["box"]
        self.assertGreater(box[3], box[2] * 3)  # a tall, narrow column

    @unittest.skipUnless(REAL_VIDEO, REAL_VIDEO_WHY)
    def test_text_from_a_video_frame(self):
        slide = self.render(self.EN)
        video = os.path.join(self.tmp, "slides.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", slide, "-t", "2", "-r", "25", "-pix_fmt", "yuv420p", video],
                       check=True)
        doc, text = self.ocr(video, at=[1.0], lang="eng", ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        score = cer("".join(self.EN), text)
        print(f"\n  Tesseract eng via ffmpeg-skill look: CER {score:.4f}", end="")
        self.assertLessEqual(score, self.CER_EN)
        self.assertEqual(doc["frames"][0]["t"], 1.0)


# ---- shots and reframe on real frames -------------------------------------------------------------
SHOTS_WHY = "needs scenedetect and opencv (pip install '.[shots]'), ffmpeg and an ffmpeg-skill checkout"
# 320x180 at 25 fps, 225 frames: six lavfi sources (several of them moving), joined by three hard cuts and two
# cross-dissolves (ffmpeg xfade). Ground truth from the construction: cuts at frames 50, 120, 185; dissolves blend
# frames 70-89 (0.8 s) and 135-159 (1.0 s).
SHOT_SOURCES = ["testsrc2=s=320x180:r=25:d=2", "smptehdbars=s=320x180:r=25:d=1.6", "mandelbrot=s=320x180:r=25,trim=duration=2",
                "gradients=s=320x180:r=25:d=1.6:speed=0.05", "testsrc=s=320x180:r=25:d=2", "yuvtestsrc=s=320x180:r=25:d=1.6"]
SHOT_CUTS = [50, 120, 185]
SHOT_DISSOLVES = [(70, 89), (135, 159)]


def render_shots_fixture(path):
    norm = "format=yuv420p,setsar=1,fps=25,settb=1/25,setpts=N"
    graph = ";".join(f"[{i}]{norm}[s{i}]" for i in range(6)) + (
        ";[s0][s1]concat=n=2,settb=1/25,setpts=N[a];[a][s2]xfade=transition=fade:duration=0.8:offset=2.8,settb=1/25,setpts=N[b]"
        ";[b][s3]concat=n=2,settb=1/25,setpts=N[c];[c][s4]xfade=transition=fade:duration=1.0:offset=5.4,settb=1/25,setpts=N[d]"
        ";[d][s5]concat=n=2,format=yuv420p[out]")
    argv = ["ffmpeg", "-v", "error", "-y"]
    for src in SHOT_SOURCES:
        argv += ["-f", "lavfi", "-i", src]
    subprocess.run(argv + ["-filter_complex", graph, "-map", "[out]", "-r", "25", "-c:v", "libx264", "-crf", "18", "-g", "50", path],
                   check=True)


@unittest.skipUnless(HAVE_SCENEDETECT and HAVE_CV2 and REAL_VIDEO, SHOTS_WHY)
class RealShotsTests(TmpCase):
    def test_cuts_within_one_frame_and_dissolves_found(self):
        video = os.path.join(self.tmp, "shots.mp4")
        render_shots_fixture(video)
        doc = detect_shots(video, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        self.assertTrue(validate_shots(doc).ok, validate_shots(doc).errors)
        self.assertEqual(doc["frames"]["frame_count"], 225)
        cuts = [b["frame"] for b in doc["boundaries"] if b["kind"] == "cut"]
        gradual = [b for b in doc["boundaries"] if b["kind"] == "gradual"]
        print(f"\n  PySceneDetect {doc['engine']['version']}: cuts {cuts} (truth {SHOT_CUTS}); gradual "
              f"{[(b['frame'], b['transition']['start_frame'], b['transition']['end_frame']) for b in gradual]} "
              f"(truth {SHOT_DISSOLVES}); {doc['engine']['engine_seconds']}s engine, {doc['processing_seconds']}s total", end="")
        self.assertEqual(len(cuts), len(SHOT_CUTS), cuts)  # no missed and no extra hard cut
        for want, got in zip(SHOT_CUTS, cuts):
            self.assertLessEqual(abs(got - want), 1, (want, got))
        self.assertEqual(len(gradual), len(SHOT_DISSOLVES))
        for (a, z), b in zip(SHOT_DISSOLVES, gradual):
            self.assertTrue(a <= b["frame"] <= z, (a, z, b["frame"]))
            self.assertLessEqual(abs(b["transition"]["start_frame"] - a), 3)
            self.assertLessEqual(abs(b["transition"]["end_frame"] - z), 3)
        self.assertEqual(doc["counts"]["shots"], 6)
        self.assertEqual(doc["engine"]["id"], "pyscenedetect")
        # TransNetV2 is optional: without its model file doctor says MISSING and the op above still ran.
        d = doctor(models_dir=MODELS_DIR, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        if not _models_file(lambda n: n.startswith("transnetv2") and n.endswith(".onnx")):
            self.assertEqual(d["checks"]["transnetv2_model"]["status"], "MISSING")
        self.assertTrue(d["ready"]["shots"])


@unittest.skipUnless(HAVE_CV2 and YUNET_FILE and HAVE_SCENEDETECT and REAL_VIDEO,
                     "needs YuNet in DETECTION_MODELS_DIR, opencv, scenedetect, ffmpeg and an ffmpeg-skill checkout")
class RealReframeTests(TmpCase):
    def test_walking_speaker_stays_inside_the_window(self):
        # 960x540, 8 s: a face walks left to right at 105 px/s for 6 s, then a cut to a second shot where a face stands
        # on the left.
        video = os.path.join(self.tmp, "walk.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=960x540:r=25:d=6",
                        "-f", "lavfi", "-i", "color=c=0x305070:s=960x540:r=25:d=2", "-loop", "1", "-i", str(FIXTURES / "astronaut.jpg"),
                        "-filter_complex", "[2:v]scale=256:256,split[a][b];[0:v][a]overlay=x='40+t*105':y=140:shortest=1[s0];"
                        "[1:v][b]overlay=x=60:y=200:shortest=1[s1];[s0][s1]concat=n=2,format=yuv420p[out]",
                        "-map", "[out]", "-r", "25", video], check=True)
        det = detect("face", video, every=0.2, models_dir=MODELS_DIR, model=YUNET_FILE, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        tracks = make_tracks(det)
        shots = detect_shots(video, ffmpeg_skill=str(FFMPEG_SKILL_DIR))
        self.assertEqual([b["frame"] for b in shots["boundaries"]], [150])
        for aspect in ("9:16", "1:1", "4:5"):
            doc = make_reframe(tracks, aspect=aspect, shots=shots)
            rep = validate_reframe(doc)
            self.assertTrue(rep.ok, rep.errors)
            ws = doc["windows"]
            faces_at = {fr["t"]: fr["detections"] for fr in det["frames"]}
            inside = 0
            for w in ws:
                for d in faces_at[w["t"]]:
                    x, y, bw, bh = d["box"]
                    inside += x >= w["x"] and y >= w["y"] and x + bw <= w["x"] + w["width"] and y + bh <= w["y"] + w["height"]
            n_faces = sum(len(v) for v in faces_at.values())
            v = doc["parameters"]["max_speed_px_per_s"]
            worst = max(abs(b["x"] - a["x"]) / (b["t"] - a["t"]) for a, b in zip(ws, ws[1:]) if a["shot"] == b["shot"])
            print(f"\n  reframe {aspect} {doc['window']['width']}x{doc['window']['height']}: face inside the window in "
                  f"{inside}/{n_faces} detected faces over {len(ws)} samples; fastest move {worst:.0f} px/s (limit {v:.0f}); "
                  f"switches {[(s['t'], s['to']) for s in doc['switches']]}", end="")
            self.assertEqual(len(ws), len(det["frames"]))
            self.assertGreaterEqual(n_faces, len(ws) - 1)
            self.assertEqual(inside, n_faces, aspect)
            self.assertTrue(speed_ok(doc), aspect)
            self.assertLessEqual(worst, v + 2 / 0.2)
            self.assertEqual([s["shot"] for s in doc["switches"]], [0, 1])
            self.assertEqual(ws[0]["shot"], 0)
            self.assertEqual(ws[-1]["shot"], 1)


@unittest.skipUnless(HAVE_TESS, "needs tesseract with jpn, jpn_vert and eng traineddata")
@unittest.skipUnless(HAVE_PIL and JP_FONT, "needs Pillow and a Japanese font (DETECTION_TEST_FONT) to render the fixture")
class RealPiiTests(TmpCase):
    # Slides rendered with Pillow (1280x720, 40 px), read by Tesseract, checked by `pii`. Ground truth from the
    # construction: the line index of each personal datum.
    WITH = ["Case presentation", "Patient ID: MRN 00482913", "DOB: 1956-03-14", "Chief complaint: chest pain"]
    NONE = ["Results", "Sensitivity 92% (n = 1203)", "Published 2024-05-01", "Follow-up: 12 months"]
    JA = ["症例提示", "患者ID: 0012345", "生年月日: 昭和45年3月14日", "主訴: 胸痛"]

    def slide(self, lines, name):
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(JP_FONT, 40)
        im = Image.new("RGB", (1280, 720), "white")
        dr = ImageDraw.Draw(im)
        for i, line in enumerate(lines):
            dr.text((80, 80 + i * 100), line, font=font, fill="black")
        path = os.path.join(self.tmp, name)
        im.save(path)
        return path

    def run_pii(self, lines, name, lang, locales):
        text = detect("text", self.slide(lines, name), lang=lang, tessdata_dir=os.environ.get("DETECTION_TESSDATA_DIR"))
        doc = make_pii(text, locales=locales)
        self.assertTrue(validate_pii(doc).ok, validate_pii(doc).errors)
        print(f"\n  pii {name} ({'+'.join(locales)}): OCR {[d['label'] for d in text['frames'][0]['detections']]} -> "
              f"{[(r['category'], r['excerpt']) for r in doc['regions']]}", end="")
        return doc

    def assert_lines(self, doc, want):
        got = [(r["category"], int(round((r["box"][1] + r["box"][3] / 2 - 80) / 100.0))) for r in doc["regions"]]
        self.assertEqual(got, want)
        for r in doc["regions"]:
            x = r["redact"]
            self.assertTrue(x["x"] <= r["box"][0] and x["y"] <= r["box"][1] and x["x"] + x["width"] >= r["box"][0] + r["box"][2]
                            and x["y"] + x["height"] >= r["box"][1] + r["box"][3])
            self.assertEqual(r["decision"], "CONFIRM")

    def test_slide_with_patient_id_and_birth_date_and_slide_with_none(self):
        self.assert_lines(self.run_pii(self.WITH, "with.png", "eng", ["intl"]), [("patient_id", 1), ("date_of_birth", 2)])
        self.assert_lines(self.run_pii(self.NONE, "none.png", "eng", ["intl"]), [])

    def test_japanese_profile(self):
        self.assert_lines(self.run_pii(self.JA, "ja.png", "jpn+eng", ["ja-JP"]), [("patient_id", 1), ("date_of_birth", 2)])



# Two cameras, 640x360 at 25 fps, 12 s. Camera 0: a wide shot of two faces (A on the left; B, the same portrait
# mirrored, on the right); camera 1: a close-up of B. A speaks 0-3 s and 6-9 s, B 3-6 s and 9-12 s: while a face
# speaks, a dark mouth opening is drawn over its mouth on every other block of three frames (lavfi drawbox), in every
# camera that shows it. Ground truth from the construction: switches at 3, 6 and 9 s, B on its close-up.
MOUTH_IN_CROP = (61, 108, 24, 10)  # x, y, w, h of the mouth in the astronaut face crop img[30:210, 150:300]


def render_talk_camera(path, faces, crops):
    def on(spans):
        return "+".join(f"between(t,{a},{b - 0.001})" for a, b in spans)
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=25:d=12"]
    graph, last = [], "[0:v]"
    for i, (name, s, x, y, spans) in enumerate(faces):
        argv += ["-loop", "1", "-i", crops[name]]
        mx = 150 - MOUTH_IN_CROP[0] - MOUTH_IN_CROP[2] if name == "B" else MOUTH_IN_CROP[0]
        bx, by = int(x + mx * s), int(y + MOUTH_IN_CROP[1] * s)
        bw, bh = int(MOUTH_IN_CROP[2] * s), int(MOUTH_IN_CROP[3] * s)
        graph.append(f"[{i + 1}:v]scale=iw*{s}:ih*{s}[f{i}];{last}[f{i}]overlay={x}:{y}:shortest=1[o{i}];"
                     f"[o{i}]drawbox=x={bx}:y={by}:w={bw}:h={bh}:color=black:t=fill:"
                     f"enable='({on(spans)})*lt(mod(n\\,6)\\,3)'[d{i}]")
        last = f"[d{i}]"
    graph.append(f"{last}format=yuv420p[out]")
    subprocess.run(argv + ["-filter_complex", ";".join(graph), "-map", "[out]", "-r", "25", path], check=True)


@unittest.skipUnless(HAVE_CV2 and YUNET_FILE and REAL_VIDEO,
                     "needs YuNet in DETECTION_MODELS_DIR, opencv, ffmpeg and an ffmpeg-skill checkout")
class RealSpeakersTests(TmpCase):
    def test_two_cameras_alternating_speakers(self):
        import cv2
        img = cv2.imread(str(FIXTURES / "astronaut.jpg"))
        crop = img[30:210, 150:300]
        crops = {"A": os.path.join(self.tmp, "a.png"), "B": os.path.join(self.tmp, "b.png")}
        cv2.imwrite(crops["A"], crop)
        cv2.imwrite(crops["B"], cv2.flip(crop, 1))
        videos = [os.path.join(self.tmp, "cam0.mp4"), os.path.join(self.tmp, "cam1.mp4")]
        render_talk_camera(videos[0], [("A", 1.2, 60, 70, TURNS_A), ("B", 1.2, 400, 70, TURNS_B)], crops)
        render_talk_camera(videos[1], [("B", 1.5, 200, 45, TURNS_B)], crops)
        cams = []
        for v in videos:
            det = detect("face", v, every=0.12, models_dir=MODELS_DIR, model=YUNET_FILE, ffmpeg_skill=str(FFMPEG_SKILL_DIR),
                         attributes=True)
            self.assertTrue(validate_detections(det).ok, validate_detections(det).errors)
            a = det["frames"][0]["detections"][0]["attributes"]
            self.assertGreater(a["frontal"], 0.9)
            self.assertGreater(a["sharpness"], 0)
            cams.append(make_tracks(det))
        self.assertEqual([c["counts"]["tracks"] for c in cams], [2, 1])
        truth = [3.0, 6.0, 9.0]
        turns = turns_doc({"SPEAKER_00": TURNS_A, "SPEAKER_01": TURNS_B})
        for label, doc in (("mouth motion", make_speakers(cams)), ("turns", make_speakers(cams, turns=turns))):
            rep = validate_speakers(doc)
            self.assertTrue(rep.ok, rep.errors)
            starts = [r["start"] for r in doc["ranges"][1:]]
            err = max(abs(a - b) for a, b in zip(starts, truth)) if len(starts) == len(truth) else None
            print(f"\n  speakers ({label}): switch {doc['switch']}; worst switch error "
                  f"{'n/a' if err is None else f'{err:.2f}s'} (limit 0.5 s)", end="")
            self.assertEqual([r["camera"] for r in doc["ranges"]], [0, 1, 0, 1], doc["switch"])
            self.assertLessEqual(err, 0.5)

if __name__ == "__main__":
    unittest.main()
