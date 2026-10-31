"""detection-skill tests.

Unit tests use a test-only engine (tests/fake_engine.py) injected as an object, and a fake ffmpeg-skill
(tests/fake_ffmpeg_skill: probe / look / scenes with the real tools' JSON shapes, frames written as PNG with the
standard library), so they need no engine, no model file and no ffmpeg.

Tests that need more skip with the reason, and run where it is available:
- ffmpeg on PATH + a real ffmpeg-skill checkout (FFMPEG_SKILL_DIR or ../ffmpeg-skill): frames through `look`/`scenes`.
- the AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS): registry conformance.
- real engines: faces need opencv + YuNet in DETECTION_MODELS_DIR; objects need onnxruntime + opencv + a yolox_*.onnx
  in DETECTION_MODELS_DIR; text needs tesseract with jpn/jpn_vert/eng (DETECTION_TESSDATA_DIR or tesseract's own),
  Pillow and a Japanese font to render the fixture.

Fixture images (tests/fixtures): astronaut.jpg (NASA portrait of Eileen Collins, public domain), chelsea.jpg (CC0),
coffee.jpg (CC0), all as shipped in scikit-image's data module. Hand-annotated boxes are below (ASTRONAUT_FACE etc.).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
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
from fake_engine import MODEL_SHA, ScriptedEngine, png_size, write_png  # noqa: E402

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
                         {"vision.detect_faces", "vision.detect_text", "vision.detect_objects", "vision.track_boxes"})
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


if __name__ == "__main__":
    unittest.main()
