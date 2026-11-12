"""Real-model tests: the optional engines of ADR-008 (PP-OCRv6 medium, RF-DETR Small) and YuNet on their real model
files, on the fixtures in tests/fixtures and on slides rendered here.

Each class skips with the reason when something is missing (a model file in DETECTION_MODELS_DIR, a Python package,
Pillow or a font). With DETECTION_REQUIRE_REAL_MODELS=1 a missing piece fails the class instead of skipping it: the
manual workflow .github/workflows/heavy-tests.yml sets it, after fetching the PP-OCRv6 files at the pinned revisions
(sha256 checked), exporting RF-DETR Small with the documented command and fetching YuNet. The PP-OCRv6 and RF-DETR
classes skip everywhere else; their skip reason starts with SKIP_PREFIX, the one skip tests.yml's integration job
lets through. The YuNet class runs wherever YuNet is (the integration job too) and has no such prefix.

Measured numbers are printed and, when DETECTION_REAL_MODELS_SUMMARY names a file, appended to it as Markdown (the
workflow's job summary).

    DETECTION_MODELS_DIR=... python -m unittest -v tests/test_real_models.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_detection as td  # noqa: E402  (a module, so its test classes are not collected twice)
from detection_skill.detections import validate_detections  # noqa: E402
from detection_skill.engines.boxes import iou  # noqa: E402
from detection_skill.engines.yolox_engine import COCO_CLASSES  # noqa: E402
from detection_skill.metrics import cer  # noqa: E402
from detection_skill.models import KNOWN_MODELS, ppocrv6_names  # noqa: E402
from detection_skill.service import detect  # noqa: E402

REQUIRE = os.environ.get("DETECTION_REQUIRE_REAL_MODELS") == "1"
SKIP_PREFIX = "real model, heavy-tests.yml only: "
SUMMARY = os.environ.get("DETECTION_REAL_MODELS_SUMMARY")
MODELS_DIR = td.MODELS_DIR
LATIN_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]
EN_FONT = next((f for f in [os.environ.get("DETECTION_TEST_FONT")] + LATIN_FONTS + td.FONT_CANDIDATES
                if f and os.path.isfile(f)), None)


def _have(name):
    return bool(MODELS_DIR and os.path.isfile(os.path.join(MODELS_DIR, name)))


PPOCRV6_FILES = list(ppocrv6_names("medium").values())
HAVE_PPOCRV6 = all(_have(n) for n in PPOCRV6_FILES)
HAVE_RFDETR = _have("rfdetr-small.onnx")
YUNET_FILE = td.YUNET_FILE


def real_model(ok, why, heavy_only=True):
    """Class decorator: run when `ok`; otherwise fail every test (DETECTION_REQUIRE_REAL_MODELS=1) or skip with `why`."""
    def wrap(cls):
        if ok:
            return cls
        if REQUIRE:
            def setUpClass(klass):
                raise AssertionError(f"DETECTION_REQUIRE_REAL_MODELS=1, but {why}")
            cls.setUpClass = classmethod(setUpClass)
            return cls
        return unittest.skip((SKIP_PREFIX if heavy_only else "") + why)(cls)
    return wrap


MEASURED = []


def report(line):
    """A measured number: printed after the last test (so each test's `... ok` stays on its own line) and appended to
    DETECTION_REAL_MODELS_SUMMARY."""
    MEASURED.append(line)
    if SUMMARY:
        with open(SUMMARY, "a", encoding="utf-8") as fh:
            fh.write(f"- {line}\n")


def well_formed(test, doc, kind):
    """The document validates; every box lies inside its frame with a positive size; scores are in 0..1."""
    rep = validate_detections(doc)
    test.assertTrue(rep.ok, rep.errors)
    w, h = doc["source"]["width"], doc["source"]["height"]
    for fr in doc["frames"]:
        for d in fr["detections"]:
            x, y, bw, bh = d["box"]
            test.assertTrue(0 <= x and 0 <= y and bw > 0 and bh > 0 and x + bw <= w + 1e-6 and y + bh <= h + 1e-6, d["box"])
            test.assertTrue(0.0 <= d["score"] <= 1.0, d["score"])
            test.assertEqual((d["engine"], d["model_sha256"]), (doc["engine"]["id"], doc["engine"]["model_sha256"]))
            if kind == "object":
                test.assertIn(d["label"], COCO_CLASSES)


@real_model(HAVE_PPOCRV6 and td.HAVE_CV2 and td.HAVE_ORT and td.HAVE_PIL and EN_FONT and td.JP_FONT,
            "needs PP-OCRv6_medium_det.onnx, PP-OCRv6_medium_rec.onnx and PP-OCRv6_medium_rec.yml in DETECTION_MODELS_DIR, "
            "onnxruntime, opencv-python-headless, Pillow, a Latin font and a Japanese font (DETECTION_TEST_FONT)")
class RealPPOCRv6Tests(td.TmpCase):
    CER_JA = float(os.environ.get("DETECTION_CER_JA", "0.10"))
    CER_EN = float(os.environ.get("DETECTION_CER_EN", "0.05"))

    def render(self, lines, font_path, size=(1280, 720), font_px=44):
        """A white slide with one line of black text per entry; returns the path and each line's ink box [x, y, w, h]."""
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, font_px)
        im = Image.new("RGB", size, "white")
        dr = ImageDraw.Draw(im)
        boxes = []
        for i, line in enumerate(lines):
            xy = (80, 80 + i * int(font_px * 2.4))
            dr.text(xy, line, font=font, fill="black")
            x0, y0, x1, y1 = dr.textbbox(xy, line, font=font)
            boxes.append([x0, y0, x1 - x0, y1 - y0])
        path = os.path.join(self.tmp, f"slide{len(os.listdir(self.tmp))}.png")
        im.save(path)
        return path, boxes

    def read(self, lines, font_path, name, limit):
        path, truth = self.render(lines, font_path)
        doc = detect("text", path, engine_id="ppocrv6", models_dir=MODELS_DIR)
        well_formed(self, doc, "text")
        dets = doc["frames"][0]["detections"]
        text = "\n".join(d["label"] for d in dets)
        score = cer("".join(lines), text)
        ious = [round(max((iou(t, d["box"]) for d in dets), default=0.0), 2) for t in truth]
        report(f"PP-OCRv6 medium, {name} slide (1280x720, {len(lines)} lines, 44 px): CER {score:.4f} over "
               f"{len(''.join(lines))} chars, {len(dets)} lines found, line-box IoU {ious}, scores "
               f"{[round(d['score'], 3) for d in dets]}, {doc['engine']['engine_seconds']} s; read: {text!r}")
        self.assertEqual(len(dets), len(lines), text)
        self.assertLessEqual(score, limit, f"{name}: {text!r}")
        self.assertGreaterEqual(min(ious), 0.5, (ious, truth, [d["box"] for d in dets]))
        for d in dets:
            self.assertEqual((d["lang"], d["engine"]), ("multi", "ppocrv6"))
            self.assertGreaterEqual(d["words"], 1)
        return doc

    def test_english_slide(self):
        doc = self.read(td.RealTextTests.EN, EN_FONT, "English", self.CER_EN)
        eng = doc["engine"]
        self.assertEqual((eng["id"], eng["model"], eng["model_license"]),
                         ("ppocrv6", "PP-OCRv6_medium det + rec (official ONNX exports)", "Apache-2.0"))
        self.assertEqual([(f["role"], f["file"], f["known"]) for f in eng["files"]],
                         [(r, n, True) for r, n in zip(("det", "rec", "dict"), PPOCRV6_FILES)])
        self.assertTrue(all(KNOWN_MODELS[f["sha256"]]["file"] == f["file"] for f in eng["files"]))
        self.assertEqual(eng["details"]["dictionary_size"], 18708)        # ADR-008: 18708 + blank + space = 18710
        self.assertEqual(doc["warnings"], [])
        self.assertEqual(doc["parameters"], {"min_score": 0.3, "lang": "multi", "vertical": False, "level": "line"})

    def test_japanese_slide(self):
        self.read(td.RealTextTests.JA, td.JP_FONT, "Japanese", self.CER_JA)

    def test_blank_frame_has_no_text(self):
        path, _ = self.render([], EN_FONT)
        doc = detect("text", path, engine_id="ppocrv6", models_dir=MODELS_DIR)
        well_formed(self, doc, "text")
        self.assertEqual(doc["counts"]["detections"], 0)


@real_model(HAVE_RFDETR and td.HAVE_CV2 and td.HAVE_ORT,
            "needs rfdetr-small.onnx (the documented export) in DETECTION_MODELS_DIR, onnxruntime and opencv-python-headless")
class RealRfdetrTests(td.TmpCase):
    TRUTH = {"astronaut.jpg": ("person", td.ASTRONAUT_PERSON), "chelsea.jpg": ("cat", td.CHELSEA_CAT),
             "coffee.jpg": ("cup", td.COFFEE_CUP)}

    def run_rfdetr(self, path, **kw):
        doc = detect("object", path, engine_id="rfdetr", models_dir=MODELS_DIR, **kw)
        well_formed(self, doc, "object")
        return doc

    def test_fixture_recall(self):
        found = []
        for name, (label, box) in self.TRUTH.items():
            doc = self.run_rfdetr(str(td.FIXTURES / name))
            dets = doc["frames"][0]["detections"]
            best = max((iou(box, d["box"]) for d in dets if d["label"] == label), default=0.0)
            top = max((d["score"] for d in dets if d["label"] == label), default=0.0)
            report(f"RF-DETR Small, {name}: {label} best IoU {best:.2f}, best score {top:.3f}, {doc['engine']['engine_seconds']} s; "
                   f"all: {[(d['label'], round(d['score'], 3)) for d in dets]}")
            found.append(best >= 0.5)
            self.assertEqual([d["score"] for d in dets], sorted((d["score"] for d in dets), reverse=True))
        eng = doc["engine"]
        self.assertEqual((eng["id"], eng["model"], eng["model_license"]),
                         ("rfdetr", "RF-DETR Small (COCO, ONNX export)", "Apache-2.0"))
        self.assertEqual((eng["details"]["input_size"], eng["details"]["class_slots"]), ([512, 512], 91))
        self.assertEqual(found, [True, True, True])

    def test_boxes_in_a_wide_frame(self):
        # two fixtures pasted at known places on a 1280x720 canvas: the export's square input is a non-uniform resize,
        # so a box that comes back at the right place checks the scaling of x and y separately
        import cv2
        import numpy as np
        cat, cup = cv2.imread(str(td.FIXTURES / "chelsea.jpg")), cv2.imread(str(td.FIXTURES / "coffee.jpg"))
        canvas = td.compose(cv2, np, 1280, 720, [(cup, 0.8, 60, 60), (cat, 1.2, 640, 300)])
        path = os.path.join(self.tmp, "wide.png")
        cv2.imwrite(path, canvas)
        doc = self.run_rfdetr(path)
        dets = doc["frames"][0]["detections"]
        want = {"cat": td.tbox(td.CHELSEA_CAT, 1.2, 640, 300), "cup": td.tbox(td.COFFEE_CUP, 0.8, 60, 60)}
        got = {k: max((iou(v, d["box"]) for d in dets if d["label"] == k), default=0.0) for k, v in want.items()}
        report(f"RF-DETR Small, both fixtures on a 1280x720 canvas: best IoU cat {got['cat']:.2f}, cup {got['cup']:.2f}; "
               f"all: {[(d['label'], round(d['score'], 3), [round(v) for v in d['box']]) for d in dets]}")
        self.assertGreaterEqual(got["cat"], 0.5)
        self.assertGreaterEqual(got["cup"], 0.5)

    def test_classes_filter(self):
        doc = self.run_rfdetr(str(td.FIXTURES / "coffee.jpg"), classes=["cup"])
        labels = [d["label"] for d in doc["frames"][0]["detections"]]
        self.assertTrue(labels and set(labels) == {"cup"}, labels)
        self.assertEqual(doc["parameters"]["classes"], ["cup"])

    def test_blank_frame_has_no_object(self):
        import cv2
        import numpy as np
        path = os.path.join(self.tmp, "gray.png")
        cv2.imwrite(path, np.full((360, 640, 3), 128, np.uint8))
        doc = self.run_rfdetr(path)
        self.assertEqual(doc["counts"]["detections"], 0, doc["frames"][0]["detections"])


@real_model(td.HAVE_CV2 and YUNET_FILE, "needs face_detection_yunet_*.onnx in DETECTION_MODELS_DIR and opencv-python-headless",
            heavy_only=False)
class RealYunetTests(td.TmpCase):
    def test_fixture_face_and_landmarks(self):
        doc = detect("face", str(td.FIXTURES / "astronaut.jpg"), models_dir=MODELS_DIR, model=YUNET_FILE)
        well_formed(self, doc, "face")
        dets = doc["frames"][0]["detections"]
        best = max(dets, key=lambda d: iou(td.ASTRONAUT_FACE, d["box"]))
        v = iou(td.ASTRONAUT_FACE, best["box"])
        report(f"YuNet ({YUNET_FILE}), astronaut.jpg: {len(dets)} face(s), IoU {v:.2f}, score {best['score']:.3f}, "
               f"{doc['engine']['engine_seconds']} s")
        self.assertEqual(len(dets), 1)
        self.assertGreaterEqual(v, 0.5)
        self.assertIsNone(best["label"])
        # five landmarks inside the box: the subject's right eye (image left), left eye, nose tip, right and left mouth
        # corners; eyes above the nose, the nose above the mouth
        (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = best["landmarks"]
        x, y, w, h = best["box"]
        for px, py in best["landmarks"]:
            self.assertTrue(x <= px <= x + w and y <= py <= y + h, (px, py, best["box"]))
        self.assertLess(rex, lex)
        self.assertLess(rmx, lmx)
        self.assertLess(max(rey, ley), ny)
        self.assertLess(ny, min(rmy, lmy))
        f = doc["engine"]["files"][0]
        if f["file"] == "face_detection_yunet_2026may.onnx":
            self.assertTrue(f["known"])
            self.assertEqual(doc["engine"]["model"], "YuNet 2026may (OpenCV Zoo)")
            self.assertEqual(doc["warnings"], [])


def tearDownModule():
    if MEASURED:
        print()
    for line in MEASURED:
        print(f"  measured: {line}")


if __name__ == "__main__":
    unittest.main()
