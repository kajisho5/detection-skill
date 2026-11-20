"""The ecosystem timebase spec's conformance vectors (velisa/timebase@1) against detection_skill.frames (ADR-012).

tests/fixtures/timebase/timebase-vectors.json is a byte-for-byte copy of Velisa's docs/specs/timebase-vectors.json at
Velisa commit VELISA_COMMIT (its sha256 is checked). This Skill reads rates (exact_rate; None is INVALID_RATE),
compares them, counts the whole frames of a duration (frame_count: frames_at `down`) and seeks to a frame
(seek_time: seek_seconds to the millisecond). Every vector of those must pass; the others are counted."""
import hashlib
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection_skill.frames import exact_rate, frame_count, seek_time  # noqa: E402

VECTORS = ROOT / "tests" / "fixtures" / "timebase" / "timebase-vectors.json"
VELISA_COMMIT = "b7df1d82586f0b7c00fed4074aecac75cbadf893"
VECTORS_SHA256 = "fecd9730470fd79fb0b8dce6a55ef1ac8fd743ea044a1f1541474ebd249e5aa1"
OFFERED = 43


class _Refused(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _rate(value):
    r = exact_rate(value)
    if r is None:
        raise _Refused("INVALID_RATE")
    return r


def _text(v):
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    return v


FUNCTIONS = {
    "parse_rate": lambda value: _rate(value),
    "same_rate": lambda a, b: _rate(a) == _rate(b),
    "frames_at": lambda seconds, rate, mode: frame_count({"duration": float(seconds), "rate_exact": _rate(rate)}),
    "seek_seconds": lambda frame, rate, places: f"{seek_time(frame, _rate(rate)):.3f}",
}


def _offered(vec):
    if vec["fn"] not in FUNCTIONS or vec.get("optional"):
        return False
    if vec["fn"] == "frames_at":
        return vec["args"].get("mode") == "down"
    if vec["fn"] == "seek_seconds":
        return vec["args"].get("places") == 3
    return True


class TimebaseVectorTests(unittest.TestCase):
    def test_the_copy_is_velisas_file(self):
        raw = VECTORS.read_bytes()
        self.assertEqual(json.loads(raw)["spec"], "velisa/timebase@1")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), VECTORS_SHA256, f"re-copy the vectors from Velisa {VELISA_COMMIT}")

    def test_every_offered_vector_passes(self):
        doc = json.loads(VECTORS.read_text(encoding="utf-8"))
        vecs = [v for v in doc["vectors"] if _offered(v)]
        for vec in vecs:
            with self.subTest(vec["id"]):
                try:
                    got = _text(FUNCTIONS[vec["fn"]](**vec["args"]))
                except _Refused as e:
                    self.assertEqual(e.code, vec.get("error"))
                    continue
                self.assertNotIn("error", vec, f"gave {got!r}")
                self.assertEqual(got, vec["expect"])
        self.assertEqual(len(vecs), OFFERED)
        self.assertEqual(len(doc["vectors"]) - len(vecs), 134 - OFFERED)

    def test_probe_rate_is_exact(self):
        self.assertEqual(exact_rate("30000/1001"), Fraction(30000, 1001))
        self.assertEqual(exact_rate(29.97), Fraction(30000, 1001))   # probe's 3-decimal fps, read by TB-2
        self.assertEqual(frame_count({"duration": 3.336666, "rate_exact": Fraction(30000, 1001)}), 100)  # 1 us short


if __name__ == "__main__":
    unittest.main()
