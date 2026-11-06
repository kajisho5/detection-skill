"""Shot-boundary arithmetic in plain Python, shared by the shot engines (ADR-003) and tested without any engine.

Frame numbers are 0-based indices into the video's frames. A boundary's `frame` is the first frame of the shot that
follows it; a gradual boundary also carries its transition span (`start_frame`..`end_frame`, inclusive, the frames
that blend the two shots) and its `frame` is the middle of that span.

- `blend_runs`: the gradual-transition heuristic used with PySceneDetect. A frame i "passes" when, h frames either
  side, the two frames differ (mean absolute BGR difference `far` >= far_min) and frame i is close to their average
  (`ratio` = mean |F_i - (F_{i-h} + F_{i+h}) / 2| / far <= ratio_max): a cross-dissolve or a fade is a linear blend,
  camera or subject motion is not (the average of two displaced frames is a double image). Consecutive passing
  frames (gaps of one frame allowed) form a run; a run of at least `min_core` frames is a gradual transition over
  [first - h, last + h].
- `merge_boundaries`: detector cuts plus gradual spans -> one ordered list. A cut inside a gradual span (or within
  `absorb` frames of it) is the same transition seen by a cut detector (AdaptiveDetector fires where a dissolve
  starts or ends) and is absorbed; boundaries closer than `min_shot_frames` keep the higher score.
- `transnet_windows` / `transnet_boundaries`: TransNetV2's inference recipe (100-frame windows, step 50, the
  first and last frame repeated 25 times as padding, the middle 50 predictions kept) and the turn of its per-frame
  probabilities into boundaries.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BLEND_DEFAULTS = {"far_min": 20.0, "ratio_max": 0.12, "min_core": 3, "absorb": 2}


def blend_h(fps: float) -> int:
    """Half-window of the blend test: about 0.16 s, at least 2 frames (4 at 25 fps)."""
    return max(2, int(round(0.16 * float(fps))))


def blend_runs(passing: Iterable[Tuple[int, float, float]], h: int, *, min_core: int = 3,
               frame_count: Optional[int] = None) -> List[Dict[str, Any]]:
    """passing: (frame, far, ratio) for every frame that passed the blend test, any order. Returns gradual
    transitions [{frame, kind: 'gradual', start_frame, end_frame, score, metric}] in time order."""
    pts = sorted(passing)
    runs: List[List[Tuple[int, float, float]]] = []
    for p in pts:
        if runs and p[0] - runs[-1][-1][0] <= 2:
            runs[-1].append(p)
        else:
            runs.append([p])
    out: List[Dict[str, Any]] = []
    last = (frame_count - 1) if frame_count else None
    for run in runs:
        if len(run) < min_core:
            continue
        a, b = run[0][0] - h, run[-1][0] + h
        a = max(0, a)
        if last is not None:
            b = min(last, b)
        if a <= 0:
            continue  # a fade-in at the very start is part of the first shot, not a boundary
        best = min(r for _f, _far, r in run)
        far = max(f for _i, f, _r in run)
        out.append({"frame": int(round((a + b) / 2.0)), "kind": "gradual", "start_frame": int(a), "end_frame": int(b),
                    "score": round(max(0.0, min(1.0, 1.0 - best)), 4),
                    "metric": {"blend_ratio_min": round(best, 4), "blend_far_max": round(far, 2), "core_frames": len(run),
                               "half_window": h}})
    return out


def merge_boundaries(cuts: Sequence[Dict[str, Any]], gradual: Sequence[Dict[str, Any]], *, min_shot_frames: int,
                     absorb: int = 2) -> Tuple[List[Dict[str, Any]], int]:
    """Returns (boundaries in time order, number of cuts absorbed or dropped)."""
    dropped = 0
    kept_cuts: List[Dict[str, Any]] = []
    for c in sorted(cuts, key=lambda c: c["frame"]):
        if any(g["start_frame"] - absorb <= c["frame"] <= g["end_frame"] + absorb + 1 for g in gradual):
            dropped += 1
            continue
        if kept_cuts and c["frame"] - kept_cuts[-1]["frame"] <= 1:
            dropped += 1  # the same cut from two detectors, one frame apart: keep the first
            if c["score"] > kept_cuts[-1]["score"]:
                kept_cuts[-1] = dict(kept_cuts[-1], score=c["score"])
            continue
        kept_cuts.append(dict(c))
    merged = sorted(kept_cuts + [dict(g) for g in gradual], key=lambda b: b["frame"])
    out: List[Dict[str, Any]] = []
    for b in merged:
        if b["frame"] <= 0:
            dropped += 1
            continue
        if out and b["frame"] - out[-1]["frame"] < min_shot_frames:
            dropped += 1
            if b["score"] > out[-1]["score"]:
                out[-1] = b
            continue
        out.append(b)
    return out, dropped


# ---- TransNetV2 ------------------------------------------------------------------------------------
TN_WINDOW, TN_STEP, TN_PAD = 100, 50, 25


def transnet_windows(n: int) -> List[List[int]]:
    """Frame indices of each 100-frame input window for n frames (TransNetV2's own padding: the first frame 25 times
    before, the last frame after, up to a multiple of 50). Each window's predictions [25:75] belong to frames
    window_no*50 .. window_no*50+49."""
    if n <= 0:
        return []
    tail = TN_PAD + TN_STEP - (n % TN_STEP if n % TN_STEP else TN_STEP)
    padded = [0] * TN_PAD + list(range(n)) + [n - 1] * tail
    return [padded[s:s + TN_WINDOW] for s in range(0, len(padded) - TN_WINDOW + 1, TN_STEP)]


def sigmoid_if_logits(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    if any(v < 0.0 or v > 1.0 for v in vals):
        return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v)))) for v in vals]
    return vals


def transnet_boundaries(single: Sequence[float], many: Optional[Sequence[float]], threshold: float,
                        gradual_frames: int = 3) -> List[Dict[str, Any]]:
    """Per-frame probabilities -> boundaries. A run of frames above `threshold` on the single-frame head is one
    transition; the frame after its peak is the first frame of the next shot. The transition is gradual when the
    run, or the all-frames head's run around it, is at least `gradual_frames` long."""
    n = len(single)
    out: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if single[i] <= threshold:
            i += 1
            continue
        j = i
        while j + 1 < n and single[j + 1] > threshold:
            j += 1
        peak = max(range(i, j + 1), key=lambda k: single[k])
        a, b = i, j
        if many is not None:
            while a - 1 >= 0 and many[a - 1] > threshold:
                a -= 1
            while b + 1 < len(many) and many[b + 1] > threshold:
                b += 1
        gradual = (b - a + 1) >= gradual_frames
        frame = int(round((a + b) / 2.0)) if gradual else peak + 1
        if 0 < frame < n:
            b_doc: Dict[str, Any] = {"frame": frame, "kind": "gradual" if gradual else "cut", "score": round(float(single[peak]), 4),
                                     "metric": {"probability": round(float(single[peak]), 4), "run_frames": j - i + 1}}
            if gradual:
                b_doc.update(start_frame=a, end_frame=b)
            out.append(b_doc)
        i = j + 1
    return out
