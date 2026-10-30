import argparse
import json
import os
import sys

from _fake import fail, load, log_call, write_png

log_call(sys.argv)
ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("-o", "--output")
ap.add_argument("--at", action="append")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--no-timecode", action="store_true")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
v = load(a.input)
w, h = v["width"], v["height"]
if v.get("rotation") in (90, -90, 270, -270):
    w, h = h, w
out_w = a.width
out_h = int(round(h * out_w / w / 2.0)) * 2  # scale=W:-2, like the real look
outdir = os.path.dirname(a.output)
if not os.path.isdir(outdir):
    fail("output directory does not exist")
outs = []
for t in a.at:
    sec = float(t)
    if sec > v["duration"]:
        fail("--at beyond the duration")
    p = os.path.join(outdir, f"{os.path.basename(a.output)}_{sec:.3f}s.png")
    write_png(p, out_w, out_h)
    outs.append(p)
print(json.dumps({"status": "completed", "output": None, "outputs": outs, "commands": []}))
