import json
import sys

from _fake import load, log_call

log_call(sys.argv)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
v = load(args[0])
scenes = v.get("scenes") or [[0, v["duration"]]]
print(json.dumps({"status": "completed", "duration": v["duration"], "scene_count": len(scenes),
                  "scenes": [{"index": i, "start": a, "end": b, "duration": b - a} for i, (a, b) in enumerate(scenes)]}))
