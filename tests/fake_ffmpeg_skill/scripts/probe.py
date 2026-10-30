import json
import sys

from _fake import fail, load, log_call

log_call(sys.argv)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
try:
    v = load(args[0])
except (OSError, ValueError):
    fail("cannot probe")
video = None if v.get("audio_only") else {"width": v["width"], "height": v["height"], "fps": v.get("fps", 25.0),
                                          "rotation": v.get("rotation", 0), "duration": v["duration"]}
print(json.dumps({"file": args[0], "duration": v["duration"], "video": video, "audio": None}))
