import json

print(json.dumps({"ok": True, "ffmpeg": "fake", "tools": {"probe": {"usable": "yes"}, "look": {"usable": "yes"},
                                                         "scenes": {"usable": "yes"}}}))
