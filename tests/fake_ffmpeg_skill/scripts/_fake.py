"""Shared by the fake ffmpeg-skill tools. A "video" here is a JSON file:
{"duration": 10.0, "width": 640, "height": 360, "fps": 25, "scenes": [[0, 4], [4, 10]], "rotation": 0}."""
import json
import os
import struct
import sys
import zlib


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_png(path, width, height):
    row = b"\x00" + b"\x28\x28\x28" * width
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(row * height, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def fail(message, code=1):
    print(json.dumps({"status": "failed", "exit_code": code, "error": {"message": message}}))
    sys.exit(code)


def log_call(argv):
    log = os.environ.get("FAKE_FFMPEG_SKILL_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
