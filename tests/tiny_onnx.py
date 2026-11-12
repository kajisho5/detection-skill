"""Tiny synthetic ONNX models for the engine tests, written with the standard library only (a minimal protobuf
encoder for the few ModelProto fields used), so no `onnx` package and no real model file is needed. Each model has
the input/output layout of the real export it stands in for and a few standard operators, so a test can run the real
engine (session creation, input names, pre- and post-processing, the worker) and know the answer exactly. They
measure nothing about the real models.

- ppocr_det(): x [N, 3, H, W] -> Sigmoid(4 x mean over channels) [N, 1, H, W]: a bright region on a dark frame reads
  as text (probability near 1), dark as background.
- ppocr_rec(classes): x [N, 3, 48, W] -> Softmax over C classes [N, W / 8, C]: each 8 px column block with a bright pixel
  reads as class 1, dark or padding as blank, so a bright line decodes to the dictionary's first character.
- rfdetr(dets, logits, size): input [1, 3, size, size] -> dets [1, Q, 4], labels [1, Q, 91]: the given constants.
"""
from __future__ import annotations

import struct
from typing import List, Sequence, Union

FLOAT, INT64 = 1, 7
Dim = Union[int, str]


def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint(field << 3 | wire)


def _len(field: int, data: bytes) -> bytes:
    return _key(field, 2) + _varint(len(data)) + data


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _str(field: int, s: str) -> bytes:
    return _len(field, s.encode("utf-8"))


def attr_int(name: str, v: int) -> bytes:
    return _str(1, name) + _int(20, 2) + _int(3, v)                      # AttributeProto: name, type INT, i


def attr_ints(name: str, vs: Sequence[int]) -> bytes:
    return _str(1, name) + _int(20, 7) + b"".join(_int(8, v) for v in vs)   # type INTS, ints


def node(op: str, inputs: Sequence[str], outputs: Sequence[str], attrs: Sequence[bytes] = ()) -> bytes:
    return (b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs) + _str(4, op)
            + b"".join(_len(5, a) for a in attrs))


def value_info(name: str, elem: int, dims: Sequence[Dim]) -> bytes:
    shape = b"".join(_len(1, _int(1, d) if isinstance(d, int) else _str(2, d)) for d in dims)
    tensor_type = _int(1, elem) + _len(2, shape)
    return _str(1, name) + _len(2, _len(1, tensor_type))


def tensor(name: str, dims: Sequence[int], values: Sequence[float], elem: int = FLOAT) -> bytes:
    raw = struct.pack(f"<{len(values)}{'f' if elem == FLOAT else 'q'}", *values)
    return b"".join(_int(1, d) for d in dims) + _int(2, elem) + _str(8, name) + _len(9, raw)


def model(nodes: List[bytes], inputs: List[bytes], outputs: List[bytes], initializers: List[bytes] = (),
          opset: int = 13) -> bytes:
    graph = (b"".join(_len(1, n) for n in nodes) + _str(2, "tiny") + b"".join(_len(5, t) for t in initializers)
             + b"".join(_len(11, i) for i in inputs) + b"".join(_len(12, o) for o in outputs))
    return _int(1, 7) + _str(2, "detection-skill-tests") + _len(7, graph) + _len(8, _int(2, opset))


def ppocr_det() -> bytes:
    return model([node("ReduceMean", ["x"], ["m"], [attr_ints("axes", [1]), attr_int("keepdims", 1)]),
                  node("Mul", ["m", "k"], ["z"]), node("Sigmoid", ["z"], ["fetch_name_0"])],
                 [value_info("x", FLOAT, ["N", 3, "H", "W"])], [value_info("fetch_name_0", FLOAT, ["N", 1, "H", "W"])],
                 [tensor("k", [], [4.0])])


def ppocr_rec(classes: int) -> bytes:
    """Class 0 (blank) has bias 0.5, class 1 weight 2 per channel, the rest bias -10: bright -> 1, dark/padding -> 0."""
    weights: List[float] = []
    for _channel in range(3):
        weights += [0.0, 2.0] + [0.0] * (classes - 2)
    bias = [0.5, 0.0] + [-10.0] * (classes - 2)
    return model([node("MaxPool", ["x"], ["p"], [attr_ints("kernel_shape", [48, 8]), attr_ints("strides", [48, 8])]),
                  node("Transpose", ["p"], ["t"], [attr_ints("perm", [0, 3, 1, 2])]),
                  node("Reshape", ["t", "shape"], ["r"]), node("MatMul", ["r", "w"], ["s"]), node("Add", ["s", "b"], ["l"]),
                  node("Softmax", ["l"], ["fetch_name_0"], [attr_int("axis", 2)])],
                 [value_info("x", FLOAT, ["N", 3, 48, "W"])], [value_info("fetch_name_0", FLOAT, ["N", "T", classes])],
                 [tensor("shape", [3], [0, -1, 3], INT64), tensor("w", [3, classes], weights), tensor("b", [classes], bias)])


def rfdetr(dets: Sequence[Sequence[float]], logits: Sequence[Sequence[float]], size: int = 64) -> bytes:
    """Constant outputs (plus 0 x the input's mean, so the input is used): dets [1, Q, 4], labels [1, Q, 91]."""
    q = len(dets)
    flat_d = [v for row in dets for v in row]
    flat_l = [v for row in logits for v in row]
    return model([node("ReduceMean", ["input"], ["m"], [attr_int("keepdims", 0)]), node("Mul", ["m", "zero"], ["z"]),
                  node("Add", ["z", "d"], ["dets"]), node("Add", ["z", "l"], ["labels"])],
                 [value_info("input", FLOAT, [1, 3, size, size])],
                 [value_info("dets", FLOAT, [1, q, 4]), value_info("labels", FLOAT, [1, q, 91])],
                 [tensor("zero", [], [0.0]), tensor("d", [1, q, 4], flat_d), tensor("l", [1, q, 91], flat_l)])


def write(path: str, data: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def dictionary_yml(chars: Sequence[str]) -> str:
    """A PaddleOCR-style inference.yml with the given character_dict (quoted the way PaddleOCR's file quotes them)."""
    lines = ["Global:", "  model_name: tiny_rec", "PostProcess:", "  name: CTCLabelDecode", "  character_dict:"]
    for c in chars:
        lines.append("  - '" + c.replace("'", "''") + "'" if not c.isalpha() else f"  - {c}")
    return "\n".join(lines) + "\n"


__all__ = ["ppocr_det", "ppocr_rec", "rfdetr", "write", "dictionary_yml"]
