"""Pure Python GGUF metadata reader."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

# GGUF magic number
GGUF_MAGIC = 0x46554747  # "GGUF" as ASCII read as little-endian uint32

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def _read_string(f: BinaryIO) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_value(f: BinaryIO, vtype: int) -> object:
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, elem_type) for _ in range(count)]
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF type: {vtype}")


def read_metadata(path: Path) -> dict[str, object]:
    """Read all KV metadata from a GGUF file header."""
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {path}")

        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        _tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            value = _read_value(f, vtype)
            metadata[key] = value

        return metadata
