"""Pure Python GGUF metadata reader."""

from __future__ import annotations

import mmap
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


def _read_header(f: BinaryIO, path: Path) -> tuple[dict[str, object], int]:
    """Read the magic, version and KV block. Returns (metadata, tensor_count)."""
    magic = struct.unpack("<I", f.read(4))[0]
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file: {path}")

    version = struct.unpack("<I", f.read(4))[0]
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version: {version}")

    tensor_count = struct.unpack("<Q", f.read(8))[0]
    kv_count = struct.unpack("<Q", f.read(8))[0]

    metadata: dict[str, object] = {}
    for _ in range(kv_count):
        key = _read_string(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        value = _read_value(f, vtype)
        metadata[key] = value

    return metadata, tensor_count


def read_metadata(path: Path) -> dict[str, object]:
    """Read all KV metadata from a GGUF file header."""
    with open(path, "rb") as f:
        metadata, _ = _read_header(f, path)
        return metadata


# Byte width of every fixed-size GGUF value type.
_SCALAR_SIZES = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


# struct format character for every fixed-size GGUF value type.
_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<?",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


def _value_at(buf, off: int, vtype: int) -> object:
    """Read a single non-array value out of a mapped buffer."""
    if vtype == GGUF_TYPE_STRING:
        length = struct.unpack_from("<Q", buf, off)[0]
        return bytes(buf[off + 8 : off + 8 + length]).decode("utf-8")
    fmt = _SCALAR_FORMATS.get(vtype)
    if fmt is None:
        raise ValueError(f"Unknown GGUF type: {vtype}")
    return struct.unpack_from(fmt, buf, off)[0]


def _skip_value(buf, off: int, vtype: int) -> int:
    """Return the offset just past a value, without materialising it.

    A tokenizer vocabulary is a GGUF array of 100k+ strings. Decoding one
    to reach the tensor block behind it costs ~300ms per model, so values
    that the caller did not ask for are walked as offsets instead.
    """
    if vtype == GGUF_TYPE_STRING:
        return off + 8 + struct.unpack_from("<Q", buf, off)[0]

    if vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack_from("<I", buf, off)[0]
        count = struct.unpack_from("<Q", buf, off + 4)[0]
        off += 12
        size = _SCALAR_SIZES.get(elem_type)
        if size is not None:
            return off + size * count
        for _ in range(count):
            off = _skip_value(buf, off, elem_type)
        return off

    size = _SCALAR_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"Unknown GGUF type: {vtype}")
    return off + size


def _scan_header(buf, path: Path) -> tuple[dict[str, object], int, int]:
    """Walk magic, version and the KV block over a mapped file.

    Array values are skipped rather than decoded. Returns the scalar
    metadata, the tensor count and the offset of the tensor info block.
    """
    if struct.unpack_from("<I", buf, 0)[0] != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file: {path}")

    version = struct.unpack_from("<I", buf, 4)[0]
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version: {version}")

    tensor_count = struct.unpack_from("<Q", buf, 8)[0]
    kv_count = struct.unpack_from("<Q", buf, 16)[0]

    off = 24
    metadata: dict[str, object] = {}
    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", buf, off)[0]
        key = bytes(buf[off + 8 : off + 8 + key_len]).decode("utf-8")
        off += 8 + key_len
        vtype = struct.unpack_from("<I", buf, off)[0]
        off += 4
        end = _skip_value(buf, off, vtype)
        if vtype != GGUF_TYPE_ARRAY:
            metadata[key] = _value_at(buf, off, vtype)
        off = end

    return metadata, tensor_count, off


def has_tensor(path: Path, name: str) -> bool:
    """Check whether a GGUF's tensor info block declares a named tensor.

    Names are compared as raw bytes so the thousands of tensor names in a
    large model never become Python strings.
    """
    target = name.encode("utf-8")
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            _, tensor_count, off = _scan_header(buf, path)

            for _ in range(tensor_count):
                name_len = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                if name_len == len(target) and buf[off : off + name_len] == target:
                    return True
                off += name_len
                n_dims = struct.unpack_from("<I", buf, off)[0]
                off += 4 + 8 * n_dims + 4 + 8

    return False


def read_scalar_metadata(path: Path) -> dict[str, object]:
    """Read the non-array KV metadata from a GGUF file header.

    Skips array values, which makes it fast enough to call per model.
    """
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            metadata, _, _ = _scan_header(buf, path)
            return metadata


def mtp_tensor_name(metadata: dict[str, object]) -> str | None:
    """Return the tensor name that marks a multi-token-prediction head.

    Mirrors llama.cpp's ``common_speculative_types_from_gguf``: the block
    past the last transformer block holds an ``nextn.eh_proj`` tensor.
    Returns None when the metadata does not state a block count.
    """
    arch = metadata.get("general.architecture")
    block_count = metadata.get(f"{arch}.block_count")
    if not isinstance(block_count, int):
        return None
    return f"blk.{block_count - 1}.nextn.eh_proj.weight"
