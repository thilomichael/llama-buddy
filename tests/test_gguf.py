"""Tests for GGUF metadata reader."""

from __future__ import annotations

import struct

from llama_buddy.gguf import (
    GGUF_MAGIC,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_STRING,
    read_metadata,
)


def _make_gguf_header(kv_pairs: list[tuple[str, int, bytes]]) -> bytes:
    """Build a minimal GGUF header with given KV pairs."""
    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)   # magic
    buf += struct.pack("<I", 3)             # version
    buf += struct.pack("<Q", 0)             # tensor count
    buf += struct.pack("<Q", len(kv_pairs)) # kv count

    for key, vtype, value_bytes in kv_pairs:
        key_encoded = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_encoded))
        buf += key_encoded
        buf += struct.pack("<I", vtype)
        buf += value_bytes

    return bytes(buf)


def _encode_string_value(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def test_read_metadata_string(tmp_path):
    kv = [("general.name", GGUF_TYPE_STRING, _encode_string_value("TestModel"))]
    gguf_path = tmp_path / "test.gguf"
    gguf_path.write_bytes(_make_gguf_header(kv))

    meta = read_metadata(gguf_path)
    assert meta["general.name"] == "TestModel"


def test_read_metadata_float(tmp_path):
    kv = [("general.sampling.temp", GGUF_TYPE_FLOAT32, struct.pack("<f", 0.7))]
    gguf_path = tmp_path / "test.gguf"
    gguf_path.write_bytes(_make_gguf_header(kv))

    meta = read_metadata(gguf_path)
    assert abs(meta["general.sampling.temp"] - 0.7) < 0.001


def test_read_metadata_not_gguf(tmp_path):
    bad_file = tmp_path / "bad.gguf"
    bad_file.write_bytes(b"NOT_GGUF_DATA_HERE")

    import pytest
    with pytest.raises(ValueError, match="Not a GGUF file"):
        read_metadata(bad_file)
