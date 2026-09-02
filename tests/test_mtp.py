"""Tests for multi-token-prediction detection and its preset wiring."""

from __future__ import annotations

import configparser
import struct

import pytest

from llama_buddy import models
from llama_buddy.gguf import (
    GGUF_MAGIC,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    has_tensor,
    mtp_tensor_name,
    read_scalar_metadata,
)
from llama_buddy.models import _detect_mtp, _mtp_cell, sync_mtp_param


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _tensor_info(name: str) -> bytes:
    return (
        _string(name)
        + struct.pack("<I", 1)      # n_dims
        + struct.pack("<Q", 4096)   # dims
        + struct.pack("<I", 0)      # type
        + struct.pack("<Q", 0)      # offset
    )


def _write_gguf(path, arch: str, block_count: int, tensors: list[str],
                vocab: list[str] | None = None):
    """Write a GGUF header with a KV block and a tensor info block."""
    kv = bytearray()
    kv_count = 2

    kv += _string("general.architecture")
    kv += struct.pack("<I", GGUF_TYPE_STRING)
    kv += _string(arch)

    kv += _string(f"{arch}.block_count")
    kv += struct.pack("<I", GGUF_TYPE_UINT32)
    kv += struct.pack("<I", block_count)

    if vocab is not None:
        kv_count += 1
        kv += _string("tokenizer.ggml.tokens")
        kv += struct.pack("<I", GGUF_TYPE_ARRAY)
        kv += struct.pack("<I", GGUF_TYPE_STRING)
        kv += struct.pack("<Q", len(vocab))
        for token in vocab:
            kv += _string(token)

    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", kv_count)
    buf += kv
    for name in tensors:
        buf += _tensor_info(name)

    path.write_bytes(bytes(buf))
    return path


def test_mtp_tensor_name_from_block_count():
    meta = {"general.architecture": "qwen3next", "qwen3next.block_count": 65}
    assert mtp_tensor_name(meta) == "blk.64.nextn.eh_proj.weight"


def test_mtp_tensor_name_without_block_count():
    assert mtp_tensor_name({"general.architecture": "llama"}) is None


def test_scalar_metadata_skips_arrays(tmp_path):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4, ["blk.0.attn_q.weight"],
        vocab=[f"tok{i}" for i in range(500)],
    )
    meta = read_scalar_metadata(path)
    assert meta["general.architecture"] == "qwen3"
    assert meta["qwen3.block_count"] == 4
    assert "tokenizer.ggml.tokens" not in meta


def test_has_tensor_finds_name_behind_a_vocab(tmp_path):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4,
        ["blk.0.attn_q.weight", "blk.3.nextn.eh_proj.weight"],
        vocab=[f"tok{i}" for i in range(500)],
    )
    assert has_tensor(path, "blk.3.nextn.eh_proj.weight") is True
    assert has_tensor(path, "blk.9.nextn.eh_proj.weight") is False


def test_detect_mtp_present(tmp_path):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4,
        ["blk.0.attn_q.weight", "blk.3.nextn.eh_proj.weight"],
    )
    assert _detect_mtp([path]) is True


def test_detect_mtp_absent(tmp_path):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4, ["blk.0.attn_q.weight"],
    )
    assert _detect_mtp([path]) is False


def test_detect_mtp_head_in_later_shard(tmp_path):
    """The block count lives in shard 1, the head may live in a later one."""
    first = _write_gguf(
        tmp_path / "m-00001-of-00002.gguf", "qwen3", 4,
        ["blk.0.attn_q.weight"],
    )
    second = _write_gguf(
        tmp_path / "m-00002-of-00002.gguf", "qwen3", 4,
        ["blk.3.nextn.eh_proj.weight"],
    )
    assert _detect_mtp([first, second]) is True


def test_detect_mtp_on_unreadable_file(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOT_A_GGUF_FILE_AT_ALL")
    assert _detect_mtp([bad]) is False


@pytest.fixture
def preset():
    config = configparser.ConfigParser()
    config.add_section("org/repo:Q8_0")
    return config


def test_sync_mtp_param_enables(monkeypatch, preset):
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    assert sync_mtp_param("org/repo:Q8_0", preset) is True
    assert preset.get("org/repo:Q8_0", "spec-type") == "draft-mtp"


def test_sync_mtp_param_skips_model_without_head(monkeypatch, preset):
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    assert sync_mtp_param("org/repo:Q8_0", preset) is False
    assert preset.has_option("org/repo:Q8_0", "spec-type") is False


def test_sync_mtp_param_respects_manual_choice(monkeypatch, preset):
    """Turning MTP off in the settings editor must survive the next sync."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    preset.set("org/repo:Q8_0", "spec-type", "none")
    assert sync_mtp_param("org/repo:Q8_0", preset) is False
    assert preset.get("org/repo:Q8_0", "spec-type") == "none"


def test_mtp_cell_states(monkeypatch, preset):
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    # Head present, no spec-type: llama-server ignores the head.
    assert "off" in _mtp_cell("org/repo:Q8_0", preset)

    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    assert "on" in _mtp_cell("org/repo:Q8_0", preset)

    preset.set("org/repo:Q8_0", "spec-type", "none")
    assert "off" in _mtp_cell("org/repo:Q8_0", preset)

    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    assert _mtp_cell("org/repo:Q8_0", preset) == "[dim]-[/dim]"


def test_mtp_cell_without_preset_section(monkeypatch):
    """configparser raises NoSectionError even with a fallback."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    empty = configparser.ConfigParser()
    assert "off" in _mtp_cell("org/repo:Q8_0", empty)


def test_model_has_mtp_uses_disk_cache(tmp_path, monkeypatch):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4, ["blk.3.nextn.eh_proj.weight"],
    )
    monkeypatch.setattr(models, "MTP_CACHE_FILE", tmp_path / "mtp.json")
    monkeypatch.setattr(models, "_mtp_cache", None)
    monkeypatch.setattr(models, "find_model_gguf_files", lambda mid: [path])
    models.model_has_mtp.cache_clear()

    assert models.model_has_mtp("org/repo:Q8_0") is True

    calls = []

    def counted(files):
        calls.append(files)
        return False

    monkeypatch.setattr(models, "_detect_mtp", counted)
    monkeypatch.setattr(models, "_mtp_cache", None)
    models.model_has_mtp.cache_clear()

    assert models.model_has_mtp("org/repo:Q8_0") is True
    assert calls == []


def test_model_has_mtp_rescans_after_redownload(tmp_path, monkeypatch):
    path = _write_gguf(
        tmp_path / "m.gguf", "qwen3", 4, ["blk.0.attn_q.weight"],
    )
    monkeypatch.setattr(models, "MTP_CACHE_FILE", tmp_path / "mtp.json")
    monkeypatch.setattr(models, "_mtp_cache", None)
    monkeypatch.setattr(models, "find_model_gguf_files", lambda mid: [path])
    models.model_has_mtp.cache_clear()

    assert models.model_has_mtp("org/repo:Q8_0") is False

    # Same name, different quant: the file size changes, so the cached
    # answer no longer applies.
    _write_gguf(
        path, "qwen3", 4,
        ["blk.0.attn_q.weight", "blk.3.nextn.eh_proj.weight"],
    )
    monkeypatch.setattr(models, "_mtp_cache", None)
    models.model_has_mtp.cache_clear()

    assert models.model_has_mtp("org/repo:Q8_0") is True


def test_backfill_runs_again_when_llama_server_changes(tmp_path, monkeypatch):
    """A head the old build could not load may work on the new one."""
    from llama_buddy import capabilities, config

    monkeypatch.setattr(config, "MTP_BACKFILL_FILE", tmp_path / "mtp_backfill_done")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config.shutil, "which", lambda name: "/bin/llama-server")
    monkeypatch.setattr(capabilities, "_binary_key", lambda b: "build-1")
    synced: list[str] = []
    monkeypatch.setattr(
        models, "sync_mtp_param", lambda mid, p: (synced.append(mid), False)[1],
    )
    monkeypatch.setattr(models, "unwire_unloadable_sidecar", lambda mid, p: False)
    preset = configparser.ConfigParser()
    preset.add_section("org/repo:Q8_0")

    config._backfill_mtp(preset)
    config._backfill_mtp(preset)
    assert synced == ["org/repo:Q8_0"]

    monkeypatch.setattr(capabilities, "_binary_key", lambda b: "build-2")
    config._backfill_mtp(preset)
    assert synced == ["org/repo:Q8_0", "org/repo:Q8_0"]
