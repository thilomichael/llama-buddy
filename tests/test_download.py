"""Tests for download utilities."""

from __future__ import annotations

from llama_buddy.config import read_preset, write_preset
from llama_buddy.download import _merge_shards, remove


def test_remove_by_alias(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    config.set("org/model-GGUF:Q4_K_M", "alias", "mymodel")
    write_preset(config)

    remove("mymodel")

    config2 = read_preset()
    assert "org/model-GGUF:Q4_K_M" not in config2.sections()


def test_remove_by_id(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    write_preset(config)

    remove("org/model-GGUF:Q4_K_M")

    config2 = read_preset()
    assert "org/model-GGUF:Q4_K_M" not in config2.sections()


def test_merge_shards_groups_split_files():
    files = [
        {"path": "Q4_K_M/Model-Q4_K_M-00001-of-00003.gguf", "size": 100},
        {"path": "Q4_K_M/Model-Q4_K_M-00002-of-00003.gguf", "size": 100},
        {"path": "Q4_K_M/Model-Q4_K_M-00003-of-00003.gguf", "size": 50},
        {"path": "Model-Q8_0.gguf", "size": 500},
    ]
    merged = _merge_shards(files)
    assert len(merged) == 2

    # Single file passes through
    single = [f for f in merged if "shard_files" not in f]
    assert len(single) == 1
    assert single[0]["path"] == "Model-Q8_0.gguf"

    # Shards are grouped
    grouped = [f for f in merged if "shard_files" in f]
    assert len(grouped) == 1
    assert grouped[0]["size"] == 250
    assert len(grouped[0]["shard_files"]) == 3
    assert grouped[0]["path"] == "Q4_K_M/Model-Q4_K_M.gguf"


def test_merge_shards_no_shards():
    files = [
        {"path": "Model-Q4_K_M.gguf", "size": 100},
        {"path": "Model-Q8_0.gguf", "size": 200},
    ]
    merged = _merge_shards(files)
    assert len(merged) == 2
    assert all("shard_files" not in f for f in merged)


def test_remove_not_found(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)

    import pytest

    with pytest.raises(SystemExit):
        remove("nonexistent")
