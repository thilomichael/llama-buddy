"""Tests for download utilities."""

from __future__ import annotations

from llama_buddy.config import read_preset, write_preset
from llama_buddy.download import remove


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


def test_remove_not_found(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)

    import pytest

    with pytest.raises(SystemExit):
        remove("nonexistent")
