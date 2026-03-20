"""Tests for configuration management."""

from __future__ import annotations

from llama_buddy.config import (
    read_pid,
    read_preset,
    remove_pid,
    resolve_model,
    write_pid,
    write_preset,
)


def test_pid_roundtrip(tmp_path, monkeypatch):
    pid_file = tmp_path / "server.pid"
    monkeypatch.setattr("llama_buddy.config.PID_FILE", pid_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    assert read_pid() is None

    write_pid(12345)
    assert read_pid() == 12345

    remove_pid()
    assert read_pid() is None


def test_pid_invalid(tmp_path, monkeypatch):
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("not-a-number")
    monkeypatch.setattr("llama_buddy.config.PID_FILE", pid_file)

    assert read_pid() is None


def test_preset_roundtrip(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    config = read_preset()
    assert config.sections() == []

    config.add_section("org/model-GGUF:Q4_K_M")
    config.set("org/model-GGUF:Q4_K_M", "alias", "model")
    write_preset(config)

    config2 = read_preset()
    assert "org/model-GGUF:Q4_K_M" in config2.sections()
    assert config2.get("org/model-GGUF:Q4_K_M", "alias") == "model"


def test_resolve_model_by_id(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)

    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    config.set("org/model-GGUF:Q4_K_M", "alias", "mymodel")
    write_preset(config)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    assert resolve_model("org/model-GGUF:Q4_K_M") == "org/model-GGUF:Q4_K_M"


def test_resolve_model_by_alias(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    config.set("org/model-GGUF:Q4_K_M", "alias", "mymodel")
    write_preset(config)

    assert resolve_model("mymodel") == "org/model-GGUF:Q4_K_M"


def test_resolve_model_case_insensitive(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    config.set("org/model-GGUF:Q4_K_M", "alias", "mymodel")
    write_preset(config)

    assert resolve_model("MyModel") == "org/model-GGUF:Q4_K_M"


def test_resolve_model_not_found(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)

    assert resolve_model("nonexistent") is None
