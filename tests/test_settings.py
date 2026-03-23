"""Tests for settings management."""

from __future__ import annotations

import json

from llama_buddy.settings import Settings, load_settings, save_settings


def test_default_settings():
    s = Settings()
    assert s.port == 8080
    assert s.idle_timeout == 300
    assert s.jinja is True
    assert s.flash_attention == "auto"
    assert s.ctx_size == 8192
    assert s.gpu_layers == "auto"


def test_settings_roundtrip(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("llama_buddy.settings.SETTINGS_FILE", settings_file)
    monkeypatch.setattr("llama_buddy.settings.CONFIG_DIR", tmp_path)

    s = Settings(port=9090, idle_timeout=600, jinja=False)
    save_settings(s)

    loaded = load_settings()
    assert loaded.port == 9090
    assert loaded.idle_timeout == 600
    assert loaded.jinja is False
    assert loaded.flash_attention == "auto"


def test_settings_ignores_unknown_keys(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"port": 9090, "unknown_key": "val"}))
    monkeypatch.setattr("llama_buddy.settings.SETTINGS_FILE", settings_file)

    loaded = load_settings()
    assert loaded.port == 9090


def test_settings_corrupt_file(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("not json")
    monkeypatch.setattr("llama_buddy.settings.SETTINGS_FILE", settings_file)

    loaded = load_settings()
    assert loaded.port == 8080  # falls back to defaults


def test_to_server_args():
    s = Settings(port=9090, idle_timeout=600, jinja=False, ctx_size=4096)
    args = s.to_server_args()
    assert "--port" in args
    assert "9090" in args
    assert "--sleep-idle-seconds" in args
    assert "600" in args
    assert "--no-jinja" in args
    assert "--ctx-size" in args
    assert "4096" in args


def test_to_server_args_default_ctx():
    s = Settings()
    args = s.to_server_args()
    assert "--ctx-size" in args
    assert "8192" in args
