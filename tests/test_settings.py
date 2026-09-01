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


def test_idle_timeout_zero_is_migrated(tmp_path, monkeypatch):
    """llama-server refuses to start on --sleep-idle-seconds 0."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"idle_timeout": 0}))
    monkeypatch.setattr("llama_buddy.settings.SETTINGS_FILE", settings_file)

    loaded = load_settings()
    assert loaded.idle_timeout == -1


def test_never_sleep_passes_minus_one():
    args = Settings(idle_timeout=-1).to_server_args()
    assert args[args.index("--sleep-idle-seconds") + 1] == "-1"


def _allow(monkeypatch, flags=True, values=True):
    monkeypatch.setattr(
        "llama_buddy.settings.has_flag",
        flags if callable(flags) else (lambda flag: flags),
    )
    monkeypatch.setattr(
        "llama_buddy.settings.has_value",
        values if callable(values) else (lambda flag, value: values),
    )


def test_unsupported_flags_are_left_out(monkeypatch):
    _allow(monkeypatch, flags=lambda flag: flag == "--load-mode")
    args = Settings(kv_unified_per_slot=4096).to_server_args()
    assert "--load-mode" in args
    assert "--lazy-mode" not in args
    assert "--kv-unified-per-slot" not in args


def test_unsupported_value_is_left_out(monkeypatch):
    """Build 10344 has --load-mode but rejects 'auto'; the build default wins."""
    _allow(monkeypatch, values=lambda flag, value: value != "auto")
    args = Settings(load_mode="auto", lazy_mode="off").to_server_args()
    assert "--load-mode" not in args
    assert args[args.index("--lazy-mode") + 1] == "off"


def test_supported_flags_are_emitted(monkeypatch):
    _allow(monkeypatch)
    args = Settings(kv_unified_per_slot=4096).to_server_args()
    assert args[args.index("--load-mode") + 1] == "auto"
    assert args[args.index("--lazy-mode") + 1] == "auto"
    assert args[args.index("--kv-unified-per-slot") + 1] == "4096"


def test_kv_unified_per_slot_unset_is_omitted(monkeypatch):
    _allow(monkeypatch)
    assert "--kv-unified-per-slot" not in Settings().to_server_args()
