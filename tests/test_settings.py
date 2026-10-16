"""Tests for settings: the preset's [*] section, the port and migration."""

from __future__ import annotations

import json

import llama_buddy.settings as settings
from llama_buddy.config import get_port, llama_config_file, read_preset, set_port
from llama_buddy.ini import IniFile
from llama_buddy.server import _drop_unsupported
from llama_buddy.settings import migrate_settings_file


def _migrate(data: dict | str, preset: IniFile | None = None) -> IniFile:
    settings.SETTINGS_FILE.write_text(
        data if isinstance(data, str) else json.dumps(data)
    )
    preset = preset or IniFile()
    migrate_settings_file(preset)
    return preset


def test_port_defaults_to_8080():
    assert get_port() == 8080


def test_port_from_config_ini():
    path = llama_config_file()
    path.parent.mkdir(parents=True)
    path.write_text("; mine\nport = 9090\n")
    assert get_port() == 9090


def test_port_env_wins(monkeypatch):
    set_port(9090)
    monkeypatch.setenv("LLAMA_ARG_PORT", "7070")
    assert get_port() == 7070


def test_set_port_keeps_the_rest_of_config_ini():
    path = llama_config_file()
    path.parent.mkdir(parents=True)
    path.write_text("; shared by all llama.cpp tools\nthreads = 8\n")
    set_port(9090)
    assert path.read_text() == (
        "; shared by all llama.cpp tools\nthreads = 8\nport = 9090\n"
    )
    set_port(8080)
    assert path.read_text() == "; shared by all llama.cpp tools\nthreads = 8\n"


def test_default_port_does_not_create_config_ini():
    set_port(8080)
    assert not llama_config_file().exists()


def test_migration_keeps_only_non_default_values():
    preset = _migrate({
        "port": 8080, "idle_timeout": 300, "jinja": True,
        "flash_attention": "auto", "ctx_size": 0, "gpu_layers": "auto",
    })
    assert preset.options("*") == ["sleep-idle-seconds"]
    assert preset.get("*", "sleep-idle-seconds") == "300"


def test_migration_removes_settings_json():
    _migrate({"port": 8080})
    assert not settings.SETTINGS_FILE.exists()


def test_migration_moves_port_to_config_ini():
    _migrate({"port": 9090})
    assert get_port() == 9090


def test_migration_fills_in_old_defaults():
    """A field missing from settings.json got llb's old default."""
    preset = _migrate({"port": 8080})
    assert preset.get("*", "sleep-idle-seconds") == "300"
    assert preset.get("*", "c") == "8192"


def test_migration_overrides_star_section():
    """The settings were router flags, which beat [*] before."""
    preset = IniFile("[*]\nc = 0\nctx-size = 4096\n")
    _migrate({"ctx_size": 8192, "jinja": False}, preset)
    assert preset.get("*", "c") == "8192"
    assert preset.get("*", "ctx-size") is None
    assert preset.get("*", "jinja") == "false"


def test_migration_turns_zero_idle_timeout_into_never():
    preset = _migrate({"idle_timeout": 0})
    assert preset.get("*", "sleep-idle-seconds") is None


def test_migration_survives_corrupt_file():
    preset = _migrate("not json")
    assert preset.get("*", "c") == "8192"
    assert not settings.SETTINGS_FILE.exists()


def test_sync_runs_the_migration(monkeypatch, tmp_path):
    import llama_buddy.config as config

    monkeypatch.setattr(config, "get_hf_hub_dir", lambda: tmp_path / "hub")
    monkeypatch.setattr(
        config, "MTP_BACKFILL_FILE", tmp_path / "mtp_backfill_done"
    )
    monkeypatch.setattr(config, "_backfill_mtp", lambda preset: False)
    settings.SETTINGS_FILE.write_text(json.dumps({"idle_timeout": 600}))

    config.sync_preset_with_cache()

    assert read_preset().get("*", "sleep-idle-seconds") == "600"


def _allow(monkeypatch, flags=True, values=True):
    monkeypatch.setattr(
        "llama_buddy.capabilities.has_flag",
        flags if callable(flags) else (lambda flag: flags),
    )
    monkeypatch.setattr(
        "llama_buddy.capabilities.has_value",
        values if callable(values) else (lambda flag, value: values),
    )


def test_unsupported_flags_are_dropped(monkeypatch):
    _allow(monkeypatch, flags=lambda flag: flag == "--load-mode")
    preset = IniFile(
        "[*]\nload-mode = mmap\nlazy-mode = on\nkv-unified-per-slot = 4096\n"
        "[org/m:Q4]\nspec-type = draft-mtp\nc = 4096\n"
    )
    _drop_unsupported(preset)
    assert preset.options("*") == ["load-mode"]
    assert preset.options("org/m:Q4") == ["c"]


def test_unsupported_value_is_dropped(monkeypatch):
    """Build 10344 has --load-mode but rejects 'auto'; the build default wins."""
    _allow(monkeypatch, values=lambda flag, value: value != "auto")
    preset = IniFile("[*]\nlm = auto\nlazy-mode = off\n")
    _drop_unsupported(preset)
    assert preset.options("*") == ["lazy-mode"]


def test_supported_settings_are_kept(monkeypatch):
    _allow(monkeypatch)
    text = "[*]\nload-mode = auto\nkv-unified-per-slot = 4096\n"
    preset = IniFile(text)
    _drop_unsupported(preset)
    assert preset.text() == text


def test_start_passes_no_model_settings(monkeypatch, tmp_path):
    """Router flags override every model's section, so none are passed."""
    import llama_buddy.server as server

    calls = []
    monkeypatch.setattr(server, "read_pid", lambda: None)
    monkeypatch.setattr(server, "find_server_binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(server, "LOG_FILE", tmp_path / "server.log")
    monkeypatch.setattr(server, "EFFECTIVE_PRESET_FILE", tmp_path / "eff.ini")
    monkeypatch.setattr(server, "_build_effective_preset", lambda: None)
    monkeypatch.setattr(server, "_remove_stale_watchdog", lambda: None)
    monkeypatch.setattr(server, "_save_extra_args", lambda args: None)
    monkeypatch.setattr(server, "write_pid", lambda pid: None)
    monkeypatch.setattr(
        server.subprocess, "Popen",
        lambda cmd, **kw: calls.append(cmd) or type("P", (), {"pid": 1})(),
    )
    set_port(9090)

    server.start(["--host", "0.0.0.0"])

    assert calls == [[
        "/bin/llama-server", "--models-preset", str(tmp_path / "eff.ini"),
        "--port", "9090", "--host", "0.0.0.0",
    ]]


def test_gpu_layers_none_becomes_zero(monkeypatch):
    """llama-server parses --gpu-layers as auto, all or a number."""
    _allow(monkeypatch)
    preset = IniFile("[*]\nngl = none\n[m]\nn-gpu-layers = none\n")
    _drop_unsupported(preset)
    assert preset.get("*", "ngl") == "0"
    assert preset.get("m", "n-gpu-layers") == "0"
    assert _migrate({"gpu_layers": "none"}).get("*", "ngl") == "0"
