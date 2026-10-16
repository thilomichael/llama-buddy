"""Keep every test away from the real config files and environment."""

from __future__ import annotations

import pytest

# Every file llb reads or writes under its config dir, by module.
_FILES = {
    "llama_buddy.config": {
        "PRESET_FILE": "models.ini",
        "PID_FILE": "server.pid",
        "LOG_FILE": "server.log",
        "VRAM_FILE": "vram.json",
        "MTP_BACKFILL_FILE": "mtp_backfill_done",
    },
    "llama_buddy.capabilities": {"CAPABILITIES_FILE": "capabilities.json"},
    "llama_buddy.models": {"MTP_CACHE_FILE": "mtp.json"},
    "llama_buddy.settings": {"SETTINGS_FILE": "settings.json"},
    "llama_buddy.server": {
        "WATCHDOG_PID_FILE": "watchdog.pid",
        "EXTRA_ARGS_FILE": "extra_args.json",
        "EFFECTIVE_PRESET_FILE": "models.effective.ini",
    },
}


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path_factory, monkeypatch):
    import llama_buddy.models as models

    home = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "xdg"))
    for var in (
        "LLAMA_ARG_PORT", "LLAMA_ARG_MODELS_PRESET", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLAMA_CACHE", str(home / "hub"))

    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", home)
    for module, files in _FILES.items():
        for attr, name in files.items():
            monkeypatch.setattr(f"{module}.{attr}", home / name)

    monkeypatch.setattr("llama_buddy.config._preset_cache", None)
    monkeypatch.setattr("llama_buddy.config._router_sources", {})
    monkeypatch.setattr(models, "_mtp_cache", None)
    models.model_has_mtp.cache_clear()
    models.get_model_meta.cache_clear()
