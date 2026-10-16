"""Tests for configuration management."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_buddy.config import (
    all_model_ids,
    find_model_gguf_files,
    find_model_mtp_file,
    get_hf_hub_dir,
    llama_config_file,
    model_option,
    model_source,
    note_router_models,
    parse_vram_from_log,
    preset_path,
    read_pid,
    read_preset,
    read_vram_usage,
    remove_pid,
    resolve_model,
    sync_preset_with_cache,
    update_vram_usage,
    write_pid,
    write_preset,
)
from llama_buddy.ini import IniFile


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


def _make_hf_model(hf_dir, org, repo, filename):
    """Helper to create a fake HF hub model entry."""
    model = hf_dir / f"models--{org}--{repo}"
    snapshot = model / "snapshots" / "abc123"
    blobs = model / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(b"x" * 100)
    (snapshot / filename).symlink_to("../../blobs/deadbeef")
    refs = model / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123")


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An HF cache, with MTP detection out of the way."""
    import llama_buddy.models as models

    hf_dir = tmp_path / "hf_hub"
    hf_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: hf_dir)
    monkeypatch.setattr("llama_buddy.config._backfill_mtp", lambda preset: False)
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(models, "find_model_mtp_file", lambda mid: None)
    return hf_dir


def test_sync_adds_no_sections_for_cached_models(hub):
    _make_hf_model(hub, "org", "model-GGUF", "model-Q4_K_M.gguf")

    sync_preset_with_cache()

    assert read_preset().sections() == []
    assert all_model_ids() == ["org/model-GGUF:Q4_K_M"]


def test_sync_drops_empty_sections(hub):
    _make_hf_model(hub, "org", "model-GGUF", "model-Q4_K_M.gguf")
    preset = IniFile(
        "version = 1\n\n[*]\n\n[org/model-GGUF:Q4_K_M]\n\n"
        "[org/model-GGUF:Q4_K_M-tuned]\nhf-repo = org/model-GGUF:Q4_K_M\n"
    )
    write_preset(preset)

    sync_preset_with_cache()

    assert read_preset().text() == (
        "version = 1\n\n"
        "[org/model-GGUF:Q4_K_M-tuned]\nhf-repo = org/model-GGUF:Q4_K_M\n"
    )


def test_sync_drops_settings_for_removed_models(hub):
    write_preset(IniFile("[org/gone-GGUF:Q4_K_M]\ntemp = 0.7\n"))
    sync_preset_with_cache()
    assert read_preset().sections() == []


def test_sync_keeps_sections_that_bring_their_own_model(hub):
    text = (
        "; named presets\n[laguna]\nmodel = /models/laguna-Q4_K_M.gguf\n\n"
        "[org/missing-GGUF:Q8_0]\nhf-repo = org/missing-GGUF:Q8_0\n\n"
        "[local-file]\ntemp = 0.5\n"
    )
    write_preset(IniFile(text))
    sync_preset_with_cache()
    assert read_preset().text() == text


def test_sync_enables_mtp_once_per_new_model(hub, monkeypatch):
    import llama_buddy.models as models

    _make_hf_model(hub, "org", "model-GGUF", "model-Q4_K_M.gguf")
    seen: list[str] = []

    def fake_sync(mid, preset):
        seen.append(mid)
        models._read_mtp_cache()[mid] = {"mtp": True}
        preset.set(mid, "spec-type", "draft-mtp")
        return True

    monkeypatch.setattr(models, "sync_mtp_param", fake_sync)

    assert sync_preset_with_cache() == ["org/model-GGUF:Q4_K_M"]
    assert read_preset().get("org/model-GGUF:Q4_K_M", "spec-type") == "draft-mtp"

    # Turning it off by removing the key sticks: the model has been seen.
    preset = read_preset()
    preset.remove_option("org/model-GGUF:Q4_K_M", "spec-type")
    write_preset(preset)
    assert sync_preset_with_cache() == []
    assert seen == ["org/model-GGUF:Q4_K_M"]


# ---------------------------------------------------------------------------
# Where the preset lives and what its sections point at
# ---------------------------------------------------------------------------


def test_preset_path_defaults_to_llb_file(tmp_path):
    import llama_buddy.config as config

    assert preset_path() == config.PRESET_FILE


def test_preset_path_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_ARG_MODELS_PRESET", str(tmp_path / "mine.ini"))
    assert preset_path() == tmp_path / "mine.ini"


def test_preset_path_from_config_ini(tmp_path):
    cfg = llama_config_file()
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f"models-preset = {tmp_path}/presets.ini ; hand-made\n")
    assert preset_path() == tmp_path / "presets.ini"


def test_named_section_resolves_local_file(tmp_path):
    gguf = tmp_path / "laguna-s-2.1-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 10)
    write_preset(IniFile(f"[laguna]\nm = {gguf}\n"))

    assert model_source("laguna") == ("path", str(gguf))
    assert find_model_gguf_files("laguna") == [gguf]
    assert find_model_mtp_file("laguna") is None


def test_named_section_resolves_local_shards(tmp_path):
    shards = [tmp_path / f"big-Q4_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    for shard in shards:
        shard.write_bytes(b"x")
    write_preset(IniFile(f"[big]\nmodel = {shards[0]}\n"))

    assert find_model_gguf_files("big") == shards


def test_named_section_resolves_hf_repo(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hub"
    hf_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: hf_dir)
    _make_hf_model(hf_dir, "org", "model-GGUF", "model-Q4_K_M.gguf")
    write_preset(IniFile("[fast]\nhf = org/model-GGUF:Q4_K_M\n"))

    files = find_model_gguf_files("fast")
    assert [f.name for f in files] == ["model-Q4_K_M.gguf"]


def test_router_args_resolve_models_the_preset_does_not_know(tmp_path):
    gguf = tmp_path / "x-Q8_0.gguf"
    gguf.write_bytes(b"x")
    note_router_models([
        {"id": "x", "status": {"args": ["llama-server", "--model", str(gguf)]}},
        {"id": "y", "status": {"args": ["llama-server", "-hf", "org/y-GGUF:Q8_0"]}},
        {"id": "z", "status": "unloaded"},
    ])
    assert find_model_gguf_files("x") == [gguf]
    assert model_source("y") == ("hf", "org/y-GGUF:Q8_0")
    assert model_source("z") == ("hf", "z")


def test_resolve_model_finds_cached_model_without_section(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hub"
    hf_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: hf_dir)
    _make_hf_model(hf_dir, "org", "model-GGUF", "model-Q4_K_M.gguf")

    assert resolve_model("org/model-GGUF:Q4_K_M") == "org/model-GGUF:Q4_K_M"
    assert resolve_model("ORG/model-gguf:q4_k_m") == "org/model-GGUF:Q4_K_M"


def test_resolve_model_by_any_alias():
    write_preset(IniFile("[laguna]\nmodel = /x.gguf\na = lag, coder\n"))
    assert resolve_model("coder") == "laguna"
    assert resolve_model("LAG") == "laguna"


def test_model_option_falls_back_to_star():
    preset = IniFile("[*]\nctx-size = 8192\n\n[m]\nfa = on\n")
    assert model_option(preset, "m", "c") == "8192"
    assert model_option(preset, "m", "fa") == "on"
    assert model_option(preset, "other", "fa", "auto") == "auto"


# ---------------------------------------------------------------------------
# VRAM log parsing
# ---------------------------------------------------------------------------

_SAMPLE_LOG = """\
srv  load: spawning server instance with name=org/model-GGUF:Q4_K_M on port 12345
srv          load: spawning server instance with args:
[12345] load_tensors:   CPU_Mapped model buffer size =   544.00 MiB
[12345] load_tensors:  MTL0_Mapped model buffer size = 47618.95 MiB
[12345] llama_context:        CPU  output buffer size =     2.00 MiB
[12345] llama_kv_cache:       MTL0 KV buffer size =   500.00 MiB
[12345] sched_reserve:       MTL0 compute buffer size =   318.48 MiB
[12345] sched_reserve:        CPU compute buffer size =   141.01 MiB
[12345] srv    load_model: initializing slots, n_slots = 4
"""


def test_parse_vram_from_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(_SAMPLE_LOG)
    result = parse_vram_from_log(log)
    assert "org/model-GGUF:Q4_K_M" in result
    assert abs(result["org/model-GGUF:Q4_K_M"] - 49124.44) < 0.1


def test_parse_vram_multiple_models(tmp_path):
    log_text = _SAMPLE_LOG + """\
srv          load: spawning server instance with name=org/other-GGUF:Q8_0 on port 12346
[12346] load_tensors:   CPU_Mapped model buffer size =   100.00 MiB
[12346] load_tensors:  MTL0_Mapped model buffer size =  2000.00 MiB
[12346] srv    load_model: initializing slots, n_slots = 2
"""
    log = tmp_path / "server.log"
    log.write_text(log_text)
    result = parse_vram_from_log(log)
    assert len(result) == 2
    assert abs(result["org/other-GGUF:Q8_0"] - 2100.0) < 0.1


def test_parse_vram_reload_overwrites(tmp_path):
    """Wake-from-sleep reload should overwrite the previous value."""
    log_text = """\
srv  load: spawning server instance with name=org/model-GGUF:Q4_K_M on port 12345
[12345] load_tensors:   CPU_Mapped model buffer size =   100.00 MiB
[12345] srv    load_model: initializing slots, n_slots = 4
[12345] load_tensors:   CPU_Mapped model buffer size =   200.00 MiB
[12345] srv    load_model: initializing slots, n_slots = 4
"""
    log = tmp_path / "server.log"
    log.write_text(log_text)
    result = parse_vram_from_log(log)
    assert abs(result["org/model-GGUF:Q4_K_M"] - 200.0) < 0.1


def test_parse_vram_no_log(tmp_path):
    result = parse_vram_from_log(tmp_path / "nonexistent.log")
    assert result == {}


def test_read_vram_usage_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("llama_buddy.config.VRAM_FILE", tmp_path / "vram.json")
    assert read_vram_usage() == {}


def test_update_vram_usage_writes_file(tmp_path, monkeypatch):
    vram_file = tmp_path / "vram.json"
    monkeypatch.setattr("llama_buddy.config.VRAM_FILE", vram_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)

    log = tmp_path / "server.log"
    log.write_text(_SAMPLE_LOG)
    result = update_vram_usage(log)
    assert "org/model-GGUF:Q4_K_M" in result
    assert vram_file.exists()

    # Reading back should return the same data
    cached = read_vram_usage()
    assert cached == result


def test_find_files_for_id_without_org(tmp_path, monkeypatch):
    # The router lists local GGUFs by bare name; they have no HF hub dir
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: tmp_path)
    assert find_model_gguf_files("laguna-s-2.1-Q4_K_M") == []
    assert find_model_gguf_files("local-model:Q4_K_M") == []
    assert find_model_mtp_file("local-model") is None


_HUB_VARS = (
    "LLAMA_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
    "XDG_CACHE_HOME",
)


def test_hf_hub_dir_follows_llama_cpp_order(tmp_path, monkeypatch):
    for var in _HUB_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_hf_hub_dir() == tmp_path / ".cache" / "huggingface" / "hub"

    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert get_hf_hub_dir() == Path("/xdg/huggingface/hub")
    monkeypatch.setenv("HF_HOME", "/hf")
    assert get_hf_hub_dir() == Path("/hf/hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/hub-legacy")
    assert get_hf_hub_dir() == Path("/hub-legacy")
    monkeypatch.setenv("HF_HUB_CACHE", "/hub")
    assert get_hf_hub_dir() == Path("/hub")
    monkeypatch.setenv("LLAMA_CACHE", "/llama")
    assert get_hf_hub_dir() == Path("/llama")
