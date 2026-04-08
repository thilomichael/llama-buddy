"""Tests for configuration management."""

from __future__ import annotations

from llama_buddy.config import (
    parse_vram_from_log,
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


def test_sync_preset_with_cache(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    hf_dir = tmp_path / "hf_hub"
    hf_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: hf_dir)

    # Create an existing preset with one model
    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    write_preset(config)

    # Create HF hub entries: one matching existing, one new
    _make_hf_model(hf_dir, "org", "model-GGUF", "model-Q4_K_M.gguf")
    _make_hf_model(hf_dir, "other", "newmodel-GGUF", "newmodel-Q8_0.gguf")

    added = sync_preset_with_cache()

    assert "other/newmodel-GGUF:Q8_0" in added
    assert len(added) == 1

    # Verify it's in the preset now
    config2 = read_preset()
    assert "other/newmodel-GGUF:Q8_0" in config2.sections()


def test_sync_preset_no_duplicates(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    hf_dir = tmp_path / "hf_hub"
    hf_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("llama_buddy.config.get_hf_hub_dir", lambda: hf_dir)

    _make_hf_model(hf_dir, "org", "model-GGUF", "model-Q4_K_M.gguf")

    # First sync adds it
    added1 = sync_preset_with_cache()
    assert len(added1) == 1

    # Second sync should find nothing new
    added2 = sync_preset_with_cache()
    assert len(added2) == 0


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
