"""Tests for configuration management."""

from __future__ import annotations

from llama_buddy.config import (
    read_pid,
    read_preset,
    remove_pid,
    resolve_model,
    sync_preset_with_cache,
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


def test_sync_preset_with_cache(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("llama_buddy.config.get_cache_dir", lambda: cache_dir)

    # Create an existing preset with one model
    config = read_preset()
    config.add_section("org/model-GGUF:Q4_K_M")
    write_preset(config)

    # Create manifests: one matching existing, two new
    manifest_json = '{"ggufFile": {"rfilename": "model.gguf", "size": 100}}'
    (cache_dir / "manifest=org=model-GGUF=Q4_K_M.json").write_text(manifest_json)
    (cache_dir / "manifest=org=model-GGUF=latest.json").write_text(manifest_json)
    (cache_dir / "manifest=other=newmodel-GGUF=Q8_0.json").write_text(manifest_json)
    # Old-format manifest (underscore) should be ignored
    (cache_dir / "manifest=org_oldmodel-GGUF=latest.json").write_text(manifest_json)

    added = sync_preset_with_cache()

    assert "org/model-GGUF" in added
    assert "other/newmodel-GGUF:Q8_0" in added
    assert len(added) == 2

    # Verify they're in the preset now
    config2 = read_preset()
    assert "org/model-GGUF" in config2.sections()
    assert "other/newmodel-GGUF:Q8_0" in config2.sections()


def test_sync_preset_no_duplicates(tmp_path, monkeypatch):
    preset_file = tmp_path / "models.ini"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr("llama_buddy.config.PRESET_FILE", preset_file)
    monkeypatch.setattr("llama_buddy.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("llama_buddy.config.get_cache_dir", lambda: cache_dir)

    manifest_json = '{"ggufFile": {"rfilename": "model.gguf", "size": 100}}'
    (cache_dir / "manifest=org=model-GGUF=Q4_K_M.json").write_text(manifest_json)

    # First sync adds it
    added1 = sync_preset_with_cache()
    assert len(added1) == 1

    # Second sync should find nothing new
    added2 = sync_preset_with_cache()
    assert len(added2) == 0
