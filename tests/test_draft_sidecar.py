"""Tests for MTP sidecars: GGUF draft heads shipped next to a model."""

from __future__ import annotations

import configparser

import pytest

from llama_buddy import download, models
from llama_buddy.config import is_draft_file, pick_draft_file


def test_is_draft_file_recognises_prefixes():
    assert is_draft_file("mtp-Qwen3.8-Flash-Next-Q8_0.gguf") is True
    assert is_draft_file("MTP/mtp-Model-Q4_K_M.gguf") is True
    assert is_draft_file("dspark-Model-Q8_0.gguf") is True
    assert is_draft_file("eagle3-Model-Q8_0.gguf") is True
    assert is_draft_file("Qwen3.8-Flash-Next-Q8_0.gguf") is False
    assert is_draft_file("MTP-Model-Q8_0.gguf") is True


def test_is_draft_file_ignores_mtp_in_the_middle():
    assert is_draft_file("Qwen3.6-27B-MTP-GGUF-Q8_0.gguf") is False


def test_pick_draft_file_prefers_matching_quant():
    paths = [
        "MTP/mtp-Model-Q4_K_M.gguf",
        "MTP/mtp-Model-Q8_0.gguf",
        "MTP/mtp-Model-BF16.gguf",
    ]
    assert pick_draft_file(paths, "Q8_0") == "MTP/mtp-Model-Q8_0.gguf"
    assert pick_draft_file(paths, "Q4_K_M") == "MTP/mtp-Model-Q4_K_M.gguf"


def test_pick_draft_file_prefers_standalone_over_shared():
    """A -shared- head leaves out weights it takes from the model."""
    paths = [
        "MTP/mtp-shared-Model-Q8_0.gguf",
        "MTP/mtp-Model-Q8_0.gguf",
    ]
    assert pick_draft_file(paths, "Q8_0") == "MTP/mtp-Model-Q8_0.gguf"


def test_pick_draft_file_falls_back_to_closest_bit_width():
    """Repos ship heads for a few quants only, so most models miss."""
    paths = [
        "MTP/mtp-Model-BF16.gguf",
        "MTP/mtp-Model-Q4_K_M.gguf",
        "MTP/mtp-Model-Q8_0.gguf",
    ]
    assert pick_draft_file(paths, "UD-Q4_K_XL") == "MTP/mtp-Model-Q4_K_M.gguf"
    assert pick_draft_file(paths, "UD-IQ1_S") == "MTP/mtp-Model-Q4_K_M.gguf"
    assert pick_draft_file(paths, "F16") == "MTP/mtp-Model-BF16.gguf"


def test_pick_draft_file_without_a_readable_quant():
    paths = ["MTP/mtp-Model-BF16.gguf", "MTP/mtp-Model-Q4_K_M.gguf"]
    assert pick_draft_file(paths, "latest") in paths


def test_pick_draft_file_on_empty_list():
    assert pick_draft_file([], "Q8_0") is None


# ---------------------------------------------------------------------------
# Repo file listing
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_listing(monkeypatch):
    """A repo that ships both quants and sidecars under MTP/."""
    files = [
        {"path": "Model-Q8_0.gguf", "size": 188_000_000_000},
        {"path": "Model-Q4_K_M.gguf", "size": 96_000_000_000},
        {"path": "MTP/mtp-Model-Q8_0.gguf", "size": 4_140_000_000},
        {"path": "MTP/mtp-shared-Model-Q8_0.gguf", "size": 2_790_000_000},
        {"path": "MTP/mtp-Model-Q4_K_M.gguf", "size": 2_100_000_000},
    ]
    monkeypatch.setattr(download, "_fetch_repo_files", lambda repo: files)
    return files


def test_get_repo_files_hides_sidecars(repo_listing):
    paths = [f["path"] for f in download._get_repo_files("org/repo")]
    assert paths == ["Model-Q8_0.gguf", "Model-Q4_K_M.gguf"]


def test_get_repo_mtp_files_returns_only_sidecars(repo_listing):
    paths = [f["path"] for f in download._get_repo_mtp_files("org/repo")]
    assert all(p.startswith("MTP/mtp-") for p in paths)
    assert len(paths) == 3


def test_fetch_repo_files_is_cached(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {
                "siblings": [{"rfilename": "Model-Q8_0.gguf", "size": 1}],
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            return FakeResponse()

    monkeypatch.setattr(download.httpx, "Client", FakeClient)
    monkeypatch.setattr(download, "_repo_file_cache", {})

    download._fetch_repo_files("org/repo")
    download._fetch_repo_files("org/repo")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Preset wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def preset():
    config = configparser.ConfigParser()
    config.add_section("org/repo:Q8_0")
    return config


def test_sync_mtp_param_writes_sidecar_path(monkeypatch, preset, tmp_path):
    sidecar = tmp_path / "mtp-Model-Q8_0.gguf"
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(models, "find_model_draft_file", lambda mid: sidecar)
    monkeypatch.setattr(models, "has_flag", lambda flag: True)

    assert models.sync_mtp_param("org/repo:Q8_0", preset) is True
    assert preset.get("org/repo:Q8_0", "spec-type") == "draft-mtp"
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == str(sidecar)


def test_sync_mtp_param_skips_sidecar_on_old_build(
    monkeypatch, preset, tmp_path,
):
    """Without --spec-draft-model, writing the key would abort the server."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(
        models, "find_model_draft_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    monkeypatch.setattr(models, "has_flag", lambda flag: False)

    assert models.sync_mtp_param("org/repo:Q8_0", preset) is False
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_sync_mtp_param_prefers_in_file_head(monkeypatch, preset, tmp_path):
    """A head inside the model file needs no draft model path."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    monkeypatch.setattr(
        models, "find_model_draft_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    assert models.sync_mtp_param("org/repo:Q8_0", preset) is True
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_turning_mtp_off_drops_the_sidecar_path(preset):
    from llama_buddy.settings import _sync_draft_model_path

    preset.set("org/repo:Q8_0", "spec-draft-model", "/cache/mtp-M-Q8_0.gguf")
    _sync_draft_model_path("org/repo:Q8_0", preset, "none")
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_turning_mtp_back_on_restores_the_sidecar_path(
    monkeypatch, preset, tmp_path,
):
    from llama_buddy import config, settings

    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    monkeypatch.setattr(config, "find_model_draft_file", lambda mid: sidecar)
    monkeypatch.setattr(settings, "has_flag", lambda flag: True)

    settings._sync_draft_model_path("org/repo:Q8_0", preset, "draft-mtp")
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == str(sidecar)


def test_mtp_cell_shows_sidecar_models(monkeypatch, preset, tmp_path):
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(
        models, "find_model_draft_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    assert "off" in models._mtp_cell("org/repo:Q8_0", preset)

    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    assert "on" in models._mtp_cell("org/repo:Q8_0", preset)
