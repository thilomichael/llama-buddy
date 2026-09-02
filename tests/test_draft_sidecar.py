"""Tests for MTP sidecars: GGUF draft heads shipped next to a model."""

from __future__ import annotations

import configparser
from pathlib import Path

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
    monkeypatch.setattr(models, "find_model_mtp_file", lambda mid: sidecar)
    monkeypatch.setattr(models, "has_flag", lambda flag: True)
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: True)

    assert models.sync_mtp_param("org/repo:Q8_0", preset) is True
    assert preset.get("org/repo:Q8_0", "spec-type") == "draft-mtp"
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == str(sidecar)


def test_sync_mtp_param_skips_a_head_that_does_not_load(
    monkeypatch, preset, tmp_path,
):
    """Enabling MTP against an unloadable head makes the model unloadable."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(
        models, "find_model_mtp_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    monkeypatch.setattr(models, "has_flag", lambda flag: True)
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: False)

    assert models.sync_mtp_param("org/repo:Q8_0", preset) is False
    assert preset.has_option("org/repo:Q8_0", "spec-type") is False
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_sync_mtp_param_skips_sidecar_on_old_build(
    monkeypatch, preset, tmp_path,
):
    """Without --spec-draft-model, writing the key would abort the server."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(
        models, "find_model_mtp_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    monkeypatch.setattr(models, "has_flag", lambda flag: False)
    monkeypatch.setattr(models.shutil, "which", lambda name: "/bin/llama-server")
    monkeypatch.setattr(
        models, "_probe_draft_model", lambda m, d: pytest.fail("probed"),
    )

    assert models.sync_mtp_param("org/repo:Q8_0", preset) is False
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_sync_mtp_param_prefers_in_file_head(monkeypatch, preset, tmp_path):
    """A head inside the model file needs no draft model path."""
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: True)
    monkeypatch.setattr(
        models, "find_model_mtp_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    assert models.sync_mtp_param("org/repo:Q8_0", preset) is True
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


# ---------------------------------------------------------------------------
# Load probe
# ---------------------------------------------------------------------------


def _fake_server(monkeypatch, lines, exits_after=None):
    """Stand in for llama-server: emit lines, then optionally exit."""
    state = {"terminated": False, "args": None}

    class FakeProc:
        def __init__(self):
            self._lines = iter(lines)
            self._reads = 0
            self.returncode = None
            self.stdout = self  # readline/close live on the same object

        def poll(self):
            if exits_after is not None and self._reads >= exits_after:
                self.returncode = 1
                return 1
            return None

        def readline(self):
            self._reads += 1
            return next(self._lines, "")

        def close(self):
            pass

        def terminate(self):
            state["terminated"] = True
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    proc = FakeProc()

    monkeypatch.setattr(models.shutil, "which", lambda name: "/bin/llama-server")
    monkeypatch.setattr(
        models.subprocess, "Popen",
        lambda args, **kw: (state.__setitem__("args", args), proc)[1],
    )
    return state, proc


MODEL = Path("/cache/Model-Q8_0.gguf")
DRAFT = Path("/cache/MTP/mtp-Model-Q8_0.gguf")


def test_probe_accepts_a_pair_that_starts_listening(monkeypatch):
    state, _ = _fake_server(monkeypatch, ["loading\n", "listening on http://x\n"])
    assert models._probe_draft_model(MODEL, DRAFT) is True
    # A probe that succeeds leaves a server running unless it is stopped.
    assert state["terminated"] is True


def test_probe_rejects_a_pair_that_exits(monkeypatch):
    _fake_server(monkeypatch, ["error loading model\n"], exits_after=1)
    assert models._probe_draft_model(MODEL, DRAFT) is False


def test_probe_loads_the_head_as_a_draft_of_the_model(monkeypatch):
    """A head on its own is rejected by builds that do support it, so only
    the real pair says whether MTP will work."""
    state, _ = _fake_server(monkeypatch, ["listening on http://x\n"])
    models._probe_draft_model(MODEL, DRAFT)
    args = state["args"]
    assert args[args.index("-m") + 1] == str(MODEL)
    assert args[args.index("-md") + 1] == str(DRAFT)
    assert args[args.index("--spec-type") + 1] == "draft-mtp"


def test_probe_runs_on_cpu_without_warmup(monkeypatch):
    """The probe must not claim VRAM or generate; it only answers 'loads?'."""
    state, _ = _fake_server(monkeypatch, ["listening on http://x\n"])
    models._probe_draft_model(MODEL, DRAFT)
    assert "--no-warmup" in state["args"]
    assert state["args"][state["args"].index("-ngl") + 1] == "0"
    assert state["args"][state["args"].index("-ngld") + 1] == "0"


def test_probe_without_llama_server(monkeypatch):
    monkeypatch.setattr(models.shutil, "which", lambda name: None)
    assert models._probe_draft_model(MODEL, DRAFT) is False


@pytest.fixture
def probe_cache(monkeypatch, tmp_path):
    """Isolate the probe cache and pin the binary and model it is keyed on."""
    model = tmp_path / "Model-Q8_0.gguf"
    model.write_bytes(b"m" * 10)
    monkeypatch.setattr(models, "find_model_gguf_files", lambda mid: [model])
    monkeypatch.setattr(models, "MTP_CACHE_FILE", tmp_path / "mtp.json")
    monkeypatch.setattr(models, "_mtp_cache", None)
    monkeypatch.setattr(models.shutil, "which", lambda name: "/bin/llama-server")
    monkeypatch.setattr(models, "has_flag", lambda flag: True)
    monkeypatch.setattr(models, "_binary_key", lambda b: "build-1")
    calls = []
    monkeypatch.setattr(
        models, "_probe_draft_model", lambda m, d: (calls.append(d), True)[1],
    )
    return calls


def test_draft_model_loads_is_cached(monkeypatch, tmp_path, probe_cache):
    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    sidecar.write_bytes(b"x" * 100)

    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is True
    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is True
    assert len(probe_cache) == 1

    # A re-download changes the size, so the answer is looked up again.
    sidecar.write_bytes(b"x" * 200)
    monkeypatch.setattr(models, "_mtp_cache", None)
    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is True
    assert len(probe_cache) == 2


def test_draft_model_loads_reprobes_after_a_llama_cpp_upgrade(
    monkeypatch, tmp_path, probe_cache,
):
    """Whether a head loads is a fact about the files and the build."""
    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    sidecar.write_bytes(b"x" * 100)
    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is True
    assert len(probe_cache) == 1

    monkeypatch.setattr(models, "_binary_key", lambda b: "build-2")
    monkeypatch.setattr(models, "_mtp_cache", None)
    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is True
    assert len(probe_cache) == 2


def test_draft_model_loads_on_missing_file(probe_cache, tmp_path):
    assert models.draft_model_loads("org/repo:Q8_0", tmp_path / "gone.gguf") is False


def test_draft_model_loads_without_llama_server(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "MTP_CACHE_FILE", tmp_path / "mtp.json")
    monkeypatch.setattr(models, "_mtp_cache", None)
    monkeypatch.setattr(models.shutil, "which", lambda name: None)
    assert models.draft_model_loads("org/repo:Q8_0", tmp_path / "mtp-M.gguf") is False


def test_draft_model_loads_skips_the_probe_on_an_old_build(
    monkeypatch, tmp_path, probe_cache,
):
    """A build without the flags cannot use the head; do not pay a model load."""
    monkeypatch.setattr(models, "has_flag", lambda flag: False)
    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    sidecar.write_bytes(b"x" * 100)
    assert models.draft_model_loads("org/repo:Q8_0", sidecar) is False
    assert probe_cache == []


def test_turning_mtp_off_drops_the_sidecar_path(preset):
    from llama_buddy.settings import _sync_draft_model_path

    preset.set("org/repo:Q8_0", "spec-type", "none")
    preset.set("org/repo:Q8_0", "spec-draft-model", "/cache/mtp-M-Q8_0.gguf")
    _sync_draft_model_path("org/repo:Q8_0", preset)
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_turning_mtp_back_on_restores_the_sidecar_path(
    monkeypatch, preset, tmp_path,
):
    from llama_buddy import config, settings

    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    monkeypatch.setattr(config, "find_model_mtp_file", lambda mid: sidecar)
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: True)

    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    settings._sync_draft_model_path("org/repo:Q8_0", preset)
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == str(sidecar)


def test_turning_mtp_on_leaves_an_unloadable_head_out(
    monkeypatch, preset, tmp_path,
):
    from llama_buddy import config, settings

    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    monkeypatch.setattr(config, "find_model_mtp_file", lambda mid: sidecar)
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: False)

    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    settings._sync_draft_model_path("org/repo:Q8_0", preset)
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False


def test_saving_with_the_path_already_set_does_not_probe(
    monkeypatch, preset, tmp_path,
):
    from llama_buddy import settings

    monkeypatch.setattr(
        models, "draft_model_loads", lambda mid, d: pytest.fail("probed"),
    )
    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    preset.set("org/repo:Q8_0", "spec-draft-model", "/cache/mtp-M-Q8_0.gguf")
    settings._sync_draft_model_path("org/repo:Q8_0", preset)
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == "/cache/mtp-M-Q8_0.gguf"


def test_mtp_cell_shows_sidecar_models(monkeypatch, preset, tmp_path):
    monkeypatch.setattr(models, "model_has_mtp", lambda mid: False)
    monkeypatch.setattr(
        models, "find_model_mtp_file", lambda mid: tmp_path / "mtp-M.gguf",
    )
    assert "off" in models._mtp_cell("org/repo:Q8_0", preset)

    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    assert "on" in models._mtp_cell("org/repo:Q8_0", preset)


def test_unwire_drops_a_sidecar_the_build_cannot_load(
    monkeypatch, preset, tmp_path,
):
    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    sidecar.write_bytes(b"x")
    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    preset.set("org/repo:Q8_0", "spec-draft-model", str(sidecar))
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: False)

    assert models.unwire_unloadable_sidecar("org/repo:Q8_0", preset) is True
    assert preset.has_option("org/repo:Q8_0", "spec-draft-model") is False
    # Gone entirely rather than set to none, so the next binary change can
    # switch it back on.
    assert preset.has_option("org/repo:Q8_0", "spec-type") is False


def test_unwire_keeps_a_sidecar_that_loads(monkeypatch, preset, tmp_path):
    sidecar = tmp_path / "mtp-M-Q8_0.gguf"
    sidecar.write_bytes(b"x")
    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    preset.set("org/repo:Q8_0", "spec-draft-model", str(sidecar))
    monkeypatch.setattr(models, "draft_model_loads", lambda mid, d: True)

    assert models.unwire_unloadable_sidecar("org/repo:Q8_0", preset) is False
    assert preset.get("org/repo:Q8_0", "spec-draft-model") == str(sidecar)


def test_unwire_drops_a_sidecar_whose_file_is_gone(monkeypatch, preset, tmp_path):
    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    preset.set("org/repo:Q8_0", "spec-draft-model", str(tmp_path / "gone.gguf"))
    monkeypatch.setattr(
        models, "draft_model_loads", lambda mid, d: pytest.fail("probed"),
    )
    assert models.unwire_unloadable_sidecar("org/repo:Q8_0", preset) is True


def test_unwire_ignores_in_file_mtp(preset):
    preset.set("org/repo:Q8_0", "spec-type", "draft-mtp")
    assert models.unwire_unloadable_sidecar("org/repo:Q8_0", preset) is False
    assert preset.get("org/repo:Q8_0", "spec-type") == "draft-mtp"


def test_find_model_mtp_file_ignores_other_draft_kinds(monkeypatch, tmp_path):
    """A dspark head wants --spec-type draft-dspark, never draft-mtp."""
    from llama_buddy import config

    dspark = tmp_path / "dspark-Model-Q8_0.gguf"
    mtp = tmp_path / "mtp-Model-Q8_0.gguf"
    monkeypatch.setattr(
        config, "find_model_draft_files", lambda mid: [dspark, mtp],
    )
    assert config.find_model_mtp_file("org/repo:Q8_0") == mtp
    monkeypatch.setattr(config, "find_model_draft_files", lambda mid: [dspark])
    assert config.find_model_mtp_file("org/repo:Q8_0") is None
    assert config.find_model_draft_file("org/repo:Q8_0") == dspark
