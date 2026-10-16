"""Tests for the effective preset and applying changes to a running router."""

from __future__ import annotations

import httpx
import pytest

import llama_buddy.server as server
from llama_buddy.config import write_preset
from llama_buddy.ini import IniFile


@pytest.fixture
def native_ctx(monkeypatch):
    """Every model reports 32K native context; capabilities allow all."""
    import llama_buddy.models as models

    monkeypatch.setattr(
        models, "get_model_meta", lambda mid: {"context_length": "32,768"},
    )
    monkeypatch.setattr(
        "llama_buddy.config._get_manifest_model_ids",
        lambda: {"org/a-GGUF:Q4_K_M", "org/b-GGUF:Q8_0"},
    )
    monkeypatch.setattr("llama_buddy.capabilities.has_flag", lambda f: True)
    monkeypatch.setattr("llama_buddy.capabilities.has_value", lambda f, v: True)


def test_effective_preset_clamps_cached_models_without_sections(native_ctx):
    write_preset(IniFile("[*]\nc = 65536\n\n[org/b-GGUF:Q8_0]\nctx-size = 16384\n"))

    server._build_effective_preset()

    effective = IniFile(server.EFFECTIVE_PRESET_FILE.read_text())
    assert effective.get("org/a-GGUF:Q4_K_M", "c") == "32768"
    assert effective.get("org/b-GGUF:Q8_0", "ctx-size") == "16384"
    assert effective.get("*", "c") == "65536"


def test_effective_preset_is_a_copy_when_nothing_needs_clamping(native_ctx):
    text = "; mine\n[*]\nsleep-idle-seconds = 300\n"
    write_preset(IniFile(text))

    server._build_effective_preset()

    assert server.EFFECTIVE_PRESET_FILE.read_text() == text


class _Router:
    """Answers GET /models like a router, recording reload requests."""

    def __init__(self, models: list[dict] | None, fail_reload: bool = False):
        self.models = models
        self.fail_reload = fail_reload
        self.reloads = 0

    def get(self, url, params=None, timeout=None):
        if self.models is None:
            raise httpx.ConnectError("refused")
        request = httpx.Request("GET", url)
        if params == {"reload": 1}:
            self.reloads += 1
            status = 500 if self.fail_reload else 200
            return httpx.Response(status, json={}, request=request)
        return httpx.Response(200, json={"data": self.models}, request=request)


@pytest.fixture
def router_env(monkeypatch):
    restarts: list[bool] = []
    monkeypatch.setattr(server, "_build_effective_preset", lambda: None)
    monkeypatch.setattr(server, "restart_if_running", lambda: restarts.append(True))

    def setup(router: _Router, managed: bool):
        monkeypatch.setattr(server.httpx, "get", router.get)
        monkeypatch.setattr(server, "is_running", lambda: managed)
        return restarts

    return setup


def test_reload_on_a_current_router(router_env):
    router = _Router([{"id": "m", "source": "cache"}])
    restarts = router_env(router, managed=True)
    server.reload_if_running()
    assert router.reloads == 1
    assert restarts == []


def test_restart_when_router_predates_reload(router_env):
    router = _Router([{"id": "m"}])
    restarts = router_env(router, managed=True)
    server.reload_if_running()
    assert router.reloads == 0
    assert restarts == [True]


def test_restart_when_reload_fails(router_env):
    router = _Router([{"id": "m", "source": "preset"}], fail_reload=True)
    restarts = router_env(router, managed=True)
    server.reload_if_running()
    assert restarts == [True]


def test_reload_a_router_llb_did_not_start(router_env):
    router = _Router([{"id": "laguna", "source": "preset"}])
    restarts = router_env(router, managed=False)
    server.reload_if_running()
    assert router.reloads == 1
    assert restarts == []


def test_nothing_running(router_env):
    restarts = router_env(_Router(None), managed=False)
    server.reload_if_running()
    assert restarts == []
