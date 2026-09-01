"""Tests for llama-server flag capability detection."""

from __future__ import annotations

import json

import pytest

from llama_buddy import capabilities
from llama_buddy.capabilities import (
    filter_supported,
    has_flag,
    has_value,
    parse_help,
    supported_choices,
    supported_flags,
)

# Shaped like real llama-server help: options at column 0, description and
# value lists indented into the second column.
HELP_TEXT = """
usage: llama-server [options]

--port PORT                             port to listen on
--sleep-idle-seconds N                  idle timeout
-lm,   --load-mode MODE                 model loading mode (default: mmap)
                                        - none: no special loading mode
                                        - mmap: memory-map model
                                        - dio: use DirectIO if available
-ngl,  --gpu-layers N                   layers to offload
"""


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        capabilities, "CAPABILITIES_FILE", tmp_path / "capabilities.json"
    )
    monkeypatch.setattr(capabilities, "ensure_config_dir", lambda: None)
    capabilities._capabilities.cache_clear()
    yield
    capabilities._capabilities.cache_clear()


def _install_binary(tmp_path, monkeypatch, help_text=HELP_TEXT):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(capabilities.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(capabilities, "_run_help", lambda b: help_text)
    return binary


def test_parse_help_collects_flags_and_values():
    flags, values = parse_help(HELP_TEXT)
    assert "--load-mode" in flags
    assert "--lazy-mode" not in flags
    assert values["--load-mode"] == ["none", "mmap", "dio"]
    # A flag whose help lists no values must not get an empty allow-list.
    assert "--port" not in values


def test_flags_parsed_from_help(tmp_path, monkeypatch):
    _install_binary(tmp_path, monkeypatch)
    flags = supported_flags()
    assert "--load-mode" in flags
    assert "--lazy-mode" not in flags


def test_missing_binary_assumes_everything_supported(monkeypatch):
    monkeypatch.setattr(capabilities.shutil, "which", lambda name: None)
    assert supported_flags() is None
    assert has_flag("--anything-at-all") is True
    assert has_value("--load-mode", "auto") is True


def test_probe_failure_assumes_everything_supported(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr(capabilities.shutil, "which", lambda n: str(binary))

    def boom(_binary):
        raise OSError("cannot execute")

    monkeypatch.setattr(capabilities, "_run_help", boom)
    assert supported_flags() is None
    assert has_flag("--load-mode") is True


def test_has_value_rejects_unknown_value(tmp_path, monkeypatch):
    """Build 10344 has --load-mode but not its 'auto' value."""
    _install_binary(tmp_path, monkeypatch)
    assert has_value("--load-mode", "mmap") is True
    assert has_value("--load-mode", "auto") is False
    # No value list in the help means no opinion.
    assert has_value("--port", "8080") is True


def test_supported_choices_narrows_the_list(tmp_path, monkeypatch):
    _install_binary(tmp_path, monkeypatch)
    choices = supported_choices(
        "--load-mode", ["auto", "none", "mmap", "dio"]
    )
    assert choices == ["none", "mmap", "dio"]


def test_supported_choices_keeps_list_when_nothing_matches(
    tmp_path, monkeypatch
):
    _install_binary(tmp_path, monkeypatch)
    assert supported_choices("--load-mode", ["auto"]) == ["auto"]


def test_result_is_cached_on_disk(tmp_path, monkeypatch):
    _install_binary(tmp_path, monkeypatch)
    supported_flags()

    cached = json.loads(capabilities.CAPABILITIES_FILE.read_text())
    assert "--load-mode" in cached["flags"]
    assert cached["values"]["--load-mode"] == ["none", "mmap", "dio"]
    assert cached["key"].startswith(str(tmp_path / "llama-server"))


def test_cache_invalidated_when_binary_changes(tmp_path, monkeypatch):
    binary = _install_binary(tmp_path, monkeypatch)
    assert has_value("--load-mode", "auto") is False

    # A new build accepts the value; a different size must be noticed.
    binary.write_text("#!/bin/sh\n# rebuilt\n")
    capabilities._capabilities.cache_clear()
    monkeypatch.setattr(
        capabilities,
        "_run_help",
        lambda b: HELP_TEXT.replace(
            "- none: no special loading mode",
            "- auto: mmap, unless a device does not support it",
        ),
    )
    assert has_value("--load-mode", "auto") is True


def test_filter_supported_drops_unknown_flags(tmp_path, monkeypatch):
    _install_binary(tmp_path, monkeypatch)
    meta = {
        "port": {"flag": "--port"},
        "lazy_mode": {"flag": "--lazy-mode"},
        "alias": {},
    }
    assert set(filter_supported(meta)) == {"port", "alias"}


def test_filter_supported_narrows_choices(tmp_path, monkeypatch):
    _install_binary(tmp_path, monkeypatch)
    meta = {
        "load_mode": {
            "flag": "--load-mode",
            "choices": ["auto", "mmap", "dio"],
        },
    }
    filtered = filter_supported(meta)
    assert filtered["load_mode"]["choices"] == ["mmap", "dio"]
    # The caller's dict must not be edited in place.
    assert meta["load_mode"]["choices"] == ["auto", "mmap", "dio"]
