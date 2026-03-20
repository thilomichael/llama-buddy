"""Tests for download utilities."""

from __future__ import annotations

from llama_buddy.download import auto_alias


def test_auto_alias_full():
    result = auto_alias("mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M")
    assert result == "ministral-3-3b-instruct-2512"


def test_auto_alias_no_quant():
    assert auto_alias("unsloth/gpt-oss-20b-GGUF") == "gpt-oss-20b"


def test_auto_alias_strips_it():
    assert auto_alias("org/Model-7B-it-GGUF:Q5_K_M") == "model-7b"


def test_auto_alias_strips_instruct():
    assert auto_alias("org/Model-7B-Instruct-GGUF:Q4_K_M") == "model-7b"
