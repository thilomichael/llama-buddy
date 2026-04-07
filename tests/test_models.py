"""Tests for model utilities."""

from __future__ import annotations

from llama_buddy.models import (
    _extract_repo_key,
    _format_mib,
    compute_model_sizes,
    format_size,
    is_bare_repo,
)


def test_format_size_gigabytes():
    assert format_size(2_147_483_648) == "2.0G"


def test_format_size_megabytes():
    assert format_size(524_288_000) == "500.0M"


def test_format_size_kilobytes():
    assert format_size(1024) == "1.0K"


def test_is_bare_repo_true():
    all_ids = {"org/model-GGUF", "org/model-GGUF:Q4_K_M", "org/model-GGUF:Q5_K_M"}
    assert is_bare_repo("org/model-GGUF", all_ids) is True


def test_is_bare_repo_false_has_quant():
    all_ids = {"org/model-GGUF:Q4_K_M"}
    assert is_bare_repo("org/model-GGUF:Q4_K_M", all_ids) is False


def test_is_bare_repo_false_no_variants():
    all_ids = {"org/model-GGUF"}
    assert is_bare_repo("org/model-GGUF", all_ids) is False


def test_extract_repo_key_simple():
    name = "unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf"
    assert _extract_repo_key(name) == "unsloth_gpt-oss-20b-GGUF"


def test_extract_repo_key_with_org_underscore():
    name = "mistralai_Ministral-3-3B-Instruct-2512-GGUF_file.gguf"
    assert _extract_repo_key(name) == "mistralai_Ministral-3-3B-Instruct-2512-GGUF"


def test_extract_repo_key_no_gguf():
    assert _extract_repo_key("random_file.gguf") is None


def test_compute_model_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr("llama_buddy.models.get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("llama_buddy.models.get_hf_hub_dir", lambda: tmp_path / "no_hf")

    # Create fake cache files
    f1 = tmp_path / "org_model-GGUF_model-Q4_K_M.gguf"
    f1.write_bytes(b"x" * 1000)
    f2 = tmp_path / "org_model-GGUF_mmproj-F16.gguf"
    f2.write_bytes(b"x" * 500)  # should be excluded

    sizes = compute_model_sizes()
    assert sizes == {"org/model-GGUF": 1000}


def test_compute_model_sizes_multi_shard(tmp_path, monkeypatch):
    monkeypatch.setattr("llama_buddy.models.get_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("llama_buddy.models.get_hf_hub_dir", lambda: tmp_path / "no_hf")

    for i in range(1, 4):
        f = tmp_path / f"org_big-model-GGUF_Q4_K_M_shard-{i:05d}.gguf"
        f.write_bytes(b"x" * 1000)

    sizes = compute_model_sizes()
    assert sizes == {"org/big-model-GGUF": 3000}


def test_format_mib_gigabytes():
    assert _format_mib(2048.0) == "2.0G"


def test_format_mib_megabytes():
    assert _format_mib(512.0) == "512M"
