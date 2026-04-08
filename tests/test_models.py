"""Tests for model utilities."""

from __future__ import annotations

from llama_buddy.models import (
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


def test_compute_model_sizes_hf_hub(tmp_path, monkeypatch):
    monkeypatch.setattr("llama_buddy.models.get_hf_hub_dir", lambda: tmp_path)

    # Create HF hub structure: models--org--model-GGUF/snapshots/abc123/
    snapshot = tmp_path / "models--org--model-GGUF" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    blob = tmp_path / "models--org--model-GGUF" / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 1000)
    (snapshot / "model-Q4_K_M.gguf").symlink_to("../../blobs/deadbeef")

    # mmproj should be excluded
    mmproj_blob = blob.parent / "cafebabe"
    mmproj_blob.write_bytes(b"x" * 500)
    (snapshot / "mmproj-F16.gguf").symlink_to("../../blobs/cafebabe")

    sizes = compute_model_sizes()
    assert sizes == {"org/model-GGUF": 1000}


def test_compute_model_sizes_multi_shard(tmp_path, monkeypatch):
    monkeypatch.setattr("llama_buddy.models.get_hf_hub_dir", lambda: tmp_path)

    snapshot = tmp_path / "models--org--big-model-GGUF" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "models--org--big-model-GGUF" / "blobs"
    blobs.mkdir(parents=True)

    for i in range(1, 4):
        blob = blobs / f"hash{i}"
        blob.write_bytes(b"x" * 1000)
        (snapshot / f"Q4_K_M_shard-{i:05d}.gguf").symlink_to(
            f"../../blobs/hash{i}"
        )

    sizes = compute_model_sizes()
    assert sizes == {"org/big-model-GGUF": 3000}


def test_format_mib_gigabytes():
    assert _format_mib(2048.0) == "2.0G"


def test_format_mib_megabytes():
    assert _format_mib(512.0) == "512M"
