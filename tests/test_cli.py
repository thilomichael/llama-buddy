"""Tests for CLI argument parsing."""

from __future__ import annotations

from llama_buddy.cli import build_parser


def test_no_command(capsys):
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_start_no_extra():
    parser = build_parser()
    args, remaining = parser.parse_known_args(["start"])
    assert args.command == "start"
    assert remaining == []


def test_start_with_extra():
    parser = build_parser()
    args, remaining = parser.parse_known_args(["start", "--port", "9090"])
    assert args.command == "start"
    assert remaining == ["--port", "9090"]


def test_stop():
    parser = build_parser()
    args = parser.parse_args(["stop"])
    assert args.command == "stop"


def test_status():
    parser = build_parser()
    args = parser.parse_args(["status"])
    assert args.command == "status"


def test_models():
    parser = build_parser()
    args = parser.parse_args(["models"])
    assert args.command == "models"


def test_download():
    parser = build_parser()
    args = parser.parse_args(["download", "org/model-GGUF:Q4_K_M"])
    assert args.command == "download"
    assert args.model == "org/model-GGUF:Q4_K_M"
    assert args.alias is None


def test_download_with_alias():
    parser = build_parser()
    args = parser.parse_args(
        ["download", "org/model-GGUF:Q4_K_M", "--alias", "mymodel"]
    )
    assert args.alias == "mymodel"


def test_remove():
    parser = build_parser()
    args = parser.parse_args(["remove", "mymodel"])
    assert args.command == "remove"
    assert args.model == "mymodel"
    assert args.delete_files is False


def test_remove_with_delete():
    parser = build_parser()
    args = parser.parse_args(["remove", "mymodel", "--delete-files"])
    assert args.delete_files is True


def test_info():
    parser = build_parser()
    args = parser.parse_args(["info", "org/model-GGUF"])
    assert args.command == "info"
    assert args.model == "org/model-GGUF"


def test_logs():
    parser = build_parser()
    args = parser.parse_args(["logs"])
    assert args.command == "logs"
