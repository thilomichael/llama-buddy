"""CLI entry point for llama-buddy (llb command)."""

from __future__ import annotations

import argparse
import subprocess

from llama_buddy.config import LOG_FILE, sync_preset_with_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llb",
        description="CLI wrapper for llama.cpp providing an ollama-like experience",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_get_version()}"
    )
    sub = parser.add_subparsers(dest="command")

    # start
    sub.add_parser("start", help="Start llama-server in the background")

    # stop
    sub.add_parser("stop", help="Stop the running llama-server")

    # restart
    sub.add_parser("restart", help="Restart the llama-server")

    # status
    sub.add_parser("status", help="Show server status")

    # models
    sub.add_parser("models", help="List configured models")

    # download
    dl_p = sub.add_parser("download", help="Download a model")
    dl_p.add_argument("model", help="HuggingFace model ID (e.g. org/model-GGUF:Q4_K_M)")
    dl_p.add_argument("--alias", help="Custom alias for the model")

    # remove
    rm_p = sub.add_parser("remove", help="Remove a model")
    rm_p.add_argument("model", help="Model ID or alias")
    rm_p.add_argument(
        "--delete-files", action="store_true", help="Also delete cached .gguf files"
    )

    # info
    info_p = sub.add_parser("info", help="Show GGUF metadata for a model")
    info_p.add_argument(
        "model", nargs="?", help="Model ID, alias, or path to .gguf file"
    )

    # logs
    sub.add_parser("logs", help="Tail the server log")

    # open
    sub.add_parser("open", help="Open the llama-server web UI in the browser")

    # settings
    sub.add_parser("settings", help="Configure llama-server settings")

    return parser


def _get_version() -> str:
    from llama_buddy import __version__

    return __version__


def main(argv: list[str] | None = None) -> None:
    try:
        _run(argv)
    except KeyboardInterrupt:
        raise SystemExit(0)


def _run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    sync_preset_with_cache()

    if args.command == "start":
        from llama_buddy.server import start

        start(remaining or None)

    elif args.command == "stop":
        from llama_buddy.server import stop

        stop()

    elif args.command == "restart":
        from llama_buddy.server import restart

        restart(remaining or None)

    elif args.command == "status":
        from llama_buddy.server import status

        status()

    elif args.command == "models":
        from llama_buddy.models import list_models
        from llama_buddy.settings import load_settings

        list_models(port=load_settings().port)

    elif args.command == "download":
        from llama_buddy.download import download

        download(args.model, args.alias)

    elif args.command == "remove":
        from llama_buddy.download import remove

        remove(args.model, args.delete_files)

    elif args.command == "info":
        from llama_buddy.info import show_info

        model = args.model
        if model is None:
            from llama_buddy.select import select_model

            model = select_model()
            print()
        show_info(model)

    elif args.command == "logs":
        if not LOG_FILE.exists():
            print(f"Log file not found: {LOG_FILE}")
            raise SystemExit(1)
        subprocess.run(["tail", "-f", str(LOG_FILE)])

    elif args.command == "open":
        import webbrowser

        from llama_buddy.settings import load_settings

        port = load_settings().port
        url = f"http://localhost:{port}"
        print(f"Opening {url}")
        webbrowser.open(url)

    elif args.command == "settings":
        from llama_buddy.settings import edit_settings

        edit_settings()
