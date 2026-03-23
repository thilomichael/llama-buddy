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
    models_p = sub.add_parser("models", help="List configured models")
    models_p.add_argument(
        "--sort",
        choices=["name", "size"],
        default="name",
        help="Sort models by name (default) or size",
    )

    # download
    dl_p = sub.add_parser("download", help="Download a model")
    dl_p.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model ID (e.g. org/model-GGUF:Q4_K_M)",
    )
    dl_p.add_argument("--alias", help="Custom alias for the model")

    # remove
    rm_p = sub.add_parser("remove", help="Remove a model")
    rm_p.add_argument("model", nargs="?", help="Model ID or alias")
    rm_p.add_argument(
        "--keep-files", action="store_true", help="Keep cached .gguf files"
    )

    # info
    info_p = sub.add_parser("info", help="Show GGUF metadata for a model")
    info_p.add_argument(
        "model", nargs="?", help="Model ID, alias, or path to .gguf file"
    )
    info_p.add_argument(
        "--apply-sampling", action="store_true",
        help="Write GGUF sampling params into the preset INI",
    )

    # logs
    sub.add_parser("logs", help="Tail the server log")

    # open
    sub.add_parser("open", help="Open the llama-server web UI in the browser")

    # settings
    sub.add_parser("settings", help="Configure llama-server settings")

    # props
    props_p = sub.add_parser("props", help="Show active server sampling params")
    props_p.add_argument(
        "model", nargs="?", help="Model ID or alias"
    )

    # chat
    chat_p = sub.add_parser("chat", help="Interactive chat with a model")
    chat_p.add_argument("model", nargs="?", help="Model ID or alias")

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

        list_models(port=load_settings().port, sort=args.sort)

    elif args.command == "download":
        from llama_buddy.download import download
        from llama_buddy.server import restart_if_running

        download(args.model or None, args.alias)
        restart_if_running()

    elif args.command == "remove":
        from llama_buddy.download import remove
        from llama_buddy.server import restart_if_running

        remove(args.model or None, args.keep_files)
        restart_if_running()

    elif args.command == "info":
        if args.apply_sampling:
            from llama_buddy.info import apply_sampling
            from llama_buddy.server import restart_if_running

            apply_sampling(args.model or None)
            restart_if_running()
        else:
            model = args.model
            if model is None:
                from llama_buddy.select import select_model

                model = select_model()
                print()

            from llama_buddy.info import show_info

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
        from llama_buddy.server import restart_if_running
        from llama_buddy.settings import edit_settings

        edit_settings()
        restart_if_running()

    elif args.command == "props":
        from llama_buddy.props import show_props

        model = args.model
        if model is None:
            from llama_buddy.props import get_loaded_model_ids
            from llama_buddy.select import select_model
            from llama_buddy.settings import load_settings

            loaded = get_loaded_model_ids(load_settings().port)
            if not loaded:
                from llama_buddy.console import console

                console.print(
                    "No models are currently loaded.", style="yellow"
                )
                raise SystemExit(1)
            model = select_model(
                title="Select a loaded model", allowed_ids=loaded,
            )
            print()
        show_props(model)

    elif args.command == "chat":
        from llama_buddy.chat import chat

        chat(args.model or None, remaining or None)
