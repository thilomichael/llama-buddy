"""Interactive chat via llama-cli."""

from __future__ import annotations

import shutil
import subprocess

from llama_buddy.config import (
    find_model_gguf_files,
    model_option,
    model_source,
    read_preset,
    sync_preset_with_cache,
)
from llama_buddy.console import console
from llama_buddy.select import select_model


def _build_settings_args(model_id: str) -> list[str]:
    """Build llama-cli args from the model's preset section and ``[*]``.

    llama-cli does not read the preset, so the settings that matter for a
    chat are passed on its command line.
    """
    preset = read_preset()
    args: list[str] = []

    ctx = model_option(preset, model_id, "c", "0")
    if ctx not in ("0", ""):
        args.extend(["-c", ctx])

    ngl = model_option(preset, model_id, "ngl", "auto")
    if ngl != "auto":
        args.extend(["-ngl", ngl])

    fa = model_option(preset, model_id, "fa", "auto")
    if fa != "auto":
        args.extend(["-fa", fa])

    return args


def chat(model_id: str | None = None, extra_args: list[str] | None = None) -> None:
    """Launch llama-cli in conversation mode for the given model."""
    llama_cli = shutil.which("llama-cli")
    if llama_cli is None:
        console.print(
            "llama-cli not found on PATH. Install llama.cpp first.",
            style="red",
        )
        raise SystemExit(1)

    sync_preset_with_cache()

    if model_id is None:
        model_id = select_model(title="Select a model to chat with")
        print()

    gguf_files = find_model_gguf_files(model_id)
    if gguf_files:
        cmd = [llama_cli, "-m", str(gguf_files[0]), "--conversation"]
    else:
        repo = model_source(model_id)[1]
        cmd = [llama_cli, "--hf-repo", repo, "--conversation"]

    settings_args = _build_settings_args(model_id)
    cmd.extend(settings_args)

    if extra_args:
        cmd.extend(extra_args)

    console.print(f"Starting chat with [bold]{model_id}[/bold]…")
    if settings_args:
        console.print(f"  Settings: [dim]{' '.join(settings_args)}[/dim]")
    if extra_args:
        console.print(f"  Extra args: [dim]{' '.join(extra_args)}[/dim]")
    print()
    subprocess.run(cmd)
