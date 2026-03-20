"""Interactive chat via llama-cli."""

from __future__ import annotations

import shutil
import subprocess

from llama_buddy.config import find_model_gguf_files, sync_preset_with_cache
from llama_buddy.console import console
from llama_buddy.select import select_model


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
        cmd = [llama_cli, "--hf-repo", model_id, "--conversation"]

    if extra_args:
        cmd.extend(extra_args)

    console.print(f"Starting chat with [bold]{model_id}[/bold]…\n")
    subprocess.run(cmd)
