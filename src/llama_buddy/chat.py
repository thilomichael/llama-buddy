"""Interactive chat via llama-cli."""

from __future__ import annotations

import shutil
import subprocess

from llama_buddy.config import find_model_gguf_files, read_preset, sync_preset_with_cache
from llama_buddy.console import console
from llama_buddy.select import select_model
from llama_buddy.settings import load_settings


def _build_settings_args(model_id: str) -> list[str]:
    """Build llama-cli args from global + per-model settings."""
    settings = load_settings()
    preset = read_preset()

    # Per-model overrides (from INI)
    ctx = preset.get(model_id, "c", fallback="0") if preset.has_section(model_id) else "0"
    ngl = preset.get(model_id, "ngl", fallback="auto") if preset.has_section(model_id) else "auto"
    fa = preset.get(model_id, "fa", fallback="auto") if preset.has_section(model_id) else "auto"

    args: list[str] = []

    # Context size: per-model override > global
    ctx_val = int(ctx) if ctx != "0" else settings.ctx_size
    if ctx_val > 0:
        args.extend(["-c", str(ctx_val)])

    # GPU layers: per-model override > global
    ngl_val = ngl if ngl != "auto" else settings.gpu_layers
    if ngl_val == "all":
        args.extend(["-ngl", "999"])
    elif ngl_val == "none":
        args.extend(["-ngl", "0"])
    elif ngl_val != "auto":
        args.extend(["-ngl", ngl_val])

    # Flash attention: per-model override > global
    fa_val = fa if fa != "auto" else settings.flash_attention
    if fa_val == "on":
        args.append("-fa")

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
        cmd = [llama_cli, "--hf-repo", model_id, "--conversation"]

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
