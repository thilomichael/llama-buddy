"""Model download and removal."""

from __future__ import annotations

import shutil
import subprocess

from rich.console import Console

from llama_buddy.config import (
    find_model_gguf_files,
    read_preset,
    resolve_model,
    write_preset,
)

console = Console()


def download(model_id: str, alias: str | None = None) -> None:
    preset = read_preset()

    if model_id in preset:
        console.print(
            f"{model_id} is already in the preset file.", style="yellow"
        )
        return

    binary = shutil.which("llama-cli")
    if binary is None:
        console.print(
            "llama-cli not found. Install with: [bold]brew install llama.cpp[/bold]",
            style="red",
        )
        raise SystemExit(1)

    console.print(f"Downloading [bold]{model_id}[/bold]...")
    proc = subprocess.Popen(
        [binary, "-hf", model_id, "-n", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()

    if not preset.has_section("*"):
        preset.add_section("*")
        preset.set("*", "c", "0")

    preset.add_section(model_id)
    if alias is not None:
        preset.set(model_id, "alias", alias)
    write_preset(preset)

    msg = f"Added [bold]{model_id}[/bold]"
    if alias:
        msg += f" (alias: {alias})"
    msg += " to preset file."
    console.print(msg, style="green")


def remove(model_id_or_alias: str, delete_files: bool = False) -> None:
    section = resolve_model(model_id_or_alias)
    if section is None:
        console.print(
            f"Model '{model_id_or_alias}' not found in preset file.",
            style="red",
        )
        raise SystemExit(1)

    preset = read_preset()
    preset.remove_section(section)
    write_preset(preset)
    console.print(f"Removed [bold]{section}[/bold] from preset file.")

    if delete_files:
        for f in find_model_gguf_files(section):
            f.unlink()
            console.print(f"Deleted {f}", style="dim")
