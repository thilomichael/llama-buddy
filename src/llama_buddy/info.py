"""Model info display."""

from __future__ import annotations

from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from llama_buddy.config import find_model_gguf_files, resolve_model
from llama_buddy.console import console
from llama_buddy.gguf import read_metadata
from llama_buddy.models import format_size

SAMPLING_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
}


def find_gguf_files(name: str) -> list[Path]:
    """Resolve a model name, alias, ID, or file path to .gguf files."""
    p = Path(name)
    if p.exists() and p.suffix == ".gguf":
        return [p]

    model_id = resolve_model(name)
    if model_id is not None:
        return find_model_gguf_files(model_id)

    return []


def show_info(model_id_or_path: str) -> None:
    gguf_files = find_gguf_files(model_id_or_path)
    if not gguf_files:
        console.print(
            f"Could not find GGUF file for '{model_id_or_path}'.",
            style="red",
        )
        raise SystemExit(1)

    path = gguf_files[0]
    total_size = sum(f.stat().st_size for f in gguf_files)

    try:
        meta = read_metadata(path)
    except (ValueError, OSError) as e:
        console.print(f"Error reading metadata: {e}", style="red")
        return

    # Model name for the panel title
    model_name = str(meta.get("general.name", path.stem))

    # General info table
    general = Table(show_header=False, box=None, padding=(0, 2))
    general.add_column(style="bold")
    general.add_column()

    general.add_row(
        "File", str(path),
    )
    general.add_row(
        "Size", f"{format_size(total_size)} ({len(gguf_files)} file(s))",
    )

    for key in ("general.architecture", "general.size_label"):
        if key in meta:
            label = key.split(".")[-1].replace("_", " ").title()
            general.add_row(label, str(meta[key]))

    arch = meta.get("general.architecture", "")
    ctx_key = f"{arch}.context_length"
    if ctx_key in meta:
        general.add_row("Context Length", f"{meta[ctx_key]:,}")

    # Sampling table
    sampling = Table(show_header=False, box=None, padding=(0, 2))
    sampling.add_column(style="bold")
    sampling.add_column()

    sampling_keys = {
        k: v for k, v in meta.items() if k.startswith("general.sampling.")
    }
    if sampling_keys:
        for key, value in sorted(sampling_keys.items()):
            param = key.removeprefix("general.sampling.")
            sampling.add_row(param, str(value))
        sampling_title = "Sampling [dim](from GGUF metadata)[/dim]"
    else:
        for param, value in SAMPLING_DEFAULTS.items():
            sampling.add_row(param, str(value))
        sampling_title = "Sampling [dim](hardcoded defaults)[/dim]"

    # Compose output
    from rich.console import Group

    body = Group(
        general,
        "",
        Panel(sampling, title=sampling_title, border_style="dim", expand=False),
    )
    console.print(Panel(body, title=model_name, border_style="cyan"))
