"""Model info display."""

from __future__ import annotations

from pathlib import Path

from llama_buddy.config import find_model_gguf_files, resolve_model
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
    # Direct path
    p = Path(name)
    if p.exists() and p.suffix == ".gguf":
        return [p]

    # Resolve via preset (alias or model ID)
    model_id = resolve_model(name)
    if model_id is not None:
        return find_model_gguf_files(model_id)

    return []


def show_info(model_id_or_path: str) -> None:
    gguf_files = find_gguf_files(model_id_or_path)
    if not gguf_files:
        print(f"Could not find GGUF file for '{model_id_or_path}'.")
        raise SystemExit(1)

    path = gguf_files[0]
    total_size = sum(f.stat().st_size for f in gguf_files)

    print(f"File:    {path}")
    print(f"Size:    {format_size(total_size)} ({len(gguf_files)} file(s))")
    print()

    try:
        meta = read_metadata(path)
    except (ValueError, OSError) as e:
        print(f"Error reading metadata: {e}")
        return

    # General metadata
    print("General:")
    for key in ("general.name", "general.architecture", "general.size_label"):
        if key in meta:
            label = key.split(".")[-1].replace("_", " ").title()
            print(f"  {label}: {meta[key]}")

    # Context length
    arch = meta.get("general.architecture", "")
    ctx_key = f"{arch}.context_length"
    if ctx_key in meta:
        print(f"  Context Length: {meta[ctx_key]}")

    # Sampling parameters
    print()
    sampling_keys = {k: v for k, v in meta.items() if k.startswith("general.sampling.")}
    if sampling_keys:
        print("Sampling (from GGUF metadata):")
        for key, value in sorted(sampling_keys.items()):
            param = key.removeprefix("general.sampling.")
            print(f"  {param}: {value}")
    else:
        print("Sampling (hardcoded defaults):")
        for param, value in SAMPLING_DEFAULTS.items():
            print(f"  {param}: {value}")
