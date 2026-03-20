"""Model download and removal."""

from __future__ import annotations

import shutil
import subprocess

from llama_buddy.config import (
    find_model_gguf_files,
    read_preset,
    resolve_model,
    write_preset,
)


def auto_alias(model_id: str) -> str:
    """Generate a short alias from a model ID like 'org/Model-Name-GGUF:Q4_K_M'."""
    # Take just the model name, drop org and quant
    name = model_id.split("/")[-1].split(":")[0]
    # Strip common suffixes (may appear in any order, repeat until stable)
    changed = True
    while changed:
        changed = False
        for suffix in ("-GGUF", "-gguf", "-Instruct", "-instruct", "-it"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    return name.lower()


def download(model_id: str, alias: str | None = None) -> None:
    preset = read_preset()

    if model_id in preset:
        print(f"Model {model_id} is already in the preset file.")
        return

    binary = shutil.which("llama-cli")
    if binary is None:
        print("Error: llama-cli not found. Install it with: brew install llama.cpp")
        raise SystemExit(1)

    print(f"Downloading {model_id}...")
    # llama-cli -hf downloads the model then enters interactive mode.
    # We run it and kill after files appear in cache.
    proc = subprocess.Popen(
        [binary, "-hf", model_id, "-n", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()

    if alias is None:
        alias = auto_alias(model_id)

    if not preset.has_section("*"):
        preset.add_section("*")
        preset.set("*", "c", "0")

    preset.add_section(model_id)
    preset.set(model_id, "alias", alias)
    write_preset(preset)
    print(f"Added {model_id} (alias: {alias}) to preset file.")


def remove(model_id_or_alias: str, delete_files: bool = False) -> None:
    section = resolve_model(model_id_or_alias)
    if section is None:
        print(f"Model '{model_id_or_alias}' not found in preset file.")
        raise SystemExit(1)

    preset = read_preset()
    preset.remove_section(section)
    write_preset(preset)
    print(f"Removed {section} from preset file.")

    if delete_files:
        for f in find_model_gguf_files(section):
            f.unlink()
            print(f"Deleted {f}")
