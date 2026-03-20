"""Model listing and management."""

from __future__ import annotations

import httpx
from rich.console import Console
from rich.table import Table

from llama_buddy.config import DEFAULT_PORT, find_model_gguf_files, get_cache_dir
from llama_buddy.gguf import read_metadata

console = Console()


def get_models(port: int = DEFAULT_PORT) -> list[dict]:
    resp = httpx.get(f"http://localhost:{port}/models", timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", [])


def compute_model_sizes() -> dict[str, int]:
    """Map model repo IDs to total size of their .gguf files in cache.

    Cache files are flat: org_repo-GGUF_filename.gguf
    The repo prefix (org_repo-GGUF) maps to org/repo-GGUF with _ as separator.
    """
    sizes: dict[str, int] = {}
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return sizes

    for gguf_file in cache_dir.glob("*.gguf"):
        if "mmproj" in gguf_file.name:
            continue
        repo_key = _extract_repo_key(gguf_file.name)
        if repo_key is None:
            continue
        repo_id = repo_key.replace("_", "/", 1)
        sizes[repo_id] = sizes.get(repo_id, 0) + gguf_file.stat().st_size

    return sizes


def _extract_repo_key(filename: str) -> str | None:
    """Extract the 'org_repo' prefix from a cache filename.

    Filenames follow the pattern: org_repo_filename.gguf
    where repo itself may contain underscores (from quant subfolders).
    We match against known manifest files to find the boundary.
    """
    parts = filename.split("_")
    for i in range(2, len(parts) + 1):
        candidate = "_".join(parts[:i])
        if candidate.endswith("-GGUF"):
            return candidate
    return None


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f}G"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f}M"
    return f"{size_bytes / 1024:.1f}K"


def is_bare_repo(model_id: str, all_ids: set[str]) -> bool:
    """Check if a model ID is a bare repo that has quant-specific variants."""
    if ":" in model_id:
        return False
    return any(other.startswith(model_id + ":") for other in all_ids)


def get_model_name(model_id: str) -> str:
    """Read the model name from GGUF metadata."""
    files = find_model_gguf_files(model_id)
    if not files:
        return ""
    try:
        meta = read_metadata(files[0])
        return str(meta.get("general.name", ""))
    except (ValueError, OSError):
        return ""


def list_models(port: int = DEFAULT_PORT) -> None:
    try:
        models = get_models(port)
    except httpx.HTTPError:
        console.print(
            "Could not connect to llama-server. Is it running?", style="red"
        )
        raise SystemExit(1)

    if not models:
        console.print("No models configured.", style="yellow")
        return

    sizes = compute_model_sizes()
    all_ids = {m["id"] for m in models}

    table = Table(title="Models", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Alias", style="dim")
    table.add_column("Status")
    table.add_column("Size", justify="right")
    table.add_column("Model ID", style="dim")

    for m in models:
        model_id = m["id"]
        if is_bare_repo(model_id, all_ids):
            continue

        name = get_model_name(model_id) or model_id
        aliases = m.get("aliases", [])
        alias = aliases[0] if aliases else ""

        is_loaded = m.get("active_slot_count", 0) > 0
        status_str = "[green]loaded[/green]" if is_loaded else "[dim]unloaded[/dim]"

        base_repo = model_id.split(":")[0]
        size = sizes.get(base_repo, 0)
        size_str = format_size(size) if size else "-"

        table.add_row(name, alias, status_str, size_str, model_id)

    if table.row_count == 0:
        console.print("No models configured.", style="yellow")
        return

    console.print(table)
