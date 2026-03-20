"""Model listing and management."""

from __future__ import annotations

import httpx
from rich.console import Console
from rich.table import Table

from llama_buddy.config import (
    DEFAULT_PORT,
    find_model_gguf_files,
    get_cache_dir,
    get_gguf_model_groups,
)
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


def get_model_meta(model_id: str) -> dict[str, str]:
    """Read model name and context length from GGUF metadata."""
    files = find_model_gguf_files(model_id)
    if not files:
        return {}
    try:
        meta = read_metadata(files[0])
        result: dict[str, str] = {}
        name = meta.get("general.name")
        if name:
            result["name"] = str(name)
        arch = meta.get("general.architecture", "")
        ctx = meta.get(f"{arch}.context_length")
        if ctx is not None:
            result["context_length"] = f"{int(ctx):,}"
        return result
    except (ValueError, OSError):
        return {}


def get_model_name(model_id: str) -> str:
    """Read the model name from GGUF metadata."""
    return get_model_meta(model_id).get("name", "")


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
    gguf_groups = get_gguf_model_groups()
    model_map = {m["id"]: m for m in models}

    # Build ordered list: group models that share a GGUF, keep ungrouped ones
    grouped_ids: set[str] = set()
    for group in gguf_groups:
        if len(group) > 1:
            for mid in group:
                grouped_ids.add(mid)

    table = Table(title="Models", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Alias", style="dim")
    table.add_column("Status")
    table.add_column("Context", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Model ID", style="dim")

    def add_model_row(model_id: str, prefix: str = "") -> None:
        m = model_map.get(model_id)
        if m is None:
            return
        meta = get_model_meta(model_id)
        name = meta.get("name", model_id)
        ctx_str = meta.get("context_length", "-")
        aliases = m.get("aliases", [])
        alias = aliases[0] if aliases else ""
        is_loaded = m.get("active_slot_count", 0) > 0
        status_str = "[green]loaded[/green]" if is_loaded else "[dim]unloaded[/dim]"
        base_repo = model_id.split(":")[0]
        size = sizes.get(base_repo, 0)
        size_str = format_size(size) if size else "-"
        display_name = f"{prefix}{name}" if prefix else name
        table.add_row(display_name, alias, status_str, ctx_str, size_str, model_id)

    # Render grouped models with tree structure, then ungrouped
    for group in sorted(gguf_groups, key=lambda g: g[0].lower()):
        present = [mid for mid in group if mid in model_map]
        if not present:
            continue
        if len(present) == 1:
            add_model_row(present[0])
        else:
            # Group header row
            meta = get_model_meta(present[0])
            shared_name = meta.get("name", present[0].split(":")[0])
            base_repo = present[0].split(":")[0]
            size = sizes.get(base_repo, 0)
            size_str = format_size(size) if size else "-"
            table.add_row(
                f"[bold]{shared_name}[/bold]",
                "", "", "", size_str, "",
            )
            for i, mid in enumerate(present):
                is_last = i == len(present) - 1
                prefix = "└─ " if is_last else "├─ "
                add_model_row(mid, prefix)

    # Add any models from the API that weren't in any manifest group
    for m in models:
        if m["id"] not in grouped_ids and not any(
            m["id"] in g for g in gguf_groups
        ):
            add_model_row(m["id"])

    if table.row_count == 0:
        console.print("No models configured.", style="yellow")
        return

    console.print(table)
