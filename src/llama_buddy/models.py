"""Model listing and management."""

from __future__ import annotations

from functools import lru_cache

import httpx
from rich.table import Table

from llama_buddy.config import (
    DEFAULT_PORT,
    find_model_gguf_files,
    get_gguf_model_groups,
    get_hf_hub_dir,
    update_vram_usage,
)
from llama_buddy.console import console
from llama_buddy.gguf import read_metadata


def get_models(port: int = DEFAULT_PORT) -> list[dict]:
    resp = httpx.get(f"http://localhost:{port}/models", timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", [])


def compute_model_sizes() -> dict[str, int]:
    """Map model repo IDs to total size of their .gguf files in cache."""
    sizes: dict[str, int] = {}
    seen: set[str] = set()

    hf_dir = get_hf_hub_dir()
    if hf_dir.exists():
        for model_dir in hf_dir.glob("models--*--*-GGUF"):
            if not model_dir.is_dir():
                continue
            parts = model_dir.name.split("--", 2)
            if len(parts) < 3:
                continue
            repo_id = f"{parts[1]}/{parts[2]}"
            snapshots = model_dir / "snapshots"
            if not snapshots.exists():
                continue
            for f in snapshots.glob("**/*.gguf"):
                if "mmproj" in f.name:
                    continue
                resolved = str(f.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    sizes[repo_id] = sizes.get(repo_id, 0) + f.stat().st_size

    return sizes


def format_size(size_bytes: int, compact: bool = True) -> str:
    """Format a byte count as a human-readable string.

    compact=True:  '4.2G'  (used in tables)
    compact=False: '4.2 GB' (used in download/remove UI)
    """
    for threshold, suffix_c, suffix_f in (
        (1_073_741_824, "G", "GB"),
        (1_048_576, "M", "MB"),
        (1_024, "K", "KB"),
    ):
        if size_bytes >= threshold:
            val = size_bytes / threshold
            suffix = suffix_c if compact else suffix_f
            sep = "" if compact else " "
            return f"{val:.1f}{sep}{suffix}"
    suffix = "B" if compact else " B"
    return f"{size_bytes}{suffix}"


def is_bare_repo(model_id: str, all_ids: set[str]) -> bool:
    """Check if a model ID is a bare repo that has quant-specific variants."""
    if ":" in model_id:
        return False
    return any(other.startswith(model_id + ":") for other in all_ids)


@lru_cache(maxsize=64)
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


def _group_sort_key(
    group: list[str],
    sizes: dict[str, int],
    meta_cache: dict[str, dict[str, str]],
    sort: str,
) -> tuple:
    """Sort key for a GGUF group."""
    first = group[0]
    if sort == "size":
        base_repo = first.split(":")[0]
        return (-sizes.get(base_repo, 0),)
    # Default: sort by model name (from metadata), then model ID
    name = meta_cache.get(first, {}).get("name", first).lower()
    return (name,)


def _format_mib(mib: float) -> str:
    """Format a MiB value as a compact human-readable string (e.g. '47.6G')."""
    if mib >= 1024:
        return f"{mib / 1024:.1f}G"
    return f"{mib:.0f}M"


def _format_ctx(n: int) -> str:
    """Format context length compactly: 131072 -> '128K', 1048576 -> '1M'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}K"
    if n >= 1_024 and n % 1_024 == 0:
        return f"{n // 1_024}K"
    return f"{n:,}"


def _effective_ctx(
    model_id: str, native_ctx: str, preset, global_ctx: int, pad: int = 0,
) -> str:
    """Build context display string: 'effective (native)' or just 'native'."""
    if not native_ctx or native_ctx == "-":
        return "-"

    # Per-model override
    model_ctx = 0
    if preset.has_section(model_id):
        try:
            model_ctx = int(preset.get(model_id, "c", fallback="0"))
        except ValueError:
            pass

    effective = model_ctx or global_ctx
    native_int = int(native_ctx.replace(",", ""))
    eff_fmt = _format_ctx(effective) if effective > 0 else ""
    native_fmt = _format_ctx(native_int)
    if effective > 0 and eff_fmt != native_fmt:
        eff_str = eff_fmt.rjust(pad)
        native_paren = f"({native_fmt})".ljust(pad + 2)
        return f"{eff_str} [dim]{native_paren}[/dim]"
    return native_fmt


def list_models(port: int = DEFAULT_PORT, sort: str = "name") -> None:
    with console.status("Loading models…", spinner="dots"):
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

        # Filter internal sections
        models = [m for m in models if m["id"] != "DEFAULT"]

        sizes = compute_model_sizes()
        vram = update_vram_usage()
        gguf_groups = get_gguf_model_groups()
        model_map = {m["id"]: m for m in models}

        from llama_buddy.config import read_preset
        from llama_buddy.settings import load_settings

        preset = read_preset()
        global_ctx = load_settings().ctx_size

        # Pre-fetch metadata for sorting
        meta_cache: dict[str, dict[str, str]] = {}
        for group in gguf_groups:
            for mid in group:
                if mid in model_map:
                    meta_cache[mid] = get_model_meta(mid)

        # Compute alignment padding for context column
        ctx_pad = 0
        for mid, meta in meta_cache.items():
            native_str = meta.get("context_length", "")
            if native_str:
                native_int = int(native_str.replace(",", ""))
                ctx_pad = max(ctx_pad, len(_format_ctx(native_int)))
                model_ctx = 0
                if preset.has_section(mid):
                    try:
                        model_ctx = int(preset.get(mid, "c", fallback="0"))
                    except ValueError:
                        pass
                eff = model_ctx or global_ctx
                if eff > 0:
                    ctx_pad = max(ctx_pad, len(_format_ctx(eff)))

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
    table.add_column("VRAM", justify="right")
    table.add_column("Model ID", style="dim")

    def add_model_row(model_id: str, prefix: str = "") -> None:
        m = model_map.get(model_id)
        if m is None:
            return
        meta = meta_cache.get(model_id) or get_model_meta(model_id)
        name = meta.get("name", model_id)
        native_ctx = meta.get("context_length", "-")
        ctx_str = _effective_ctx(model_id, native_ctx, preset, global_ctx, ctx_pad)
        aliases = m.get("aliases", [])
        alias = aliases[0] if aliases else ""
        model_status = m.get("status", {})
        status_val = model_status.get("value", "") if isinstance(model_status, dict) else ""
        is_loaded = status_val == "loaded"
        status_str = "[green]loaded[/green]" if is_loaded else "[dim]unloaded[/dim]"
        base_repo = model_id.split(":")[0]
        size = sizes.get(base_repo, 0)
        size_str = format_size(size) if size else "-"
        vram_mib = vram.get(model_id, 0.0)
        vram_str = _format_mib(vram_mib) if vram_mib else "-"
        display_name = f"{prefix}{name}" if prefix else name
        table.add_row(
            display_name, alias, status_str, ctx_str,
            size_str, vram_str, model_id,
        )

    # Render grouped models with tree structure, then ungrouped
    sorted_groups = sorted(
        gguf_groups,
        key=lambda g: _group_sort_key(g, sizes, meta_cache, sort),
    )
    for group in sorted_groups:
        present = [mid for mid in group if mid in model_map]
        if not present:
            continue
        if len(present) == 1:
            add_model_row(present[0])
        else:
            # Group header row
            meta = meta_cache.get(present[0], {})
            shared_name = meta.get("name", present[0].split(":")[0])
            base_repo = present[0].split(":")[0]
            size = sizes.get(base_repo, 0)
            size_str = format_size(size) if size else "-"
            table.add_row(
                f"[bold]{shared_name}[/bold]",
                "", "", "", size_str, "", "",
            )
            for i, mid in enumerate(present):
                is_last = i == len(present) - 1
                prefix = "└─ " if is_last else "├─ "
                add_model_row(mid, prefix)

    # Add any models from the API that weren't in any manifest group
    ungrouped = [
        m for m in models
        if m["id"] not in grouped_ids
        and not any(m["id"] in g for g in gguf_groups)
    ]
    for m in sorted(
        ungrouped,
        key=lambda m: _group_sort_key(
            [m["id"]], sizes, meta_cache, sort
        ),
    ):
        add_model_row(m["id"])

    if table.row_count == 0:
        console.print("No models configured.", style="yellow")
        return

    console.print(table)
