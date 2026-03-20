"""Model listing and management."""

from __future__ import annotations

import httpx

from llama_buddy.config import DEFAULT_PORT, get_cache_dir


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
        # Extract repo prefix: everything before the second underscore
        # e.g. "unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf"
        #    -> repo_key = "unsloth_gpt-oss-20b-GGUF"
        #    -> repo_id  = "unsloth/gpt-oss-20b-GGUF"
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
    # The repo key is org_repoName-GGUF — find the longest prefix
    # ending with -GGUF (before the next underscore)
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


def list_models(port: int = DEFAULT_PORT) -> None:
    try:
        models = get_models(port)
    except httpx.HTTPError:
        print("Error: Could not connect to llama-server. Is it running?")
        raise SystemExit(1)

    if not models:
        print("No models configured.")
        return

    sizes = compute_model_sizes()
    all_ids = {m["id"] for m in models}

    rows: list[tuple[str, str, str, str]] = []
    for m in models:
        model_id = m["id"]
        if is_bare_repo(model_id, all_ids):
            continue

        aliases = m.get("aliases", [])
        alias = aliases[0] if aliases else ""
        status_str = "loaded" if m.get("active_slot_count", 0) > 0 else "unloaded"

        # Match size: try full ID first, then base repo
        base_repo = model_id.split(":")[0]
        size = sizes.get(base_repo, 0)
        size_str = format_size(size) if size else "-"

        rows.append((model_id, alias, status_str, size_str))

    if not rows:
        print("No models configured.")
        return

    # Print table
    headers = ("MODEL", "ALIAS", "STATUS", "SIZE")
    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(4)
    ]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(val.ljust(w) for val, w in zip(row, col_widths)))
