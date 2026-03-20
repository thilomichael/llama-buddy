"""Model download and removal."""

from __future__ import annotations

import json
import re
import threading
import time
import webbrowser
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

from llama_buddy.config import (
    find_model_gguf_files,
    get_cache_dir,
    read_preset,
    resolve_model,
    write_preset,
)
from llama_buddy.tui import read_key, read_key_timeout, require_tty

console = Console()

HF_API = "https://huggingface.co/api/models"


def _compact_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _file_size(entry: dict) -> int:
    return entry.get("size", 0) or entry.get("lfs", {}).get("size", 0)


# ---------------------------------------------------------------------------
# HuggingFace API helpers
# ---------------------------------------------------------------------------


def _search_hf(query: str, limit: int = 20) -> list[dict]:
    """Search HuggingFace for GGUF models.

    Results are sorted by downloads, with exact substring matches on the
    model ID promoted to the top (both halves stay sorted by downloads).
    """
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            HF_API,
            params={
                "filter": "gguf",
                "search": query,
                "sort": "downloads",
                "direction": "-1",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        results = resp.json()

    q_lower = query.lower()
    exact: list[dict] = []
    rest: list[dict] = []
    for entry in results:
        if q_lower in entry["id"].lower():
            exact.append(entry)
        else:
            rest.append(entry)
    return exact + rest


def _get_repo_files(repo_id: str) -> list[dict]:
    """Get GGUF files in a HF repo with sizes (excludes mmproj)."""
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        resp = client.get(f"{HF_API}/{repo_id}/tree/main")
        resp.raise_for_status()
        files = resp.json()
    return [
        f
        for f in files
        if f.get("path", "").endswith(".gguf")
        and "mmproj" not in f.get("path", "").lower()
    ]


# ---------------------------------------------------------------------------
# Partial download detection
# ---------------------------------------------------------------------------


def _find_partial_downloads() -> list[dict]:
    """Scan cache manifests for files that are missing or incomplete.

    Returns a list of dicts with keys: model_id, repo_id, filename, quant,
    size, downloaded.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return []

    partials: list[dict] = []
    for manifest_path in sorted(cache_dir.glob("manifest=*.json")):
        parts = manifest_path.stem.split("=")
        if len(parts) != 4:
            continue
        _, org, repo, quant = parts
        try:
            data = json.loads(manifest_path.read_text())
            gguf_info = data.get("ggufFile", {})
            filename = gguf_info.get("rfilename", "")
            size = gguf_info.get("size", 0)
        except (json.JSONDecodeError, OSError):
            continue
        if not filename or not size:
            continue

        cache_filename = f"{org}_{repo}_{filename.replace('/', '_')}"
        dest = cache_dir / cache_filename
        downloaded = dest.stat().st_size if dest.exists() else 0

        if downloaded < size:
            model_id = (
                f"{org}/{repo}:{quant}"
                if quant.lower() != "latest"
                else f"{org}/{repo}"
            )
            partials.append(
                {
                    "model_id": model_id,
                    "repo_id": f"{org}/{repo}",
                    "filename": filename,
                    "quant": quant,
                    "size": size,
                    "downloaded": downloaded,
                }
            )
    return partials


# ---------------------------------------------------------------------------
# TUI: repo picker
# ---------------------------------------------------------------------------


def _render_repo_menu(
    entries: list[dict],
    selected: int,
    query: str,
    *,
    partials: list[dict] | None = None,
    searching: bool = False,
    searched_query: str = "",
) -> Text:
    text = Text()
    text.append("  Search HuggingFace GGUF models\n\n", style="bold")

    # Show partial downloads when there's no query and no search results
    if not entries and not query and partials:
        text.append("  Incomplete downloads:\n\n", style="bold yellow")
        for i, p in enumerate(partials):
            pct = p["downloaded"] / p["size"] * 100 if p["size"] else 0
            size_str = _human_size(p["size"])
            progress_str = f"{pct:.0f}% of {size_str}"
            is_sel = i == selected
            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(p["model_id"], style="bold cyan")
                text.append(f"  ({progress_str})", style="cyan")
            else:
                text.append(f"    {p['model_id']}", style="yellow")
                text.append(f"  ({progress_str})", style="yellow dim")
            text.append("\n")
        text.append("\n")
    elif not entries:
        if query and searching:
            text.append("    Searching…\n", style="dim italic")
        elif query and searched_query == query:
            text.append("    No results\n", style="dim italic")
        elif not query:
            text.append(
                "    Type to search\n", style="dim italic"
            )
    else:
        for i, entry in enumerate(entries):
            model_id = entry["id"]
            downloads = entry.get("downloads", 0)
            likes = entry.get("likes", 0)
            dl_str = f"↓{_compact_number(downloads)}"
            like_str = f"♥{likes}"
            is_sel = i == selected

            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(model_id, style="bold cyan")
                text.append(f"  {dl_str}  {like_str}", style="cyan dim")
            else:
                text.append(f"    {model_id}", style="dim")
                text.append(f"  {dl_str}  {like_str}", style="dim italic")
            text.append("\n")

    text.append("\n")
    text.append("  Search: ", style="dim italic")
    text.append(query, style="bold")
    text.append("_", style="bold blink")
    show_nav = entries or (not query and partials)
    if show_nav:
        text.append(
            "  ↑/↓ navigate  Enter select  Ctrl-O open  Esc clear",
            style="dim italic",
        )
    return text


_DEBOUNCE_SECS = 0.3


def _pick_repo_interactive() -> str | dict:
    """Interactive search and selection of a HF GGUF repo.

    Returns a repo ID string (from search) or a partial download dict
    (to resume).  Searches HuggingFace as you type with debouncing.
    """
    require_tty()
    query = ""
    entries: list[dict] = []
    partials = _find_partial_downloads()
    selected = 0

    # Background search state
    searched_query = ""
    searching = False
    last_type_time = 0.0
    result_lock = threading.Lock()
    pending_results: list[dict] | None = None
    pending_for_query = ""

    def _bg_search(q: str) -> None:
        nonlocal pending_results, pending_for_query
        try:
            results = _search_hf(q)
        except httpx.HTTPError:
            results = []
        with result_lock:
            pending_results = results
            pending_for_query = q

    def _render() -> Text:
        return _render_repo_menu(
            entries,
            selected,
            query,
            partials=partials,
            searching=searching,
            searched_query=searched_query,
        )

    with Live(
        _render(),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            # Pick up completed search results
            with result_lock:
                if pending_results is not None:
                    entries = pending_results
                    searched_query = pending_for_query
                    searching = False
                    pending_results = None
                    selected = min(selected, max(len(entries) - 1, 0))

            # Trigger debounced search (min 3 chars)
            if (
                len(query) >= 3
                and query != searched_query
                and not searching
                and time.monotonic() - last_type_time >= _DEBOUNCE_SECS
            ):
                searching = True
                threading.Thread(
                    target=_bg_search, args=(query,), daemon=True
                ).start()

            key = read_key_timeout(0.1)
            if key is None:
                # No keypress — just refresh (picks up search results)
                live.update(_render(), refresh=True)
                continue

            # Determine which list is active
            showing_partials = not entries and not query and bool(partials)
            active_len = (
                len(partials) if showing_partials else len(entries)
            )

            if key == "up" and selected > 0:
                selected -= 1
            elif key == "down" and selected < active_len - 1:
                selected += 1
            elif key in ("enter", "right"):
                if showing_partials:
                    return partials[selected]
                elif entries and active_len > 0:
                    return entries[selected]["id"]
            elif key == "ctrl-o":
                if showing_partials and partials:
                    repo_id = partials[selected]["repo_id"]
                    webbrowser.open(f"https://huggingface.co/{repo_id}")
                elif entries:
                    webbrowser.open(
                        f"https://huggingface.co/{entries[selected]['id']}"
                    )
            elif key == "ctrl-c":
                raise SystemExit(0)
            elif key == "\x1b":
                query = ""
                entries = []
                searched_query = ""
                searching = False
                selected = 0
            elif key == "backspace":
                if query:
                    query = query[:-1]
                    last_type_time = time.monotonic()
                    if not query:
                        entries = []
                        searched_query = ""
                        searching = False
                    elif len(query) < 3:
                        searching = False
            elif len(key) == 1 and key.isprintable():
                query += key
                last_type_time = time.monotonic()
                selected = 0
            else:
                continue

            live.update(_render(), refresh=True)


# ---------------------------------------------------------------------------
# TUI: quant file picker
# ---------------------------------------------------------------------------


_BIT_GROUP_LABELS = {
    16: "16-bit (full)",
    8: "8-bit",
    6: "6-bit",
    5: "5-bit",
    4: "4-bit",
    3: "3-bit",
    2: "2-bit",
    1: "1-bit",
    0: "Other",
}


def _quant_bit_group(filename: str) -> int:
    """Determine the bit-group for a GGUF filename from its quant tag."""
    basename = filename.rsplit("/", 1)[-1].upper()
    # Full / half precision
    if "F32" in basename or "F16" in basename or "BF16" in basename:
        return 16
    # Match Q/IQ prefix followed by digit
    m = re.search(r"(?:IQ|Q)(\d)", basename)
    if m:
        return int(m.group(1))
    return 0


def _group_files(files: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group files by bit depth, sorted high-to-low bits.

    Within each group, files keep their original order (typically by name).
    Returns list of (group_label, files) tuples.
    """
    groups: dict[int, list[dict]] = {}
    for f in files:
        bits = _quant_bit_group(f["path"])
        groups.setdefault(bits, []).append(f)

    # Sort groups: high bits first, "Other" (0) last
    order = sorted(groups.keys(), key=lambda b: (-b if b > 0 else 1))
    return [(_BIT_GROUP_LABELS.get(bits, "Other"), groups[bits]) for bits in order]


def _render_file_menu(
    grouped: list[tuple[str, list[dict]]],
    flat_files: list[dict],
    selected: int,
    repo_id: str,
) -> Text:
    text = Text()
    text.append(f"  Select quantization — {repo_id}\n\n", style="bold")

    idx = 0
    for label, group_files in grouped:
        text.append(f"  {label}\n", style="bold dim")
        for f in group_files:
            path = f["path"]
            size = _human_size(_file_size(f))
            is_sel = idx == selected

            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(path, style="bold cyan")
                text.append(f"  ({size})", style="cyan")
            else:
                text.append(f"    {path}", style="dim")
                text.append(f"  ({size})", style="dim italic")
            text.append("\n")
            idx += 1
        text.append("\n")

    text.append(
        "  ↑/↓ navigate  Enter select  Ctrl-O open  Ctrl-C cancel",
        style="dim italic",
    )
    return text


def _pick_quant_interactive(repo_id: str, files: list[dict]) -> dict:
    """Interactive selection of a GGUF file from a repo."""
    require_tty()
    grouped = _group_files(files)
    # Flat list in display order for selection indexing
    flat = [f for _, group_files in grouped for f in group_files]

    # Auto-select Q4_K_M if available
    selected = 0
    for i, f in enumerate(flat):
        if "Q4_K_M" in f["path"].upper():
            selected = i
            break

    with Live(
        _render_file_menu(grouped, flat, selected, repo_id),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = read_key()
            if key == "up" and selected > 0:
                selected -= 1
            elif key == "down" and selected < len(flat) - 1:
                selected += 1
            elif key in ("enter", "right"):
                return flat[selected]
            elif key == "ctrl-o":
                webbrowser.open(f"https://huggingface.co/{repo_id}")
            elif key == "ctrl-c":
                raise SystemExit(0)
            else:
                continue

            live.update(
                _render_file_menu(grouped, flat, selected, repo_id),
                refresh=True,
            )


# ---------------------------------------------------------------------------
# Quant extraction
# ---------------------------------------------------------------------------


def _extract_quant(filename: str) -> str:
    """Extract quant tag from a GGUF filename.

    E.g. 'Model-Q4_K_M.gguf' -> 'Q4_K_M', 'model-f16.gguf' -> 'f16'.
    """
    basename = filename.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    # Strip shard suffix (-00001-of-00003)
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    match = re.search(r"[-_]((?:IQ|Q|F|BF)\d\w*)$", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    return stem


# ---------------------------------------------------------------------------
# Download + cache management
# ---------------------------------------------------------------------------


def _download_gguf(
    repo_id: str, filename: str, dest: Path, size: int
) -> str | None:
    """Download a GGUF file with a Rich progress bar and resume support.

    If *dest* already exists and is smaller than *size*, resumes from where
    it left off using an HTTP Range request.  Returns the ETag.
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    existing = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with httpx.Client(
        timeout=httpx.Timeout(connect=15, read=30, write=30, pool=15),
        follow_redirects=True,
    ) as client:
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            etag = resp.headers.get("etag")

            if resp.status_code == 206:
                # Partial content — resume
                remaining = int(
                    resp.headers.get("content-length", size - existing)
                )
                total = existing + remaining
                mode = "ab"
            else:
                # Full response (server ignored Range, or fresh download)
                total = int(resp.headers.get("content-length", size))
                existing = 0
                mode = "wb"

            with Progress(
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    filename, total=total, completed=existing
                )
                with open(dest, mode) as f:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    return etag


def _create_manifest(
    cache_dir: Path,
    org: str,
    repo: str,
    quant: str,
    filename: str,
    size: int,
) -> None:
    """Create a manifest JSON so llama-server recognises the cached model."""
    manifest_name = f"manifest={org}={repo}={quant}.json"
    manifest = {
        "ggufFile": {
            "rfilename": filename,
            "size": size,
        }
    }
    (cache_dir / manifest_name).write_text(json.dumps(manifest, indent=2))


def _resolve_download_target(
    model_id: str,
) -> tuple[str, str, str, int]:
    """Resolve a model_id to (repo_id, filename, quant, size).

    For interactive mode (model_id is None), call the pickers instead.
    """
    if ":" in model_id:
        base, quant = model_id.rsplit(":", 1)
    else:
        base = model_id
        quant = None

    repo_id = base
    console.print(f"Fetching files for [bold]{repo_id}[/bold]…")

    try:
        files = _get_repo_files(repo_id)
    except httpx.HTTPError as e:
        console.print(f"Failed to fetch repo files: {e}", style="red")
        raise SystemExit(1)

    if not files:
        console.print("No GGUF files found in this repo.", style="red")
        raise SystemExit(1)

    if quant is None:
        if len(files) == 1:
            chosen = files[0]
        else:
            chosen = _pick_quant_interactive(repo_id, files)
        quant = _extract_quant(chosen["path"])
    else:
        q_lower = quant.lower()
        matches = [f for f in files if q_lower in f["path"].lower()]
        if not matches:
            console.print(
                f"No GGUF file matching '{quant}' in {repo_id}.", style="red"
            )
            raise SystemExit(1)
        chosen = matches[0]

    return repo_id, chosen["path"], quant, _file_size(chosen)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download(model_id: str | None = None, alias: str | None = None) -> None:
    """Download a model — interactively or by explicit ID."""
    if model_id is None:
        result = _pick_repo_interactive()
        if isinstance(result, dict):
            # Resuming a partial download
            repo_id = result["repo_id"]
            filename = result["filename"]
            quant = result["quant"]
            size = result["size"]
            model_id = result["model_id"]
        else:
            repo_id = result
            console.print(f"\nFetching files for [bold]{repo_id}[/bold]…")
            try:
                files = _get_repo_files(repo_id)
            except httpx.HTTPError as e:
                console.print(
                    f"Failed to fetch repo files: {e}", style="red"
                )
                raise SystemExit(1)

            if not files:
                console.print(
                    "No GGUF files found in this repo.", style="red"
                )
                raise SystemExit(1)

            chosen = _pick_quant_interactive(repo_id, files)
            filename = chosen["path"]
            size = _file_size(chosen)
            quant = _extract_quant(filename)
            model_id = f"{repo_id}:{quant}"
    else:
        repo_id, filename, quant, size = _resolve_download_target(model_id)
        model_id = f"{repo_id}:{quant}"

    # Prepare cache paths
    org, repo_name = repo_id.split("/", 1)
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = f"{org}_{repo_name}_{filename.replace('/', '_')}"
    dest = cache_dir / cache_filename

    # Create manifest early so partial downloads are discoverable
    _create_manifest(cache_dir, org, repo_name, quant, filename, size)

    # Download (or resume) if file is missing or incomplete
    if dest.exists() and dest.stat().st_size >= size > 0:
        console.print(f"File already cached: {dest.name}", style="yellow")
    else:
        action = "Resuming" if dest.exists() else "Downloading"
        console.print(
            f"{action} [bold]{model_id}[/bold] ({_human_size(size)})…"
        )
        etag = _download_gguf(repo_id, filename, dest, size)
        if etag:
            (cache_dir / f"{cache_filename}.etag").write_text(etag)

    # Add to preset (if not already there)
    preset = read_preset()
    if model_id not in preset:
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
    else:
        console.print(
            f"[bold]{model_id}[/bold] is ready.", style="green"
        )


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
