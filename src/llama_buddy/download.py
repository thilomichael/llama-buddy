"""Model download and removal."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import webbrowser
from pathlib import Path

import httpx
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
    get_hf_hub_dir,
    hf_model_dir,
    read_preset,
    resolve_model,
    write_preset,
)
from llama_buddy.console import console
from llama_buddy.models import format_size
from llama_buddy.tui import read_key, read_key_timeout, require_tty

HF_API = "https://huggingface.co/api/models"


def _compact_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _human_size(size: int) -> str:
    return format_size(size, compact=False)


def _file_size(entry: dict) -> int:
    return entry.get("size", 0) or entry.get("lfs", {}).get("size", 0)


def _file_oid(entry: dict) -> str | None:
    """Get the LFS OID (blob hash) from an HF API file entry."""
    return entry.get("lfs", {}).get("oid") or entry.get("oid")


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


def _get_repo_commit(client: httpx.Client, repo_id: str) -> str:
    """Fetch the latest commit SHA for a repo's main branch."""
    resp = client.get(f"{HF_API}/{repo_id}/revision/main")
    resp.raise_for_status()
    return resp.json()["sha"]


def _get_repo_files(repo_id: str) -> list[dict]:
    """Get GGUF files in a HF repo with sizes (excludes mmproj).

    Uses the model info endpoint with ``?blobs=true`` to get all files
    (including subdirectories) with sizes and LFS OIDs in a single call.

    Split-model shards (e.g. ``model-00001-of-00003.gguf``) are grouped
    into a single entry whose ``"path"`` is the first shard, ``"size"``
    is the total across all shards, and ``"shard_files"`` lists every
    shard dict.
    """
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        resp = client.get(f"{HF_API}/{repo_id}?blobs=true")
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])

        gguf_files: list[dict] = []
        for entry in siblings:
            fname = entry.get("rfilename", "")
            if not fname.endswith(".gguf"):
                continue
            if "mmproj" in fname.lower():
                continue
            # Normalise to the same shape as /tree/main entries
            lfs = entry.get("lfs", {})
            gguf_files.append({
                "path": fname,
                "size": entry.get("size", 0),
                "lfs": {
                    "oid": lfs.get("sha256", ""),
                    "size": lfs.get("size", 0),
                },
            })

    return _merge_shards(gguf_files)


_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def _merge_shards(files: list[dict]) -> list[dict]:
    """Group split-model shards into single entries.

    Shard files matching ``*-00001-of-00005.gguf`` are merged into one dict
    keyed by the first shard.  Non-shard files pass through unchanged.
    """
    shard_groups: dict[str, list[dict]] = {}
    singles: list[dict] = []

    for f in files:
        path = f.get("path", "")
        m = _SHARD_RE.search(path)
        if m:
            # Key = path with shard suffix stripped
            key = path[: m.start()]
            shard_groups.setdefault(key, []).append(f)
        else:
            singles.append(f)

    merged: list[dict] = list(singles)
    for _key, shards in sorted(shard_groups.items()):
        shards.sort(key=lambda s: s["path"])
        total_size = sum(_file_size(s) for s in shards)
        merged.append(
            {
                "path": re.sub(r"-\d{5}-of-\d{5}", "", shards[0]["path"]),
                "size": total_size,
                "shard_files": shards,
            }
        )
    return merged


# ---------------------------------------------------------------------------
# Partial download detection
# ---------------------------------------------------------------------------


def _find_file_in_caches(org: str, repo: str, filename: str) -> Path | None:
    """Find a GGUF file in HF hub cache, return the path or None."""
    model_dir = hf_model_dir(org, repo)
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for f in snapshots.glob(f"**/{Path(filename).name}"):
            if f.exists():  # symlink target must exist
                return f
    return None


def _find_partial_downloads() -> list[dict]:
    """Scan HF hub blobs dirs for .downloadInProgress files.

    Returns a list of dicts with keys: model_id, repo_id, filename, quant,
    display_name, size, downloaded.
    """
    from llama_buddy.config import _quant_from_stem

    hf_dir = get_hf_hub_dir()
    if not hf_dir.exists():
        return []

    partials: list[dict] = []
    for model_dir in hf_dir.glob("models--*--*"):
        if not model_dir.is_dir():
            continue
        blobs_dir = model_dir / "blobs"
        if not blobs_dir.exists():
            continue

        parts = model_dir.name.split("--", 2)
        if len(parts) < 3:
            continue
        org, repo = parts[1], parts[2]

        for wip in blobs_dir.glob("*.downloadInProgress"):
            # OID is the filename minus the suffix — we can't recover
            # the original GGUF filename from it, so use the repo name
            quant = "unknown"
            display_name = f"{org}/{repo}"
            model_id = f"{org}/{repo}"

            # Try to find the quant from snapshot symlinks in this repo
            # (check both broken and valid symlinks — for multi-shard
            # models some shards may already be complete)
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                for gguf in snapshots.glob("**/*.gguf"):
                    if gguf.is_symlink():
                        quant = _quant_from_stem(gguf.stem)
                        display_name = f"{org}/{repo} ({_extract_quant(gguf.name)})"
                        model_id = (
                            f"{org}/{repo}:{quant}"
                            if quant != "unknown"
                            else f"{org}/{repo}"
                        )
                        break

            partials.append(
                {
                    "model_id": model_id,
                    "repo_id": f"{org}/{repo}",
                    "filename": "",
                    "quant": quant,
                    "display_name": display_name,
                    "size": 0,
                    "downloaded": wip.stat().st_size,
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
                text.append(p["display_name"], style="bold cyan")
                text.append(f"  ({progress_str})", style="cyan")
            else:
                text.append(f"    {p['display_name']}", style="yellow")
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


class _RepoPickerResult:
    """Result from _pick_repo_interactive with state for ← back."""

    __slots__ = ("value", "query", "entries")

    def __init__(
        self, value: str | dict, query: str, entries: list[dict],
    ) -> None:
        self.value = value
        self.query = query
        self.entries = entries


def _pick_repo_interactive(
    *,
    prev_query: str = "",
    prev_entries: list[dict] | None = None,
) -> _RepoPickerResult:
    """Interactive search and selection of a HF GGUF repo.

    Returns a ``_RepoPickerResult`` whose ``.value`` is a repo ID string
    or a partial-download dict.  The result also carries the search state
    so it can be restored when returning from the quant picker via ← back.
    """
    require_tty()
    query = prev_query
    entries: list[dict] = prev_entries if prev_entries is not None else []
    partials = _find_partial_downloads()
    selected = 0

    # Background search state
    searched_query = prev_query if prev_entries is not None else ""
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
                    return _RepoPickerResult(
                        partials[selected], query, entries,
                    )
                elif entries and active_len > 0:
                    return _RepoPickerResult(
                        entries[selected]["id"], query, entries,
                    )
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
            shards = f.get("shard_files")
            shard_info = f"  {len(shards)} parts" if shards else ""
            is_sel = idx == selected

            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(path, style="bold cyan")
                text.append(f"  ({size}{shard_info})", style="cyan")
            else:
                text.append(f"    {path}", style="dim")
                text.append(f"  ({size}{shard_info})", style="dim italic")
            text.append("\n")
            idx += 1
        text.append("\n")

    text.append(
        "  ↑/↓ navigate  Enter select  ← back  Ctrl-O open  Ctrl-C cancel",
        style="dim italic",
    )
    return text


def _pick_quant_interactive(repo_id: str, files: list[dict]) -> dict | None:
    """Interactive selection of a GGUF file from a repo.

    Returns the chosen file dict, or None if the user pressed ← to go back.
    """
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
            elif key == "left":
                return None
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
    it left off using an HTTP Range request.  Returns the ETag (stripped of
    quotes), which is the SHA-256 content hash used as the blob filename in
    the HF hub cache.
    """
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    existing = dest.stat().st_size if dest.exists() else 0
    if existing >= size > 0:
        return None  # Already complete
    headers: dict[str, str] = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with httpx.Client(
        timeout=httpx.Timeout(connect=15, read=30, write=30, pool=15),
        follow_redirects=True,
    ) as client:
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            etag = resp.headers.get("etag", "").strip('"')

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

    return etag or None



def _resolve_download_target(
    model_id: str,
) -> tuple[str, dict, str]:
    """Resolve a model_id to (repo_id, chosen_file_dict, quant).

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
            if chosen is None:
                raise SystemExit(0)
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

    return repo_id, chosen, quant


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _setup_hf_hub_entry(
    org: str,
    repo_name: str,
    commit_sha: str,
) -> tuple[Path, Path, Path]:
    """Create HF hub directory structure.

    Replicates the native HuggingFace hub layout:
        models--org--repo/
            refs/main           (commit SHA)
            blobs/{content-hash} (actual file data)
            snapshots/{sha}/
                file.gguf       (symlink → ../../blobs/{hash})

    Returns (model_dir, snapshot_dir, blobs_dir).
    """
    model_dir = hf_model_dir(org, repo_name)
    snapshot_dir = model_dir / "snapshots" / commit_sha
    blobs_dir = model_dir / "blobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(commit_sha)
    return model_dir, snapshot_dir, blobs_dir


def _download_files(
    repo_id: str,
    chosen: dict,
    org: str,
    repo_name: str,
    model_id: str,
) -> str:
    """Download one or more GGUF files for a model.

    Replicates the native llama.cpp / HF hub download behaviour:
      1. Download to ``blobs/{oid}.downloadInProgress``
      2. Rename to ``blobs/{oid}`` on completion
      3. Symlink ``snapshots/{sha}/filename.gguf`` → ``../../blobs/{oid}``

    Returns the first shard's filename.
    """
    shard_files = chosen.get("shard_files")
    if shard_files:
        file_list = shard_files
    else:
        file_list = [chosen]

    total_size = sum(_file_size(f) for f in file_list)
    first_filename = file_list[0]["path"]

    # Fetch commit SHA and set up HF hub directory
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            commit_sha = _get_repo_commit(client, repo_id)
    except (httpx.HTTPError, KeyError):
        commit_sha = hashlib.sha1(repo_id.encode()).hexdigest()
    _, snapshot_dir, blobs_dir = _setup_hf_hub_entry(
        org, repo_name, commit_sha,
    )

    all_cached = True
    for f in file_list:
        filename = f["path"]
        size = _file_size(f)
        oid = _file_oid(f)

        link = snapshot_dir / filename
        link.parent.mkdir(parents=True, exist_ok=True)

        # Derive blob name: prefer LFS OID, fall back to filename hash
        if not oid:
            oid = hashlib.sha256(filename.encode()).hexdigest()
        blob_path = blobs_dir / oid

        # Relative path from link's parent to blobs dir (handles subdirs)
        rel_to_blobs = Path(os.path.relpath(blobs_dir, link.parent))

        # Already complete: blob exists with correct size and symlink is valid
        if blob_path.exists() and blob_path.stat().st_size >= size > 0:
            if not (link.is_symlink() and link.exists()):
                # Blob is fine but symlink is missing — create it
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(rel_to_blobs / blob_path.name)
            console.print(
                f"File already cached: {filename}", style="yellow"
            )
            continue

        # Need to download
        all_cached = False
        wip = blobs_dir / f"{oid}.downloadInProgress"

        # Create symlink early so partial downloads are discoverable
        # (broken symlink until blob is complete, but is_symlink() still
        # returns True — lets _find_partial_downloads extract the quant)
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(rel_to_blobs / blob_path.name)

        if shard_files:
            shard_idx = file_list.index(f) + 1
            console.print(
                f"Downloading part {shard_idx}/{len(file_list)}"
                f" ({_human_size(size)})…"
            )
        else:
            action = "Resuming" if wip.exists() else "Downloading"
            console.print(
                f"{action} [bold]{model_id}[/bold]"
                f" ({_human_size(total_size)})…"
            )

        _download_gguf(repo_id, filename, wip, size)
        wip.rename(blob_path)

    if all_cached and shard_files:
        console.print("All parts already cached.", style="yellow")

    return first_filename


# GGUF general.sampling.* → llama-server INI key mapping
_SAMPLING_KEY_MAP = {
    "temperature": "temp",
    "top_k": "top-k",
    "top_p": "top-p",
    "min_p": "min-p",
    "typical_p": "typical",
    "repeat_penalty": "repeat-penalty",
    "repeat_last_n": "repeat-last-n",
    "presence_penalty": "presence-penalty",
    "frequency_penalty": "frequency-penalty",
    "mirostat": "mirostat",
    "mirostat_tau": "mirostat-tau",
    "mirostat_eta": "mirostat-eta",
    "dynatemp_range": "dynatemp-range",
    "dynatemp_exponent": "dynatemp-exponent",
    "xtc_probability": "xtc-probability",
    "xtc_threshold": "xtc-threshold",
    "top_n_sigma": "top-n-sigma",
}


def _sync_sampling_params(model_id: str, preset) -> None:
    """Read GGUF sampling metadata and write to preset if not already set."""
    from llama_buddy.gguf import read_metadata

    gguf_files = find_model_gguf_files(model_id)
    if not gguf_files:
        return

    try:
        meta = read_metadata(gguf_files[0])
    except (ValueError, OSError):
        return

    sampling = {
        k.removeprefix("general.sampling."): v
        for k, v in meta.items()
        if k.startswith("general.sampling.")
    }
    if not sampling:
        return

    wrote_any = False
    for gguf_key, value in sampling.items():
        ini_key = _SAMPLING_KEY_MAP.get(gguf_key)
        if ini_key is None:
            continue
        # Don't overwrite existing user customizations
        if preset.has_option(model_id, ini_key):
            continue
        preset.set(model_id, ini_key, str(value))
        wrote_any = True

    if wrote_any:
        console.print(
            "Applied recommended sampling params from GGUF metadata.",
            style="dim",
        )


def download(model_id: str | None = None, alias: str | None = None) -> None:
    """Download a model — interactively or by explicit ID."""
    if model_id is None:
        prev_query = ""
        prev_entries: list[dict] | None = None
        while True:
            picker = _pick_repo_interactive(
                prev_query=prev_query, prev_entries=prev_entries,
            )
            result = picker.value
            if isinstance(result, dict):
                # Resuming a partial download — re-fetch file list
                # from HF to get proper paths, sizes, and shard info
                repo_id = result["repo_id"]
                quant = result["quant"]
                model_id = result["model_id"]
                console.print(
                    f"Fetching files for [bold]{repo_id}[/bold]…"
                )
                files = _get_repo_files(repo_id)
                q_lower = quant.lower()
                matches = [
                    f for f in files if q_lower in f["path"].lower()
                ]
                if matches:
                    chosen = matches[0]
                    quant = _extract_quant(chosen["path"])
                    model_id = f"{repo_id}:{quant}"
                else:
                    console.print(
                        f"Could not find '{quant}' in {repo_id}.",
                        style="red",
                    )
                    raise SystemExit(1)
                break

            repo_id = result
            console.print(
                f"\nFetching files for [bold]{repo_id}[/bold]…"
            )
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
            if chosen is None:
                prev_query = picker.query
                prev_entries = picker.entries
                continue
            quant = _extract_quant(chosen["path"])
            model_id = f"{repo_id}:{quant}"
            break
    else:
        repo_id, chosen, quant = _resolve_download_target(model_id)
        model_id = f"{repo_id}:{quant}"

    org, repo_name = repo_id.split("/", 1)

    _download_files(repo_id, chosen, org, repo_name, model_id)

    # Add to preset (if not already there)
    preset = read_preset()
    is_new = model_id not in preset
    if is_new:
        if not preset.has_section("*"):
            preset.add_section("*")
            preset.set("*", "c", "0")

        preset.add_section(model_id)
        if alias is not None:
            preset.set(model_id, "alias", alias)

    # Sync GGUF sampling params into preset (only for keys not already set)
    _sync_sampling_params(model_id, preset)

    write_preset(preset)

    if is_new:
        msg = f"Added [bold]{model_id}[/bold]"
        if alias:
            msg += f" (alias: {alias})"
        msg += " to preset file."
        console.print(msg, style="green")
    else:
        console.print(
            f"[bold]{model_id}[/bold] is ready.", style="green"
        )


# ---------------------------------------------------------------------------
# TUI: remove picker
# ---------------------------------------------------------------------------


def _render_remove_menu(
    complete: list[tuple[str, str, int]],
    partial: list[dict],
    selected: int,
) -> Text:
    text = Text()
    text.append("  Select a model to remove\n\n", style="bold")

    idx = 0
    if complete:
        text.append("  Models\n", style="bold dim")
        for section, name, size in complete:
            is_sel = idx == selected
            size_str = _human_size(size) if size else ""
            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(name or section, style="bold cyan")
                if size_str:
                    text.append(f"  ({size_str})", style="cyan")
                if name:
                    text.append(f"  {section}", style="cyan dim")
            else:
                text.append(f"    {name or section}", style="dim")
                if size_str:
                    text.append(f"  ({size_str})", style="dim italic")
                if name:
                    text.append(f"  {section}", style="dim italic")
            text.append("\n")
            idx += 1
        text.append("\n")

    if partial:
        text.append("  Incomplete downloads\n", style="bold yellow")
        for p in partial:
            is_sel = idx == selected
            pct = p["downloaded"] / p["size"] * 100 if p["size"] else 0
            size_str = _human_size(p["size"])
            progress_str = f"{pct:.0f}% of {size_str}"
            if is_sel:
                text.append("  > ", style="bold cyan")
                text.append(p["display_name"], style="bold cyan")
                text.append(f"  ({progress_str})", style="cyan")
            else:
                text.append(f"    {p['display_name']}", style="yellow")
                text.append(f"  ({progress_str})", style="yellow dim")
            text.append("\n")
            idx += 1
        text.append("\n")

    if not complete and not partial:
        text.append("    No models to remove\n", style="dim italic")

    text.append("\n")
    text.append(
        "  ↑/↓ navigate  Enter remove  Ctrl-C cancel", style="dim italic"
    )
    return text


def _pick_remove_interactive() -> str:
    """Interactive picker for removing a model. Returns model ID."""
    from llama_buddy.models import compute_model_sizes, get_model_meta

    require_tty()

    with console.status("Loading models…", spinner="dots"):
        preset = read_preset()
        sections = [s for s in preset.sections() if s != "*"]
        sizes = compute_model_sizes()
        partial = _find_partial_downloads()
        partial_ids = {p["model_id"] for p in partial}

        # Complete models (in preset, not partial)
        complete: list[tuple[str, str, int]] = []
        for section in sections:
            if section in partial_ids:
                continue
            meta = get_model_meta(section)
            name = meta.get("name", "")
            base_repo = section.split(":")[0]
            size = sizes.get(base_repo, 0)
            complete.append((section, name, size))
        complete.sort(key=lambda e: (e[1] or e[0]).lower())

    if not complete and not partial:
        console.print(
            "No models to remove.", style="yellow"
        )
        raise SystemExit(0)

    selected = 0
    total = len(complete) + len(partial)

    with Live(
        _render_remove_menu(complete, partial, selected),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = read_key()
            if key == "up" and selected > 0:
                selected -= 1
            elif key == "down" and selected < total - 1:
                selected += 1
            elif key in ("enter", "right"):
                if selected < len(complete):
                    return complete[selected][0]
                else:
                    return partial[selected - len(complete)]["model_id"]
            elif key == "ctrl-c":
                raise SystemExit(0)
            else:
                continue

            live.update(
                _render_remove_menu(complete, partial, selected),
                refresh=True,
            )


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------


def _remove_model(section: str, keep_files: bool) -> None:
    """Remove a model: preset entry and optionally GGUF files + blobs.

    GGUF files are only deleted if no other models share them.
    """
    import shutil

    from llama_buddy.config import _hf_model_dir_for_id, get_model_groups

    preset = read_preset()
    if preset.has_section(section):
        preset.remove_section(section)
        write_preset(preset)

    # Check if other models share the same GGUF files
    siblings = get_model_groups().get(section, [])

    # Only delete GGUF files if no siblings remain
    if not keep_files and not siblings:
        for f in find_model_gguf_files(section):
            # Delete blob (symlink target) and the snapshot symlink
            resolved = f.resolve()
            f.unlink()
            if resolved != f and resolved.exists():
                resolved.unlink()
            console.print(f"Deleted {f.name}", style="dim")

        # Clean up HF hub model dir if no GGUFs remain
        hf_dir = _hf_model_dir_for_id(section)
        if hf_dir.exists():
            remaining = list(hf_dir.glob("snapshots/**/*.gguf"))
            if not remaining:
                shutil.rmtree(hf_dir)
                console.print(f"Removed {hf_dir.name}", style="dim")
    elif not keep_files and siblings:
        console.print(
            f"Keeping GGUF files (shared with {', '.join(siblings)})",
            style="dim",
        )

    console.print(f"Removed [bold]{section}[/bold].", style="green")


def _render_confirm(
    section: str,
    info_lines: list[tuple[str, str]],
    selected: int,
) -> Text:
    text = Text()
    text.append("  Remove ", style="bold")
    text.append(section, style="bold red")
    text.append("?\n\n")

    for line, style in info_lines:
        text.append(f"  {line}\n", style=style)

    text.append("\n")
    options = ["Cancel", "Remove"]
    for i, label in enumerate(options):
        if i == selected:
            text.append("  > ", style="bold cyan")
            text.append(label, style="bold cyan")
        else:
            text.append(f"    {label}", style="dim")
        text.append("    ")

    text.append("\n\n")
    text.append("  ←/→ navigate  Enter confirm", style="dim italic")
    return text


def _confirm_remove(section: str, keep_files: bool) -> bool:
    """Show a confirmation prompt before removing a model.

    Defaults to Cancel. User must navigate to Remove before pressing Enter.
    Skips confirmation in non-interactive environments.
    """
    import sys

    if not sys.stdin.isatty():
        return True

    from llama_buddy.config import get_model_groups

    files = find_model_gguf_files(section)
    total_size = sum(f.stat().st_size for f in files)
    siblings = get_model_groups().get(section, [])

    info_lines: list[tuple[str, str]] = []
    if siblings:
        info_lines.append(
            (f"GGUF files shared with: {', '.join(siblings)}", "dim")
        )
        info_lines.append(("Files will be kept on disk", "dim"))
    elif files and not keep_files:
        info_lines.append(
            (
                f"This will delete {len(files)} file(s)"
                f" ({_human_size(total_size)})",
                "dim",
            )
        )
    elif keep_files:
        info_lines.append(("GGUF files will be kept on disk", "dim"))

    selected = 0  # 0 = Cancel, 1 = Remove

    with Live(
        _render_confirm(section, info_lines, selected),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = read_key()
            if key in ("left", "right"):
                selected = 1 - selected
            elif key == "enter":
                return selected == 1
            elif key == "ctrl-c":
                return False
            else:
                continue

            live.update(
                _render_confirm(section, info_lines, selected),
                refresh=True,
            )


def remove(
    model_id_or_alias: str | None = None, keep_files: bool = False
) -> None:
    if model_id_or_alias is None:
        section = _pick_remove_interactive()
    else:
        section = resolve_model(model_id_or_alias)
        if section is None:
            console.print(
                f"Model '{model_id_or_alias}' not found in preset file.",
                style="red",
            )
            raise SystemExit(1)

    if not _confirm_remove(section, keep_files):
        return

    _remove_model(section, keep_files)
