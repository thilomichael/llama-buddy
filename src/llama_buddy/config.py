"""Configuration paths and preset file management."""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "llama"
PRESET_FILE = CONFIG_DIR / "models.ini"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"
VRAM_FILE = CONFIG_DIR / "vram.json"

DEFAULT_PORT = 8080


def _default_cache_dir() -> Path:
    """Platform-appropriate default cache directory for llama.cpp."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    # Linux / other: follow XDG
    xdg = Path.home() / ".cache"
    return xdg / "llama.cpp"


@lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    """Detect the llama.cpp cache directory.

    Queries `llama-server --cache-list` which prints the cache path.
    Falls back to platform-specific defaults.
    """
    binary = shutil.which("llama-server")
    if binary is not None:
        try:
            result = subprocess.run(
                [binary, "--cache-list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if line.startswith("model cache directory:"):
                    path = Path(line.split(":", 1)[1].strip())
                    if path.exists():
                        return path
        except (subprocess.TimeoutExpired, OSError):
            pass
    return _default_cache_dir()


def get_hf_hub_dir() -> Path:
    """Return the HuggingFace hub cache directory.

    llama-server 8500+ uses this as the primary model cache.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_dir(org: str, repo: str) -> Path:
    """Return the HF hub directory for a specific model."""
    return get_hf_hub_dir() / f"models--{org}--{repo}"


def _hf_model_dir_for_id(model_id: str) -> Path:
    """Return the HF hub dir for a model ID (org/repo or org/repo:quant)."""
    base_repo = model_id.split(":")[0]
    org, repo = base_repo.split("/", 1)
    return hf_model_dir(org, repo)


_QUANT_RE = re.compile(r"[-_]((?:IQ|Q|F|BF)\d\w*)$")


def _quant_from_stem(stem: str) -> str:
    """Extract quantisation tag from a GGUF filename stem."""
    # Strip shard suffix first
    clean = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    m = _QUANT_RE.search(clean)
    return m.group(1) if m else "latest"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def read_preset() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if PRESET_FILE.exists():
        text = PRESET_FILE.read_text()
        # llama-server preset files may have top-level keys (e.g. "version = 1")
        # before any section header. Wrap them in a [DEFAULT] section.
        if text and not text.lstrip().startswith("["):
            text = "[DEFAULT]\n" + text
        config.read_string(text, source=str(PRESET_FILE))
    return config


def write_preset(config: configparser.ConfigParser) -> None:
    ensure_config_dir()
    with open(PRESET_FILE, "w") as f:
        config.write(f)


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    text = PID_FILE.read_text().strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def write_pid(pid: int) -> None:
    ensure_config_dir()
    PID_FILE.write_text(str(pid))


def remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def resolve_model(name: str) -> str | None:
    """Resolve a model name or alias to a preset section (model ID).

    Matches against: exact section name, alias, or case-insensitive alias.
    Returns the section name (model ID) or None.
    """
    preset = read_preset()
    # Exact section match
    if name in preset:
        return name
    # Alias match
    for section in preset.sections():
        alias = preset.get(section, "alias", fallback=None)
        if alias is not None and alias == name:
            return section
    # Case-insensitive alias match
    name_lower = name.lower()
    for section in preset.sections():
        alias = preset.get(section, "alias", fallback=None)
        if alias is not None and alias.lower() == name_lower:
            return section
    return None


def parse_manifest_stem(stem: str) -> tuple[str, str, str, str] | None:
    """Parse a manifest filename stem into (model_id, org, repo, quant).

    Stem format: manifest=org=repo=quant
    Returns None if the stem doesn't match the expected format.
    """
    parts = stem.split("=")
    if len(parts) != 4:
        return None
    _, org, repo, quant = parts
    if quant.lower() == "latest":
        model_id = f"{org}/{repo}"
    else:
        model_id = f"{org}/{repo}:{quant}"
    return model_id, org, repo, quant


def manifest_filename(org: str, repo: str, quant: str) -> str:
    """Build a manifest filename from components."""
    return f"manifest={org}={repo}={quant}.json"


def _read_manifest_map() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse cache manifests and HF hub into model-to-GGUF mappings.

    Single source of truth for manifest parsing. Returns (model_to_gguf,
    gguf_to_models).
    """
    model_to_gguf: dict[str, str] = {}

    # Legacy flat-cache manifests
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        for manifest in cache_dir.glob("manifest=*.json"):
            parsed = parse_manifest_stem(manifest.stem)
            if parsed is None:
                continue
            model_id, _, _, _ = parsed
            try:
                data = json.loads(manifest.read_text())
                gguf = data.get("ggufFile", {}).get("rfilename", "")
                if gguf:
                    model_to_gguf[model_id] = gguf
            except (json.JSONDecodeError, OSError):
                continue

    # HF hub cache (llama-server 8500+) — only scan GGUF repos
    hf_dir = get_hf_hub_dir()
    if hf_dir.exists():
        for model_dir in hf_dir.glob("models--*--*-GGUF"):
            if not model_dir.is_dir():
                continue
            parts = model_dir.name.split("--", 2)
            if len(parts) < 3:
                continue
            org, repo = parts[1], parts[2]
            snapshots = model_dir / "snapshots"
            if not snapshots.exists():
                continue
            for gguf in snapshots.glob("**/*.gguf"):
                if "mmproj" in gguf.name:
                    continue
                quant = _quant_from_stem(gguf.stem)
                mid = f"{org}/{repo}:{quant}" if quant != "latest" else f"{org}/{repo}"
                if mid not in model_to_gguf:
                    model_to_gguf[mid] = gguf.name

    gguf_to_models: dict[str, list[str]] = {}
    for mid, gguf in model_to_gguf.items():
        gguf_to_models.setdefault(gguf, []).append(mid)

    return model_to_gguf, gguf_to_models


def _get_manifest_model_ids() -> set[str]:
    """Return the set of model IDs that have cache manifests."""
    model_to_gguf, _ = _read_manifest_map()
    return set(model_to_gguf.keys())


def _link_flat_cache_to_hf_hub() -> None:
    """Ensure flat-cache GGUFs are symlinked into HF hub so llama-server finds them.

    For each HF hub model dir that has refs/main but no snapshot GGUFs,
    look for matching files in the flat cache and create symlinks.
    """
    hf_dir = get_hf_hub_dir()
    cache_dir = get_cache_dir()
    if not hf_dir.exists() or not cache_dir.exists():
        return

    for model_dir in hf_dir.glob("models--*--*"):
        if not model_dir.is_dir():
            continue
        parts = model_dir.name.split("--", 2)
        if len(parts) < 3:
            continue
        org, repo = parts[1], parts[2]

        # Read commit SHA from refs/main
        refs_main = model_dir / "refs" / "main"
        if not refs_main.exists():
            continue
        sha = refs_main.read_text().strip()
        if not sha:
            continue

        snapshot_dir = model_dir / "snapshots" / sha

        # Skip if snapshot already has GGUFs
        if snapshot_dir.exists() and list(snapshot_dir.glob("**/*.gguf")):
            continue

        # Find matching flat-cache files
        prefix = f"{org}_{repo}_"
        flat_files = [
            f for f in cache_dir.glob(f"{prefix}*.gguf")
            if "mmproj" not in f.name
        ]
        if not flat_files:
            continue

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for flat_file in flat_files:
            # Recover original filename: strip org_repo_ prefix
            original_name = flat_file.name[len(prefix):]
            link = snapshot_dir / original_name
            if not link.exists():
                link.symlink_to(flat_file.resolve())


def sync_preset_with_cache() -> list[str]:
    """Sync the preset file with the cache.

    - Links flat-cache GGUFs into HF hub so llama-server can find them.
    - Adds models found in cache manifests that are missing from the preset.
    - Removes orphaned preset entries (no manifest and no GGUF files).

    Returns a list of model IDs that were added.
    """
    _link_flat_cache_to_hf_hub()
    manifest_ids = _get_manifest_model_ids()

    preset = read_preset()
    sections = {s.lower(): s for s in preset.sections()}
    changed = False
    added: list[str] = []

    # Add missing manifest models to preset
    for model_id in sorted(manifest_ids):
        if model_id.lower() not in sections:
            preset.add_section(model_id)
            sections[model_id.lower()] = model_id
            added.append(model_id)
            changed = True

    # Remove orphaned preset entries (no manifest, no GGUF files)
    for section in list(preset.sections()):
        if section == "*":
            continue
        if section in manifest_ids:
            continue
        if find_model_gguf_files(section):
            continue
        preset.remove_section(section)
        changed = True

    if changed:
        write_preset(preset)

    return added


def get_model_groups() -> dict[str, list[str]]:
    """Map each model_id to its siblings (other IDs sharing the same GGUF)."""
    model_to_gguf, gguf_to_models = _read_manifest_map()
    groups: dict[str, list[str]] = {}
    for mid, gguf in model_to_gguf.items():
        groups[mid] = [m for m in gguf_to_models[gguf] if m != mid]
    return groups


def get_gguf_model_groups() -> list[list[str]]:
    """Return groups of model IDs that share the same GGUF file.

    Each group is a list of model_ids. Groups with one entry are singletons.
    """
    _, gguf_to_models = _read_manifest_map()
    return list(gguf_to_models.values())


def find_model_gguf_files(model_id: str) -> list[Path]:
    """Find all .gguf files in cache for a model ID (excluding mmproj).

    Searches HF hub cache first, then falls back to flat cache.
    """
    files: list[Path] = []
    seen: set[Path] = set()

    # HF hub format: models--org--repo/snapshots/hash/*.gguf
    hf_dir = _hf_model_dir_for_id(model_id)
    snapshots = hf_dir / "snapshots"
    if snapshots.exists():
        for f in snapshots.glob("**/*.gguf"):
            resolved = f.resolve()
            if "mmproj" not in f.name and resolved not in seen:
                files.append(f)
                seen.add(resolved)

    # Flat cache fallback: org_repo_filename.gguf
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        base_repo = model_id.split(":")[0]
        prefix = base_repo.replace("/", "_") + "_"
        for f in cache_dir.glob(f"{prefix}*.gguf"):
            resolved = f.resolve()
            if "mmproj" not in f.name and resolved not in seen:
                files.append(f)
                seen.add(resolved)

    return sorted(files)


def find_model_manifests(model_id: str) -> list[Path]:
    """Find manifest files in cache for a model ID.

    Checks both flat cache (manifest=org=repo=quant.json) and HF hub.
    """
    results: list[Path] = []
    base_repo = model_id.split(":")[0]
    org, repo = base_repo.split("/", 1)

    # Flat cache manifests
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        results.extend(cache_dir.glob(f"manifest={org}={repo}=*.json"))

    # HF hub manifest (llb-created)
    hf_dir = hf_model_dir(org, repo)
    manifest = hf_dir / "llb_manifest.json"
    if manifest.exists():
        results.append(manifest)

    return sorted(results)


# ---------------------------------------------------------------------------
# VRAM usage parsing from server.log
# ---------------------------------------------------------------------------

_RE_SPAWN = re.compile(
    r"srv\s+load: spawning server instance with name=(\S+) on port (\d+)"
)
_RE_BUFFER = re.compile(
    r"\[(\d+)\].*buffer size\s*=\s*([\d.]+)\s*MiB"
)
_RE_SLOTS = re.compile(
    r"\[(\d+)\].*load_model: initializing slots"
)


def parse_vram_from_log(log_path: Path | None = None) -> dict[str, float]:
    """Parse server.log and return {model_id: total_MiB} for each loaded model.

    Uses a state machine:
    1. "spawning server instance with name=MODEL on port PORT" maps PORT→MODEL
    2. "[PORT] ... buffer size = X MiB" accumulates MiB for that PORT
    3. "[PORT] ... initializing slots" flushes the accumulated value

    Only the last load per model is kept (handles sleep/wake reloads).
    """
    if log_path is None:
        log_path = LOG_FILE
    if not log_path.exists():
        return {}

    port_to_model: dict[str, str] = {}
    port_accum: dict[str, float] = {}
    result: dict[str, float] = {}

    for line in log_path.read_text(errors="replace").splitlines():
        m = _RE_SPAWN.search(line)
        if m:
            model_id, port = m.group(1), m.group(2)
            port_to_model[port] = model_id
            port_accum[port] = 0.0
            continue

        m = _RE_BUFFER.search(line)
        if m:
            port, mib = m.group(1), float(m.group(2))
            if port in port_to_model:
                port_accum[port] = port_accum.get(port, 0.0) + mib
            continue

        m = _RE_SLOTS.search(line)
        if m:
            port = m.group(1)
            if port in port_to_model and port_accum.get(port, 0.0) > 0:
                result[port_to_model[port]] = port_accum[port]
                port_accum[port] = 0.0

    return result


def parse_child_ports(log_path: Path | None = None) -> dict[str, int]:
    """Parse server.log and return {model_id: port} for each child server.

    Only the last spawn per model is kept (handles restarts).
    """
    if log_path is None:
        log_path = LOG_FILE
    if not log_path.exists():
        return {}

    result: dict[str, int] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = _RE_SPAWN.search(line)
        if m:
            model_id, port = m.group(1), int(m.group(2))
            result[model_id] = port
    return result


def read_vram_usage() -> dict[str, float]:
    """Read cached VRAM usage from vram.json."""
    if not VRAM_FILE.exists():
        return {}
    try:
        return json.loads(VRAM_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def update_vram_usage(log_path: Path | None = None) -> dict[str, float]:
    """Parse server.log, merge into vram.json, and return the result."""
    existing = read_vram_usage()
    parsed = parse_vram_from_log(log_path)
    existing.update(parsed)
    if parsed:
        ensure_config_dir()
        VRAM_FILE.write_text(json.dumps(existing, indent=2) + "\n")
    return existing
