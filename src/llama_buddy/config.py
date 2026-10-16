"""Configuration paths and preset file management."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from llama_buddy.ini import IniFile

CONFIG_DIR = Path.home() / ".config" / "llama"
PRESET_FILE = CONFIG_DIR / "models.ini"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"
VRAM_FILE = CONFIG_DIR / "vram.json"
MTP_BACKFILL_FILE = CONFIG_DIR / "mtp_backfill_done"

DEFAULT_PORT = 8080


def get_hf_hub_dir() -> Path:
    """Return the HuggingFace hub cache directory.

    llama-server 8500+ uses this as the primary model cache. The lookup
    order mirrors hf_cache::get_cache_directory() in llama.cpp, so llb reads
    and writes the same directory the router lists models from.
    """
    for var, sub in (
        ("LLAMA_CACHE", ()),
        ("HF_HUB_CACHE", ()),
        ("HUGGINGFACE_HUB_CACHE", ()),
        ("HF_HOME", ("hub",)),
        ("XDG_CACHE_HOME", ("huggingface", "hub")),
    ):
        base = os.environ.get(var)
        if base:
            return Path(base).joinpath(*sub)
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_dir(org: str, repo: str) -> Path:
    """Return the HF hub directory for a specific model."""
    return get_hf_hub_dir() / f"models--{org}--{repo}"


def _hf_model_dir_for_id(model_id: str) -> Path | None:
    """Return the HF hub dir for a model ID (org/repo or org/repo:quant).

    The router also lists models that never came from HF, such as a local
    GGUF named after its file; those have no org and no hub dir.
    """
    base_repo = model_id.split(":")[0]
    if "/" not in base_repo:
        return None
    org, repo = base_repo.split("/", 1)
    return hf_model_dir(org, repo)


# Speculative decoding sidecars: a draft head shipped as its own GGUF next to
# the model it drafts for. llama.cpp identifies them by filename prefix, see
# draft_prefixes in common/preset.cpp.
_DRAFT_PREFIXES = ("mtp-", "dspark-", "dflash-", "eagle3-")


def is_draft_file(name: str) -> bool:
    """Whether a GGUF path names a speculative decoding sidecar."""
    return name.rsplit("/", 1)[-1].lower().startswith(_DRAFT_PREFIXES)


_QUANT_BITS_RE = re.compile(r"(?:iq|q|bf|f)(\d+)", re.IGNORECASE)


def _quant_bits(tag: str) -> int | None:
    """Bits per weight named by a quant tag: Q4_K_M → 4, BF16 → 16."""
    m = _QUANT_BITS_RE.search(tag)
    return int(m.group(1)) if m else None


def pick_draft_file(paths: list[str], quant: str) -> str | None:
    """Choose the sidecar that belongs with a quant.

    Takes the file whose quant tag matches the model's. Repos ship heads for
    a few quants only, so when none matches it falls back to the closest bit
    width, the way find_best_sibling in llama.cpp's common/download.cpp does.
    A ``-shared-`` head loses to a standalone one because it leaves out the
    weights it expects to take from the model.
    """
    if not paths:
        return None
    suffix = f"-{quant.lower()}.gguf"
    want = _quant_bits(quant)

    def rank(path: str) -> tuple:
        base = path.rsplit("/", 1)[-1].lower()
        bits = _quant_bits(base.removesuffix(".gguf").rsplit("-", 1)[-1])
        if want is None or bits is None:
            distance = 99
        else:
            distance = abs(bits - want)
        return (not base.endswith(suffix), distance, "-shared-" in base, base)

    return min(paths, key=rank)


_QUANT_RE = re.compile(r"[-_]((?:IQ|Q|F|BF)\d\w*)$")


def _quant_from_stem(stem: str) -> str:
    """Extract quantisation tag from a GGUF filename stem."""
    # Strip shard suffix first
    clean = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    m = _QUANT_RE.search(clean)
    return m.group(1) if m else "latest"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# llama.cpp's own configuration
# ---------------------------------------------------------------------------


def llama_config_file() -> Path:
    """llama.cpp's user-level config.ini, read by every llama.cpp tool."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "llama.cpp" / "config.ini"


def _llama_config_value(key: str) -> str | None:
    """Read a key from config.ini. Only its default and ``[*]`` sections apply."""
    config = IniFile.load(llama_config_file())
    return config.get("*", key, fallback=config.get(None, key))


def get_port() -> int:
    """The router's port, looked up the way llama-server looks it up."""
    for raw in (os.environ.get("LLAMA_ARG_PORT"), _llama_config_value("port")):
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return DEFAULT_PORT


def set_port(port: int) -> None:
    """Store the port in config.ini, or drop it there when it is the default."""
    path = llama_config_file()
    config = IniFile.load(path)
    config.remove_option("*", "port")
    if port == DEFAULT_PORT:
        if not config.remove_option(None, "port"):
            return
    else:
        config.set(None, "port", str(port))
    config.write(path)


def preset_path() -> Path:
    """The preset file the router is given with ``--models-preset``.

    A preset the user already points llama-server at, through the
    environment or config.ini, is used as is; llb's own file is only the
    fallback.
    """
    for raw in (
        os.environ.get("LLAMA_ARG_MODELS_PRESET"),
        _llama_config_value("models-preset"),
    ):
        if raw:
            return Path(raw).expanduser()
    return PRESET_FILE


def read_preset() -> IniFile:
    return IniFile.load(preset_path())


def write_preset(preset: IniFile) -> None:
    preset.write(preset_path())


# Every name llama.cpp accepts for a flag works as a preset key, so a
# hand-written file may spell a setting differently from llb.
_KEY_NAMES: dict[str, tuple[str, ...]] = {
    "c": ("c", "ctx-size", "LLAMA_ARG_CTX_SIZE"),
    "ngl": ("ngl", "gpu-layers", "n-gpu-layers", "LLAMA_ARG_N_GPU_LAYERS"),
    "fa": ("fa", "flash-attn", "LLAMA_ARG_FLASH_ATTN"),
    "jinja": ("jinja", "LLAMA_ARG_JINJA"),
    "alias": ("alias", "a", "LLAMA_ARG_ALIAS"),
    "load-mode": ("load-mode", "lm"),
    "lazy-mode": ("lazy-mode", "lzm"),
    "ncmoe": ("ncmoe", "n-cpu-moe"),
    "ncffn": ("ncffn", "n-cpu-ffn"),
    "spec-type": ("spec-type", "LLAMA_ARG_SPEC_TYPE"),
    "spec-draft-model": (
        "spec-draft-model", "md", "model-draft", "LLAMA_ARG_SPEC_DRAFT_MODEL",
    ),
}
_MODEL_KEYS = ("model", "m", "LLAMA_ARG_MODEL")
_HF_KEYS = ("hf-repo", "hf", "hfr", "LLAMA_ARG_HF_REPO")


def key_names(key: str) -> tuple[str, ...]:
    """All spellings of a preset key, llb's own first."""
    return _KEY_NAMES.get(key, (key,))


def model_option(
    preset: IniFile, model_id: str, key: str, fallback: str | None = None,
) -> str | None:
    """A model's setting, falling back to the preset's ``[*]`` section."""
    names = key_names(key)
    value = preset.get_any(model_id, names)
    if value is None:
        value = preset.get_any("*", names, fallback)
    return value


def section_aliases(preset: IniFile, section: str) -> list[str]:
    raw = preset.get_any(section, key_names("alias"), "") or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


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


def all_model_ids(preset: IniFile | None = None) -> list[str]:
    """Every model the router offers: the cached ones plus preset sections.

    The router lists cached models on its own, so a model only has a preset
    section when something about it differs from the defaults, or when the
    section defines a model under a name of its own.
    """
    if preset is None:
        preset = read_preset()
    ids = _get_manifest_model_ids()
    ids.update(s for s in preset.sections() if s != "*")
    return sorted(ids)


def resolve_model(name: str) -> str | None:
    """Resolve a model ID or alias to the ID the router knows it by.

    Matches the exact ID, then an alias, then either one ignoring case.
    """
    preset = read_preset()
    ids = all_model_ids(preset)
    if name in ids:
        return name
    aliases = {
        alias: section
        for section in preset.sections()
        for alias in section_aliases(preset, section)
    }
    if name in aliases:
        return aliases[name]
    name_lower = name.lower()
    for alias, section in aliases.items():
        if alias.lower() == name_lower:
            return section
    for model_id in ids:
        if model_id.lower() == name_lower:
            return model_id
    return None



def _read_manifest_map() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Scan HF hub cache for model-to-GGUF mappings.

    Single source of truth for model discovery. Returns (model_to_gguf,
    gguf_to_models).
    """
    model_to_gguf: dict[str, str] = {}

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
                if "mmproj" in gguf.name or is_draft_file(gguf.name):
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


def sync_preset_with_cache() -> list[str]:
    """Bring the preset in line with the cache.

    - Moves settings out of the old settings.json (once).
    - Drops sections that set nothing, and sections for cached models whose
      files are gone.
    - Turns MTP on for models it has not seen before, and re-checks every
      model when the llama-server binary changes.

    Returns the model IDs MTP was turned on for.
    """
    from llama_buddy.models import mtp_checked, sync_mtp_param
    from llama_buddy.settings import migrate_settings_file

    preset = read_preset()
    changed = migrate_settings_file(preset)
    discovered = _get_manifest_model_ids()

    for section in list(preset.sections()):
        if not preset.options(section):
            preset.remove_section(section)
            changed = True
        elif (
            section != "*"
            and section not in discovered
            and _is_orphan(preset, section)
        ):
            preset.remove_section(section)
            changed = True

    # MTP detection reads the GGUF tensor block, so it runs once per model
    # rather than on every command.
    enabled: list[str] = []
    for model_id in all_model_ids(preset):
        if not mtp_checked(model_id) and sync_mtp_param(model_id, preset):
            enabled.append(model_id)
            changed = True

    if _backfill_mtp(preset):
        changed = True

    if changed:
        write_preset(preset)

    return enabled


def _is_orphan(preset: IniFile, section: str) -> bool:
    """Whether a section is settings for a cached model that is gone.

    Only sections named after an HF model count. A section that brings its
    own model (``hf-repo``, ``model``) or is named after a file in
    ``--models-dir`` is the user's to remove.
    """
    if "/" not in section:
        return False
    if preset.get_any(section, _MODEL_KEYS + _HF_KEYS) is not None:
        return False
    return not find_model_gguf_files(section)


def _backfill_mtp(preset) -> bool:
    """Enable MTP for models whose head could not be used before.

    Covers models added before llb detected heads, and sidecar heads the
    llama.cpp of the time could not load: support for an architecture's
    MTP graph lands in llama.cpp after the model does, so the answer has
    to be asked again when the binary changes. Models added from here on
    are checked as they arrive, so this runs once per binary and records
    which one it checked. Returns True when the preset was changed.
    """
    from llama_buddy.capabilities import _binary_key
    from llama_buddy.models import sync_mtp_param, unwire_unloadable_sidecar

    binary = shutil.which("llama-server")
    marker = _binary_key(binary) if binary else "no-binary"
    try:
        if MTP_BACKFILL_FILE.read_text().strip() == marker:
            return False
    except OSError:
        pass

    from llama_buddy.console import console

    changed = False
    enabled: list[str] = []
    dropped: list[str] = []
    # A sidecar that has not been probed against this binary costs a model
    # load here (about a minute for 87 GB), so name the model being checked.
    with console.status("", spinner="dots") as status:
        for section in all_model_ids(preset):
            status.update(
                "Checking multi-token-prediction heads with this llama.cpp "
                f"build: [bold]{section}[/bold] [dim](a model with a separate "
                "head is test-loaded, which can take a minute)[/dim]"
            )
            if sync_mtp_param(section, preset):
                enabled.append(section)
                changed = True
            # The other direction: a build that can no longer load a head
            # that was wired in must not be handed it.
            elif unwire_unloadable_sidecar(section, preset):
                dropped.append(section)
                changed = True

    for section in enabled:
        console.print(f"Enabled MTP speculative decoding for {section}.", style="dim")
    for section in dropped:
        console.print(
            f"Disabled MTP for {section}: this llama-server cannot load its head.",
            style="yellow",
        )

    ensure_config_dir()
    MTP_BACKFILL_FILE.write_text(marker + "\n")
    return changed


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


# Where models the router reported, but the preset does not define, come
# from: a router started with someone else's preset file still says so in
# each model's launch args.
_router_sources: dict[str, tuple[str, str]] = {}

_SOURCE_FLAGS = {
    "-m": "path", "--model": "path",
    "-hf": "hf", "-hfr": "hf", "--hf-repo": "hf",
}


def note_router_models(models: list[dict]) -> None:
    """Remember the model file or HF repo behind each model the router lists."""
    for m in models:
        status = m.get("status")
        args = status.get("args") if isinstance(status, dict) else None
        if not isinstance(args, list):
            continue
        for flag, value in zip(args, args[1:]):
            kind = _SOURCE_FLAGS.get(flag)
            if kind is not None and isinstance(value, str):
                _router_sources[m.get("id", "")] = (kind, value)
                break


_preset_cache: tuple[tuple, IniFile] | None = None


def _current_preset() -> IniFile:
    """The preset as on disk, parsed once per change to the file."""
    global _preset_cache
    path = preset_path()
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (str(path),)
    if _preset_cache is None or _preset_cache[0] != key:
        _preset_cache = (key, IniFile.load(path))
    return _preset_cache[1]


def model_source(model_id: str) -> tuple[str, str]:
    """Where a model's weights come from: ``("path", file)`` or ``("hf", repo)``.

    A preset section can give a model any name and point it at a local file
    or an HF repo. A model with neither is a cached HF model named by its
    repo and quant.
    """
    preset = _current_preset()
    if preset.has_section(model_id):
        path = preset.get_any(model_id, _MODEL_KEYS)
        if path:
            return "path", path
        repo = preset.get_any(model_id, _HF_KEYS)
        if repo:
            return "hf", repo
    return _router_sources.get(model_id, ("hf", model_id))


_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-(\d{5})\.gguf$")


def _local_gguf_files(path: Path) -> list[Path]:
    """A local model file, with its sibling shards when it is split."""
    m = _SHARD_RE.match(path.name)
    if m:
        shards = sorted(path.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf"))
        if shards:
            return shards
    return [path] if path.exists() else []


def _snapshot_gguf_files(model_id: str, drafts: bool) -> list[Path]:
    """List the model's GGUFs, either its own files or its sidecars."""
    files: list[Path] = []
    seen: set[Path] = set()

    kind, source = model_source(model_id)
    if kind == "path":
        # llama.cpp finds a sidecar next to a local file by itself.
        return [] if drafts else _local_gguf_files(Path(source).expanduser())

    hf_dir = _hf_model_dir_for_id(source)
    if hf_dir is None:
        return files
    snapshots = hf_dir / "snapshots"
    if snapshots.exists():
        for f in snapshots.glob("**/*.gguf"):
            if "mmproj" in f.name or is_draft_file(f.name) is not drafts:
                continue
            resolved = f.resolve()
            if resolved not in seen:
                files.append(f)
                seen.add(resolved)

    return sorted(files)


def find_model_gguf_files(model_id: str) -> list[Path]:
    """Find all .gguf files in HF hub cache for a model ID.

    Excludes mmproj files and speculative decoding sidecars, which are
    separate GGUFs that belong to a model rather than being one.
    """
    return _snapshot_gguf_files(model_id, drafts=False)


def find_model_draft_files(model_id: str) -> list[Path]:
    """Find the speculative decoding sidecars cached for a model ID."""
    return _snapshot_gguf_files(model_id, drafts=True)


def find_model_draft_file(model_id: str) -> Path | None:
    """Return the cached sidecar that goes with a model ID's quant."""
    return _pick_model_draft(model_id, find_model_draft_files(model_id))


def find_model_mtp_file(model_id: str) -> Path | None:
    """Return the cached MTP head that goes with a model ID's quant.

    Other draft kinds (``dspark-``, ``dflash-``, ``eagle3-``) want their own
    ``spec-type`` and must never be wired in as ``draft-mtp``.
    """
    heads = [
        f for f in find_model_draft_files(model_id)
        if f.name.lower().startswith("mtp-")
    ]
    return _pick_model_draft(model_id, heads)


def _pick_model_draft(model_id: str, files: list[Path]) -> Path | None:
    source = model_source(model_id)[1]
    quant = source.rsplit(":", 1)[1] if ":" in source else ""
    chosen = pick_draft_file([str(f) for f in files], quant)
    return Path(chosen) if chosen is not None else None


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
