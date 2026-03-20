"""Configuration paths and preset file management."""

from __future__ import annotations

import configparser
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "llama"
PRESET_FILE = CONFIG_DIR / "models.ini"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"

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


def find_model_gguf_files(model_id: str) -> list[Path]:
    """Find all .gguf files in cache for a model ID (excluding mmproj).

    Cache is flat: files are named org_repo-GGUF_filename.gguf
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return []
    base_repo = model_id.split(":")[0]
    prefix = base_repo.replace("/", "_") + "_"
    return sorted(
        f
        for f in cache_dir.glob(f"{prefix}*.gguf")
        if "mmproj" not in f.name
    )
