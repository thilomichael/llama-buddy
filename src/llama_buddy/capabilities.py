"""Detection of which CLI flags the installed llama-server supports.

llama-server refuses to start when handed a flag it does not know, so a
setting written against a newer build takes the whole server down. Parsing
``llama-server --help`` once tells us which settings are safe to offer.

The accepted values of a flag also move between builds: build 10344 takes
``--load-mode mmap|mlock|...`` while master added ``auto``, and passing a
value the build does not know aborts the same way an unknown flag does. The
help lists the values of such flags as ``- name: description`` lines under
the flag, so both are read out of the same text.

The parse costs ~65ms, so the result is cached in ``capabilities.json``,
keyed on the binary's path, size and mtime. Replacing the binary invalidates
the cache on its own.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from llama_buddy.config import CONFIG_DIR, ensure_config_dir

CAPABILITIES_FILE = CONFIG_DIR / "capabilities.json"

_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")
# A value line sits in the description column: "        - mmap: memory-map …"
_VALUE_RE = re.compile(r"^\s{8,}-\s*([A-Za-z0-9][A-Za-z0-9+_-]*):")


def _binary_key(binary: str) -> str:
    st = Path(binary).stat()
    return f"{binary}:{st.st_size}:{int(st.st_mtime)}"


def _run_help(binary: str) -> str:
    result = subprocess.run(
        [binary, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def parse_help(text: str) -> tuple[set[str], dict[str, list[str]]]:
    """Return the long flags in a help text and the values each one accepts."""
    flags = set(_FLAG_RE.findall(text))
    values: dict[str, list[str]] = {}
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("-") and indent < 8:
            current = _FLAG_RE.findall(line)
            continue
        m = _VALUE_RE.match(line)
        if m and current:
            for flag in current:
                values.setdefault(flag, []).append(m.group(1))

    return flags, values


def _probe(binary: str) -> tuple[set[str], dict[str, list[str]]] | None:
    try:
        text = _run_help(binary)
    except (subprocess.TimeoutExpired, OSError):
        return None
    flags, values = parse_help(text)
    if not flags:
        return None
    return flags, values


@lru_cache(maxsize=1)
def _capabilities() -> tuple[frozenset[str], dict[str, list[str]]] | None:
    """Return the installed llama-server's flags and their accepted values.

    Returns None when the binary is missing or could not be probed, which
    callers treat as "assume everything is supported" rather than hiding
    every setting.
    """
    binary = shutil.which("llama-server")
    if binary is None:
        return None

    try:
        key = _binary_key(binary)
    except OSError:
        return None

    if CAPABILITIES_FILE.exists():
        try:
            cached = json.loads(CAPABILITIES_FILE.read_text())
            if cached.get("key") == key:
                return frozenset(cached["flags"]), cached.get("values", {})
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass

    probed = _probe(binary)
    if probed is None:
        return None
    flags, values = probed

    try:
        ensure_config_dir()
        CAPABILITIES_FILE.write_text(
            json.dumps(
                {"key": key, "flags": sorted(flags), "values": values},
                indent=2,
            )
            + "\n"
        )
    except OSError:
        pass

    return frozenset(flags), values


def supported_flags() -> frozenset[str] | None:
    """Return the long flags the installed llama-server accepts, or None."""
    caps = _capabilities()
    return None if caps is None else caps[0]


def has_flag(flag: str) -> bool:
    """Check whether the installed llama-server accepts a long flag.

    Unknown capabilities count as supported, so a failure to probe never
    hides a setting the user relies on.
    """
    flags = supported_flags()
    if flags is None:
        return True
    return flag in flags


def has_value(flag: str, value: str) -> bool:
    """Check whether a flag takes a given value on the installed build.

    Only flags whose help lists their values are checked; anything else
    counts as accepted.
    """
    caps = _capabilities()
    if caps is None:
        return True
    known = caps[1].get(flag)
    if not known:
        return True
    return value in known


def supported_choices(flag: str, choices: list[str]) -> list[str]:
    """Drop the choices the installed llama-server would reject."""
    kept = [c for c in choices if has_value(flag, c)]
    return kept or choices


def filter_supported(meta: dict[str, dict]) -> dict[str, dict]:
    """Drop settings whose ``flag`` the installed llama-server does not know.

    Settings that survive get their choice list narrowed to the values this
    build accepts.
    """
    result: dict[str, dict] = {}
    for key, entry in meta.items():
        flag = entry.get("flag")
        if flag is not None and not has_flag(flag):
            continue
        if flag is not None and entry.get("choices"):
            entry = {
                **entry,
                "choices": supported_choices(flag, entry["choices"]),
            }
        result[key] = entry
    return result
