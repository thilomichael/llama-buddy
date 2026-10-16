"""Round-trip editing of llama.cpp INI files.

The preset file and llama.cpp's ``config.ini`` are often written by hand, so
llb changes only the lines it means to: comments, blank lines, key spelling
and order all survive a write. configparser would drop the comments and
lowercase every key, and llama.cpp keys are case-sensitive
(``LLAMA_ARG_CACHE_RAM``).

Parsing follows the grammar in llama.cpp's common/preset.cpp: a key is
``[A-Za-z_][A-Za-z0-9_.-]*``, a value runs up to the first ``;`` or ``#``,
and keys before the first section header belong to the default section,
which this module calls ``None``.
"""

from __future__ import annotations

import re
from pathlib import Path

_HEADER_RE = re.compile(r"^\s*\[\s*([^\]]*?)\s*\]")
_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*=(.*)$")


def _parse_value(raw: str) -> str:
    return re.split(r"[;#]", raw, maxsplit=1)[0].strip()


class IniFile:
    """An INI document that remembers its own text.

    Each section is a header line plus the lines under it. Reads go through
    the key lines; writes rewrite or insert single lines and leave the rest
    alone. The method names follow configparser's, which the callers used
    before.
    """

    def __init__(self, text: str = "") -> None:
        self._sections: list[tuple[str | None, list[str]]] = [(None, [])]
        for line in text.splitlines():
            m = _HEADER_RE.match(line)
            if m:
                self._sections.append((m.group(1), [line]))
            else:
                self._sections[-1][1].append(line)

    @classmethod
    def load(cls, path: Path) -> IniFile:
        try:
            return cls(path.read_text())
        except OSError:
            return cls()

    # -- sections ----------------------------------------------------------

    def sections(self) -> list[str]:
        return [name for name, _ in self._sections if name is not None]

    def has_section(self, section: str | None) -> bool:
        return self._lines(section) is not None

    def __contains__(self, section: str) -> bool:
        return self.has_section(section)

    def add_section(self, section: str) -> None:
        if self.has_section(section):
            return
        last = self._sections[-1][1]
        if last and last[-1].strip():
            last.append("")
        self._sections.append((section, [f"[{section}]"]))

    def remove_section(self, section: str) -> bool:
        for i, (name, _) in enumerate(self._sections):
            if name == section:
                del self._sections[i]
                return True
        return False

    # -- options -----------------------------------------------------------

    def options(self, section: str | None) -> list[str]:
        keys: list[str] = []
        for line in self._lines(section) or []:
            m = _KEY_RE.match(line)
            if m and m.group(1) not in keys:
                keys.append(m.group(1))
        return keys

    def has_option(self, section: str | None, key: str) -> bool:
        return key in self.options(section)

    def get(
        self, section: str | None, key: str, fallback: str | None = None,
    ) -> str | None:
        value = fallback
        for line in self._lines(section) or []:
            m = _KEY_RE.match(line)
            if m and m.group(1) == key:
                value = _parse_value(m.group(2))
        return value

    def get_any(
        self, section: str | None, keys: tuple[str, ...],
        fallback: str | None = None,
    ) -> str | None:
        """Return the value under the first of several spellings of a key.

        llama.cpp accepts every name of a flag as a key (``c``, ``ctx-size``,
        ``LLAMA_ARG_CTX_SIZE``), so a hand-written file may use any of them.
        """
        for key in keys:
            value = self.get(section, key)
            if value is not None:
                return value
        return fallback

    def set(self, section: str | None, key: str, value: str) -> None:
        """Set a key, adding the section if needed.

        An existing line keeps its position; a new one goes after the
        section's last key, so blank lines and comments that lead into the
        next section stay with it.
        """
        if section is not None:
            self.add_section(section)
        lines = self._lines(section)
        assert lines is not None
        new_line = f"{key} = {value}"
        found = [
            i for i, line in enumerate(lines)
            if (m := _KEY_RE.match(line)) and m.group(1) == key
        ]
        if found:
            lines[found[-1]] = new_line
            for i in reversed(found[:-1]):
                del lines[i]
            return
        keys = [i for i, line in enumerate(lines) if _KEY_RE.match(line)]
        if keys:
            insert_at = keys[-1] + 1
        elif section is not None:
            insert_at = 1  # right below the header
        else:
            insert_at = len(lines)
            while insert_at > 0 and not lines[insert_at - 1].strip():
                insert_at -= 1
        lines.insert(insert_at, new_line)

    def remove_option(self, section: str | None, key: str) -> bool:
        lines = self._lines(section)
        if lines is None:
            return False
        kept = [
            line for line in lines
            if not ((m := _KEY_RE.match(line)) and m.group(1) == key)
        ]
        removed = len(kept) != len(lines)
        lines[:] = kept
        return removed

    def remove_options(self, section: str | None, keys: tuple[str, ...]) -> bool:
        removed = False
        for key in keys:
            removed = self.remove_option(section, key) or removed
        return removed

    # -- output ------------------------------------------------------------

    def text(self) -> str:
        lines = [line for _, body in self._sections for line in body]
        body = "\n".join(lines).strip("\n")
        return body + "\n" if body.strip() else ""

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text())

    def _lines(self, section: str | None) -> list[str] | None:
        for name, lines in self._sections:
            if name == section:
                return lines
        return None
