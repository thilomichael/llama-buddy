"""Interactive model selection menu with arrow-key navigation and search."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.text import Text

from llama_buddy.config import read_preset
from llama_buddy.models import get_model_name
from llama_buddy.tui import read_key, require_tty

console = Console()


def _render_menu(
    entries: list[tuple[str, str]],
    selected: int,
    title: str,
    query: str,
) -> Text:
    """Render the menu as a rich Text object."""
    text = Text()
    text.append(f"  {title}\n\n", style="bold")

    if not entries:
        text.append("    No matches\n", style="dim italic")
    else:
        for i, (_section, name) in enumerate(entries):
            if i == selected:
                text.append("  > ", style="bold cyan")
                text.append(f"{name}\n", style="bold cyan")
            else:
                text.append(f"    {name}\n", style="dim")

    text.append("\n")
    if query:
        text.append("  Search: ", style="dim italic")
        text.append(query, style="bold")
        text.append("_", style="bold blink")
        text.append("  Esc clear", style="dim italic")
    else:
        text.append(
            "  ↑/↓ navigate  Enter select  Type to search  Ctrl-C quit",
            style="dim italic",
        )
    return text


def get_model_entries() -> list[tuple[str, str]]:
    """Get sorted list of (model_id, display_name) from preset file."""
    preset = read_preset()
    sections = [s for s in preset.sections() if s != "*"]
    entries: list[tuple[str, str]] = []
    for section in sections:
        name = get_model_name(section) or section
        entries.append((section, name))
    entries.sort(key=lambda e: e[1].lower())
    return entries


def _filter_entries(
    all_entries: list[tuple[str, str]], query: str
) -> list[tuple[str, str]]:
    """Filter entries by query (case-insensitive substring match)."""
    if not query:
        return all_entries
    q = query.lower()
    return [e for e in all_entries if q in e[1].lower()]


def select_model(title: str = "Select a model") -> str:
    """Show an interactive menu to pick a model from the preset file.

    Returns the model ID (preset section name).
    """
    with console.status("Loading models…", spinner="dots"):
        all_entries = get_model_entries()

    if not all_entries:
        console.print(
            "No models configured. Add one with: llb download <model>",
            style="yellow",
        )
        raise SystemExit(1)

    require_tty()
    query = ""
    entries = all_entries
    selected = 0

    with Live(
        _render_menu(entries, selected, title, query),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = read_key()
            if key == "up" and selected > 0:
                selected -= 1
            elif key == "down" and selected < len(entries) - 1:
                selected += 1
            elif key in ("enter", "right"):
                if entries:
                    return entries[selected][0]
            elif key == "left":
                continue
            elif key == "ctrl-c":
                raise SystemExit(0)
            elif key == "\x1b":
                # Esc clears search
                if query:
                    query = ""
                    entries = all_entries
                    selected = min(selected, max(len(entries) - 1, 0))
            elif key == "backspace":
                if query:
                    query = query[:-1]
                    entries = _filter_entries(all_entries, query)
                    selected = min(selected, max(len(entries) - 1, 0))
            elif len(key) == 1 and key.isprintable():
                query += key
                entries = _filter_entries(all_entries, query)
                selected = min(selected, max(len(entries) - 1, 0))
            else:
                continue

            live.update(
                _render_menu(entries, selected, title, query),
                refresh=True,
            )
