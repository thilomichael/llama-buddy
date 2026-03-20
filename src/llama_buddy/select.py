"""Interactive model selection menu with arrow-key navigation."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.live import Live
from rich.text import Text

from llama_buddy.config import read_preset
from llama_buddy.models import get_model_name

console = Console()


def _read_key() -> str:
    """Read a single keypress, handling arrow key escape sequences."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            if seq == "[C":
                return "right"
            if seq == "[D":
                return "left"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        if ch == "q":
            return "quit"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_menu(
    entries: list[tuple[str, str]], selected: int, title: str
) -> Text:
    """Render the menu as a rich Text object."""
    text = Text()
    text.append(f"  {title}\n\n", style="bold")
    for i, (_section, name) in enumerate(entries):
        if i == selected:
            text.append("  > ", style="bold cyan")
            text.append(f"{name}\n", style="bold cyan")
        else:
            text.append(f"    {name}\n", style="dim")
    text.append("\n")
    text.append("  ↑/↓ navigate  Enter select  q quit", style="dim italic")
    return text


def select_model() -> str:
    """Show an interactive menu to pick a model from the preset file.

    Returns the model ID (preset section name).
    """
    preset = read_preset()
    sections = [s for s in preset.sections() if s != "*"]

    if not sections:
        console.print(
            "No models configured. Add one with: llb download <model>",
            style="yellow",
        )
        raise SystemExit(1)

    entries: list[tuple[str, str]] = []
    for section in sections:
        name = get_model_name(section) or section
        entries.append((section, name))
    entries.sort(key=lambda e: e[1].lower())

    if not sys.stdin.isatty():
        console.print("No interactive terminal available.", style="red")
        raise SystemExit(1)

    selected = 0

    with Live(
        _render_menu(entries, selected, "Select a model"),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        while True:
            key = _read_key()
            if key == "up" and selected > 0:
                selected -= 1
            elif key == "down" and selected < len(entries) - 1:
                selected += 1
            elif key in ("enter", "right"):
                return entries[selected][0]
            elif key == "left":
                continue
            elif key in ("quit", "ctrl-c"):
                raise SystemExit(0)

            live.update(
                _render_menu(entries, selected, "Select a model"),
                refresh=True,
            )
