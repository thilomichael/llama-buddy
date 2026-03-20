"""Shared TUI utilities for interactive menus."""

from __future__ import annotations

import sys


def read_key() -> str:
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
            return ch
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        if ch == "\x7f":
            return "backspace"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def require_tty() -> None:
    """Exit if stdin is not a terminal."""
    if not sys.stdin.isatty():
        from rich.console import Console

        Console().print("No interactive terminal available.", style="red")
        raise SystemExit(1)
