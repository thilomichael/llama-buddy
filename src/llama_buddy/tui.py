"""Shared TUI utilities for interactive menus."""

from __future__ import annotations

import select
import sys


def _parse_key(fd: int) -> str:
    """Parse a single keypress from a raw-mode file descriptor."""
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
    if ch == "\x0f":
        return "ctrl-o"
    if ch == "\x7f":
        return "backspace"
    return ch


def read_key() -> str:
    """Read a single keypress (blocking), handling arrow key escape sequences."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return _parse_key(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_timeout(timeout: float = 0.1) -> str | None:
    """Read a single keypress with timeout. Returns None if no key pressed."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        return _parse_key(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def require_tty() -> None:
    """Exit if stdin is not a terminal."""
    if not sys.stdin.isatty():
        from rich.console import Console

        Console().print("No interactive terminal available.", style="red")
        raise SystemExit(1)
