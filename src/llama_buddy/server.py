"""Server lifecycle management (start, stop, status)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

from rich.console import Console

from llama_buddy.config import (
    LOG_FILE,
    PRESET_FILE,
    ensure_config_dir,
    read_pid,
    remove_pid,
    write_pid,
)

console = Console()


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_server_binary() -> str | None:
    return shutil.which("llama-server")


def start(extra_args: list[str] | None = None) -> None:
    from llama_buddy.settings import load_settings

    pid = read_pid()
    if pid is not None and is_process_running(pid):
        console.print(
            f"llama-server is already running [dim](PID {pid})[/dim].",
            style="yellow",
        )
        return

    binary = find_server_binary()
    if binary is None:
        console.print(
            "llama-server not found. Please install llama.cpp.",
            style="red",
        )
        raise SystemExit(1)

    if not PRESET_FILE.exists():
        console.print(f"No preset file at {PRESET_FILE}", style="red")
        console.print(
            "Add models first with: [bold]llb download <model>[/bold]"
        )
        raise SystemExit(1)

    settings = load_settings()
    ensure_config_dir()

    log_fh = open(LOG_FILE, "a")

    cmd = [binary, "--models-preset", str(PRESET_FILE)]
    cmd.extend(settings.to_server_args())
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_pid(proc.pid)
    console.print(
        f"llama-server started [dim](PID {proc.pid})[/dim].", style="green"
    )


def stop() -> None:
    pid = read_pid()
    if pid is None:
        console.print("llama-server is not running.", style="yellow")
        return

    if not is_process_running(pid):
        console.print(
            f"llama-server [dim](PID {pid})[/dim] is not running. "
            "Cleaning up.",
            style="yellow",
        )
        remove_pid()
        return

    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not is_process_running(pid):
            break
        time.sleep(0.25)
    else:
        os.kill(pid, signal.SIGKILL)

    remove_pid()
    console.print("llama-server stopped.", style="green")


def restart(extra_args: list[str] | None = None) -> None:
    stop()
    start(extra_args)


def is_running() -> bool:
    """Check if the llama-server is currently running."""
    pid = read_pid()
    return pid is not None and is_process_running(pid)


def restart_if_running() -> None:
    """Restart the server if it is currently running."""
    if is_running():
        console.print("Restarting llama-server…", style="dim")
        restart()


def status() -> None:
    pid = read_pid()
    if pid is None or not is_process_running(pid):
        if pid is not None:
            remove_pid()
        console.print("llama-server is [red]not running[/red].")
        return

    console.print(
        f"llama-server is [green]running[/green] [dim](PID {pid})[/dim]."
    )
