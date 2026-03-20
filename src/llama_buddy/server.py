"""Server lifecycle management (start, stop, status)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

from llama_buddy.config import (
    DEFAULT_IDLE_SECONDS,
    LOG_FILE,
    PRESET_FILE,
    ensure_config_dir,
    read_pid,
    remove_pid,
    write_pid,
)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_server_binary() -> str | None:
    return shutil.which("llama-server")


def start(extra_args: list[str] | None = None) -> None:
    pid = read_pid()
    if pid is not None and is_process_running(pid):
        print(f"llama-server is already running (PID {pid}).")
        return

    binary = find_server_binary()
    if binary is None:
        print("Error: llama-server not found. Install it with: brew install llama.cpp")
        raise SystemExit(1)

    if not PRESET_FILE.exists():
        print(f"Error: No preset file found at {PRESET_FILE}")
        print("Add models first with: llb download <model>")
        raise SystemExit(1)

    ensure_config_dir()
    log_fh = open(LOG_FILE, "a")

    cmd = [
        binary,
        "--jinja",
        "--sleep-idle-seconds", str(DEFAULT_IDLE_SECONDS),
        "--models-preset", str(PRESET_FILE),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_pid(proc.pid)
    print(f"llama-server started (PID {proc.pid}).")


def stop() -> None:
    pid = read_pid()
    if pid is None:
        print("llama-server is not running (no PID file).")
        return

    if not is_process_running(pid):
        print(f"llama-server (PID {pid}) is not running. Cleaning up PID file.")
        remove_pid()
        return

    os.kill(pid, signal.SIGTERM)
    # Wait briefly for graceful shutdown
    for _ in range(20):
        if not is_process_running(pid):
            break
        time.sleep(0.25)
    else:
        os.kill(pid, signal.SIGKILL)

    remove_pid()
    print("llama-server stopped.")


def restart(extra_args: list[str] | None = None) -> None:
    stop()
    start(extra_args)


def status() -> None:
    pid = read_pid()
    if pid is None or not is_process_running(pid):
        if pid is not None:
            remove_pid()
        print("llama-server is not running.")
        return

    print(f"llama-server is running (PID {pid}).")
