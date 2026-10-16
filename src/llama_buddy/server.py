"""Server lifecycle management (start, stop, status)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time

import httpx

from llama_buddy.config import (
    CONFIG_DIR,
    LOG_FILE,
    ensure_config_dir,
    get_port,
    read_pid,
    remove_pid,
    write_pid,
)
from llama_buddy.console import console

WATCHDOG_PID_FILE = CONFIG_DIR / "watchdog.pid"
EXTRA_ARGS_FILE = CONFIG_DIR / "extra_args.json"


def _save_extra_args(extra_args: list[str] | None) -> None:
    ensure_config_dir()
    if extra_args:
        EXTRA_ARGS_FILE.write_text(json.dumps(extra_args))
    else:
        EXTRA_ARGS_FILE.unlink(missing_ok=True)


def _load_extra_args() -> list[str] | None:
    if EXTRA_ARGS_FILE.exists():
        try:
            args = json.loads(EXTRA_ARGS_FILE.read_text())
            if isinstance(args, list):
                return args
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_server_binary() -> str | None:
    return shutil.which("llama-server")


def _remove_stale_watchdog() -> None:
    """Kill a watchdog process left over from a version that still spawned one.

    Releases from before 0.1.14 ran ``llama_buddy.watchdog`` alongside the
    server. Sleeping already frees the model weights, so the watchdog is gone;
    this stops one that an older ``llb start`` left behind.
    """
    if not WATCHDOG_PID_FILE.exists():
        return
    text = WATCHDOG_PID_FILE.read_text().strip()
    WATCHDOG_PID_FILE.unlink(missing_ok=True)
    if not text:
        return
    try:
        pid = int(text)
    except ValueError:
        return
    if is_process_running(pid):
        os.kill(pid, signal.SIGTERM)


EFFECTIVE_PRESET_FILE = CONFIG_DIR / "models.effective.ini"


def _build_effective_preset() -> None:
    """Write the copy of the preset that the router actually reads.

    Two things differ from the user's file. Context is clamped to each
    model's native maximum, since llama-server allocates KV cache for the
    full ctx-size even past what the model was trained on. And settings the
    installed llama-server does not know are dropped: it refuses to start on
    an unknown flag or value, which would take every model down rather than
    lose one setting.
    """
    from llama_buddy.config import all_model_ids, key_names, read_preset
    from llama_buddy.models import configured_ctx, get_model_meta

    preset = read_preset()

    for model_id in all_model_ids(preset):
        ctx = configured_ctx(preset, model_id)
        if ctx <= 0:
            continue
        native_str = get_model_meta(model_id).get("context_length", "")
        if not native_str:
            continue
        native = int(native_str.replace(",", ""))
        if ctx > native:
            preset.remove_options(model_id, key_names("c"))
            preset.set(model_id, "c", str(native))

    _drop_unsupported(preset)

    ensure_config_dir()
    EFFECTIVE_PRESET_FILE.write_text(preset.text())


def _drop_unsupported(preset) -> None:
    """Remove settings the installed llama-server would refuse to start on."""
    from llama_buddy.capabilities import has_flag, has_value
    from llama_buddy.config import key_names
    from llama_buddy.settings import GLOBAL_SETTING_META, MODEL_SETTING_META

    flags = {
        name: meta["flag"]
        for key, meta in {**GLOBAL_SETTING_META, **MODEL_SETTING_META}.items()
        if "flag" in meta
        for name in key_names(key)
    }
    ngl_names = key_names("ngl")
    for section in preset.sections():
        for key in preset.options(section):
            # Older llb offered "none", which llama-server cannot parse.
            if key in ngl_names and preset.get(section, key) == "none":
                preset.set(section, key, "0")
            flag = flags.get(key)
            if flag is None:
                continue
            value = preset.get(section, key, "")
            if not has_flag(flag) or (
                flag in ("--load-mode", "--lazy-mode")
                and not has_value(flag, value)
            ):
                preset.remove_option(section, key)


def start(extra_args: list[str] | None = None) -> None:
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

    ensure_config_dir()
    _remove_stale_watchdog()
    _build_effective_preset()
    _save_extra_args(extra_args)

    log_fh = open(LOG_FILE, "a")

    # Model settings travel in the preset: a flag given here would override
    # the same setting in every model's section.
    cmd = [
        binary, "--models-preset", str(EFFECTIVE_PRESET_FILE),
        "--port", str(get_port()),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()
    write_pid(proc.pid)
    console.print(
        f"llama-server started [dim](PID {proc.pid})[/dim].", style="green"
    )


def stop() -> None:
    _remove_stale_watchdog()

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
    EFFECTIVE_PRESET_FILE.unlink(missing_ok=True)
    EXTRA_ARGS_FILE.unlink(missing_ok=True)
    console.print("llama-server stopped.", style="green")


def restart(extra_args: list[str] | None = None) -> None:
    if extra_args is None:
        extra_args = _load_extra_args()
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


def reload_if_running() -> None:
    """Make a running router pick up changes to the preset and the cache.

    ``GET /models?reload=1`` re-reads every model source in place: models
    whose settings changed are unloaded and come back with the new ones on
    their next request, the rest keep running. A router that predates it
    is restarted instead, if llb started it. A router started some other
    way reads the user's preset directly and only needs the reload.
    """
    managed = is_running()
    if managed:
        _build_effective_preset()
    url = f"http://localhost:{get_port()}/models"
    try:
        models = httpx.get(url, timeout=5).json().get("data", [])
    except (httpx.HTTPError, ValueError):
        if managed:
            restart_if_running()
        return
    # The source field came with the model management API (June 2026),
    # a month after ?reload=1; an older router ignores the parameter.
    if any("source" in m for m in models):
        try:
            httpx.get(url, params={"reload": 1}, timeout=30).raise_for_status()
            console.print("Reloaded llama-server's models.", style="dim")
            return
        except httpx.HTTPError:
            pass
    if managed:
        restart_if_running()
    else:
        console.print(
            "Restart llama-server to apply the change.", style="yellow",
        )


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
