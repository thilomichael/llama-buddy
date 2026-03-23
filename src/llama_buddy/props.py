"""Display active server properties for a loaded model."""

from __future__ import annotations

import httpx
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from llama_buddy.config import parse_child_ports, resolve_model
from llama_buddy.console import console

# Sampling params in display order (subset that's most relevant)
SAMPLING_KEYS = [
    "temperature", "dynatemp_range", "dynatemp_exponent",
    "top_k", "top_p", "min_p", "typical_p",
    "repeat_penalty", "repeat_last_n",
    "presence_penalty", "frequency_penalty",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "mirostat", "mirostat_tau", "mirostat_eta",
    "top_n_sigma",
    "xtc_probability", "xtc_threshold",
    "seed",
]

# Params that are effectively disabled at these values (skip for cleaner output)
_DISABLED = {
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "typical_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "mirostat": 0,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
}


def _fmt(value: object) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def get_loaded_model_ids(port: int) -> set[str]:
    """Query /models and return IDs of models with status 'loaded'."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"http://localhost:{port}/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
    except (httpx.HTTPError, httpx.ConnectError):
        return set()
    return {
        m["id"]
        for m in data
        if isinstance(m.get("status"), dict)
        and m["status"].get("value") == "loaded"
    }


def _fetch_props(port: int) -> dict:
    """Fetch /props from a child server port."""
    with httpx.Client(timeout=5) as client:
        resp = client.get(f"http://localhost:{port}/props")
        resp.raise_for_status()
        return resp.json()


def show_props(model_id_or_alias: str) -> None:
    """Show the active sampling parameters for a loaded model."""
    model_id = resolve_model(model_id_or_alias)
    if model_id is None:
        model_id = model_id_or_alias

    ports = parse_child_ports()
    port = ports.get(model_id)
    if port is None:
        console.print(
            f"No child server found for '{model_id}'.\n"
            "Is the server running and has this model been loaded?",
            style="red",
        )
        raise SystemExit(1)

    try:
        data = _fetch_props(port)
    except (httpx.HTTPError, httpx.ConnectError) as e:
        console.print(
            f"Failed to reach child server on port {port}: {e}",
            style="red",
        )
        raise SystemExit(1)

    gen = data.get("default_generation_settings", {})
    params = gen.get("params", {})
    title = data.get("model_alias", model_id)

    # General info table
    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()

    info.add_row("Model", data.get("model_alias", model_id))
    info.add_row("Port", str(port))
    if "n_ctx" in gen:
        info.add_row("Context", f"{gen['n_ctx']:,}")
    info.add_row("Slots", str(data.get("total_slots", "?")))
    if "is_sleeping" in data:
        info.add_row("Sleeping", _fmt(data["is_sleeping"]))
    samplers = params.get("samplers", [])
    if samplers:
        info.add_row("Samplers", " → ".join(samplers))

    # Sampling params table (skip disabled values)
    sampling = Table(show_header=False, box=None, padding=(0, 2))
    sampling.add_column(style="bold")
    sampling.add_column()

    for key in SAMPLING_KEYS:
        if key not in params:
            continue
        val = params[key]
        if key in _DISABLED and val == _DISABLED[key]:
            continue
        sampling.add_row(key, _fmt(val))

    body = Group(
        info,
        "",
        Panel(
            sampling,
            title="Sampling",
            border_style="dim",
            expand=False,
        ),
    )
    console.print(Panel(body, title=title, border_style="cyan"))
