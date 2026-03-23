"""Background watchdog that unloads idle models from the llama-server router.

The router's child servers go to sleep after --sleep-idle-seconds of inactivity
(reported via /props is_sleeping), but the router never unloads them. This
watchdog polls loaded models and calls /models/unload on sleeping ones.

Intended to be spawned as a detached subprocess by server.start().
Usage: python -m llama_buddy.watchdog <port> <poll_interval>
"""

from __future__ import annotations

import sys
import time

import httpx


def _get_child_port(model: dict) -> int | None:
    """Extract the child server port from the model status args."""
    args = model.get("status", {}).get("args", [])
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
                if port > 0:
                    return port
            except ValueError:
                pass
    return None


def _is_sleeping(port: int) -> bool:
    """Check if a child server is sleeping."""
    try:
        resp = httpx.get(f"http://localhost:{port}/props", timeout=3)
        return resp.json().get("is_sleeping", False)
    except Exception:
        return False


def _unload_model(router_port: int, model_id: str) -> bool:
    """Ask the router to unload a model."""
    try:
        resp = httpx.post(
            f"http://localhost:{router_port}/models/unload",
            json={"model": model_id},
            timeout=10,
        )
        return resp.json().get("success", False)
    except Exception:
        return False


def run(router_port: int, poll_interval: int) -> None:
    """Main watchdog loop. Runs until the router becomes unreachable."""
    while True:
        time.sleep(poll_interval)
        try:
            resp = httpx.get(
                f"http://localhost:{router_port}/models", timeout=5
            )
            models = resp.json().get("data", [])
        except Exception:
            # Server gone — exit
            break

        for model in models:
            if model.get("status", {}).get("value") != "loaded":
                continue
            child_port = _get_child_port(model)
            if child_port is None:
                continue
            if _is_sleeping(child_port):
                _unload_model(router_port, model["id"])


if __name__ == "__main__":
    router_port = int(sys.argv[1])
    poll_interval = int(sys.argv[2])
    run(router_port, poll_interval)
