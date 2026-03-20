# llama-buddy

CLI wrapper for llama.cpp (`llb` command). Wraps `llama-server` with preset-based multi-model routing.

## Project structure

```
src/llama_buddy/
  cli.py        — Entry point, argparse, subcommand dispatch
  config.py     — Paths, preset I/O, cache dir detection, manifest sync, model groups
  server.py     — Server lifecycle (start/stop/restart/status)
  models.py     — Model listing with Rich table, GGUF metadata, tree-grouped display
  download.py   — Download via llama-cli, preset registration, removal
  info.py       — GGUF metadata display with Rich panels
  select.py     — Interactive model selector with search/filter
  settings.py   — Global + per-model settings editor (Rich Live TUI)
  tui.py        — Shared TUI utilities (read_key, require_tty)
  gguf.py       — Pure Python GGUF metadata parser
```

## Key conventions

- **uv** for project management, building, and publishing to PyPI
- **Rich** for all terminal output (Console, Table, Panel, Live)
- **No macOS-specific error messages** — keep install instructions platform-agnostic
- **INI preset file** (`~/.config/llama/models.ini`) — sections are HF repo IDs, `version = 1` top-level key requires `[DEFAULT]\n` prepend hack in `read_preset()`
- **Settings** — global in `settings.json`, per-model in `models.ini` using llama-server's native INI keys
- **Cache sync** — `sync_preset_with_cache()` runs on every CLI command, adds missing manifest models to INI
- **Model groups** — manifests map model IDs to GGUF files; models sharing a file are grouped in display
- **Flat cache** — llama.cpp stores files as `org_repo-GGUF_filename.gguf` (not nested dirs)
- **Interactive menus** use `auto_refresh=False` with manual `refresh=True` to avoid infinite reprint
- **Nested Live contexts** — must exit outer Live before launching sub-menus (break out of loop, run sub-menu, re-enter)

## Running

```bash
uv sync              # install deps
uv run llb <cmd>     # run CLI
uv run pytest        # run tests (44 tests)
uv run ruff check src/ tests/  # lint
```

## Testing

Tests in `tests/` use `monkeypatch` to override config paths (`PRESET_FILE`, `CONFIG_DIR`, `PID_FILE`, `get_cache_dir`). No network or server required for tests.
