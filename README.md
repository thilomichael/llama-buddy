# llama-buddy

A CLI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp) providing an ollama-like experience.

## Features

- Start/stop/restart `llama-server` as a background daemon
- Multi-model router mode with automatic load/unload
- Download models from HuggingFace
- Rich terminal UI with interactive model selector and search
- Inspect GGUF metadata and embedded sampling parameters
- Configurable global and per-model settings via interactive menus
- Preset-based configuration (`models.ini`) auto-synced with cache
- Tree-grouped model display showing which models share a GGUF file

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) installed and on your `PATH`

## Installation

```bash
pip install llama-buddy
```

Or install from source with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

This installs the `llb` command.

## Quick start

```bash
# Download a model
llb download mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M

# Start the server
llb start

# List models
llb models

# Open the web UI
llb open

# View model metadata (interactive selector when no model specified)
llb info

# Configure settings
llb settings

# Stop the server
llb stop
```

## Commands

| Command | Description |
|---------|-------------|
| `llb start [args...]` | Start llama-server in the background. Extra args are forwarded. |
| `llb stop` | Stop the running server. |
| `llb restart [args...]` | Restart the server. |
| `llb status` | Show whether the server is running. |
| `llb models` | List all models with status, size, and GGUF grouping. |
| `llb download <model> [--alias NAME]` | Download a model and add it to the preset file. |
| `llb remove <model> [--delete-files]` | Remove a model from the preset (optionally delete files). |
| `llb info [model]` | Show GGUF metadata. Opens interactive selector if no model given. |
| `llb open` | Open the llama-server web UI in the browser. |
| `llb settings` | Interactive editor for global and per-model settings. |
| `llb logs` | Tail the server log file. |

## Configuration

Config files live in `~/.config/llama/`:

| File | Description |
|------|-------------|
| `models.ini` | llama-server preset file (INI format with HF repo IDs as sections) |
| `settings.json` | Global server settings (port, context size, GPU layers, etc.) |
| `server.pid` | PID of the running server |
| `server.log` | Server stdout/stderr |

The preset file is automatically kept in sync with models in the llama.cpp cache.

### Per-model settings

Use `llb settings` → **Model Settings** to configure per-model overrides like context size, GPU layers, flash attention, aliases, or any custom llama-server parameter.

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/
```

## License

MIT
