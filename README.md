# llama-buddy

A CLI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp) providing an ollama-like experience on macOS.

## Features

- Start/stop `llama-server` as a background daemon
- Multi-model router mode with automatic load/unload
- Download models from HuggingFace with auto-generated aliases
- List models with status and disk size
- Inspect GGUF metadata and embedded sampling parameters
- Simple preset-based configuration (`models.ini`)

## Requirements

- macOS
- Python 3.10+
- llama.cpp installed via Homebrew: `brew install llama.cpp`

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

# Check server status
llb status

# View model metadata
llb info ministral-3b

# Tail server logs
llb logs

# Stop the server
llb stop
```

## Commands

| Command | Description |
|---------|-------------|
| `llb start [args...]` | Start llama-server in the background. Extra args are forwarded. |
| `llb stop` | Stop the running server. |
| `llb status` | Show whether the server is running. |
| `llb models` | List all configured models with aliases, status, and size. |
| `llb download <model> [--alias NAME]` | Download a model and add it to the preset file. |
| `llb remove <model> [--delete-files]` | Remove a model from the preset (optionally delete files). |
| `llb info <model>` | Show GGUF metadata for a model. |
| `llb logs` | Tail the server log file. |

## Configuration

Config files live in `~/.config/llama/`:

- **`models.ini`** -- llama-server preset file (INI format with HF repo IDs as sections)
- **`server.pid`** -- PID of the running server
- **`server.log`** -- Server stdout/stderr

Models are cached in `~/Library/Caches/llama.cpp` (the llama.cpp default on macOS).

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
