<div align="center">

# llama-buddy

**A friendly CLI wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp)**

Manage, download, and serve local LLMs with a single command.
Think of it as an ollama-like experience built on top of `llama-server`.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llama-buddy)](https://pypi.org/project/llama-buddy/)

</div>

---

## Features

- **Background server** &mdash; start/stop/restart `llama-server` as a daemon
- **Multi-model routing** &mdash; uses llama.cpp's own preset file, so nothing llb writes is llb-specific
- **Interactive downloads** &mdash; search HuggingFace, pick a quant, download with progress and resume
- **Rich terminal UI** &mdash; tables, panels, interactive selectors, and live search
- **GGUF inspector** &mdash; view model metadata, architecture, and sampling parameters
- **Server props** &mdash; inspect active sampling parameters on loaded models
- **Sampling sync** &mdash; automatically applies GGUF-recommended sampling params to your preset
- **Per-model settings** &mdash; context size, GPU layers, flash attention, and more
- **Multi-token prediction** &mdash; models that ship an MTP head get speculative decoding switched on by themselves. A head shipped as a separate file is downloaded with the model and only wired in once the installed llama.cpp is confirmed to load it; the check runs again after every llama.cpp upgrade
- **Idle sleeping** &mdash; `llama-server` releases the weights of a model after a configurable idle timeout
- **VRAM tracking** &mdash; automatically parses server logs to show memory usage per model
- **Live changes** &mdash; downloads, removals and settings changes reload the running router's model list instead of restarting it

## Screenshots

<details open>
<summary><b>Model listing</b> &mdash; <code>llb models</code></summary>
<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/models.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/models.svg">
    <img alt="llb models" src="assets/models.svg" width="700">
  </picture>
</p>
</details>

<details open>
<summary><b>Interactive download</b> &mdash; <code>llb download</code></summary>
<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/download.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/download.svg">
    <img alt="llb download" src="assets/download.svg" width="700">
  </picture>
</p>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/download-quant.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/download-quant.svg">
    <img alt="llb download quantization" src="assets/download-quant.svg" width="700">
  </picture>
</p>
</details>

<details open>
<summary><b>Model info</b> &mdash; <code>llb info</code></summary>
<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/info.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/info.svg">
    <img alt="llb info" src="assets/info.svg" width="600">
  </picture>
</p>
</details>

## Installation

```bash
pipx install llama-buddy
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install llama-buddy
```

This installs the `llb` command into an isolated environment and adds it to your `PATH`.

### Prerequisites

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) installed and `llama-server` on your `PATH`

## Quick start

```bash
# Download a model (interactive search)
llb download

# Or specify directly
llb download mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M

# Start the server
llb start

# List all models
llb models

# Chat with a model (uses llama-cli)
llb chat

# Inspect model metadata
llb info

# Show active sampling params for a loaded model
llb props

# Apply GGUF-recommended sampling params to all models
llb info --apply-sampling

# Configure settings (interactive TUI)
llb settings

# Open the web UI in your browser
llb open

# Stop the server
llb stop
```

## Commands

| Command | Description |
|---------|-------------|
| `llb start` | Start `llama-server` in the background. Extra args are forwarded. |
| `llb stop` | Stop the running server. |
| `llb restart` | Restart the server. |
| `llb status` | Show whether the server is running. |
| `llb models` | List all models with status, size, VRAM usage, and grouping. Supports `--sort size`. |
| `llb download [model]` | Download a model. Interactive HF search when no model given. |
| `llb remove [model]` | Remove a model with confirmation dialog. `--keep-files` to preserve GGUFs. |
| `llb info [model]` | Show GGUF metadata. Interactive selector when no model given. |
| `llb info --apply-sampling [model]` | Write GGUF sampling params into the preset. All models when no model given. |
| `llb props [model]` | Show active server sampling params for a loaded model. |
| `llb settings` | Interactive editor for the preset's `[*]` section and per-model sections. |
| `llb chat [model]` | Interactive chat via `llama-cli`. Model selector when no model given. |
| `llb open` | Open the `llama-server` web UI in your browser. |
| `llb logs` | Tail the server log file. |

## Configuration

llb keeps no configuration format of its own. Settings live in llama.cpp's
[preset file](https://github.com/ggml-org/llama.cpp/blob/master/docs/preset.md),
which `llama-server --models-preset` reads, and the port lives in llama.cpp's
`config.ini`. Anything llb writes works the same without llb.

| File | Purpose |
|------|---------|
| `~/.config/llama/models.ini` | The preset file, unless you already have one (see below) |
| `~/.config/llama.cpp/config.ini` | llama.cpp's own config; llb only writes `port` there, and only when it is not 8080 |

The router lists every cached model by itself, so the preset only has a
section for a model whose settings differ from the defaults. `[*]` holds the
settings for all models; a model's own section overrides them. Defaults are
llama-server's own: context from the model (shrunk to fit memory), no idle
sleep. llb edits the file line by line, so comments and hand-written keys
survive.

### Using your own preset file

If you already point llama-server at a preset, llb uses that one instead.
It looks at `LLAMA_ARG_MODELS_PRESET`, then `models-preset` in
`~/.config/llama.cpp/config.ini`. Sections can give a model any name and
point it at an HF repo or a local file:

```ini
[*]
sleep-idle-seconds = 300

[coder]
hf-repo = unsloth/Qwen3.8-27B-GGUF:Q8_0
temp = 0.6

[laguna]
model = /data/models/laguna-s-2.1-Q4_K_M.gguf
```

### Per-model settings

Run `llb settings` and select **Model Settings** to configure per-model overrides:

- Context size, GPU layers, flash attention
- Custom aliases
- Any `llama-server` parameter

Other files in `~/.config/llama/` are llb's own state: `server.pid`,
`server.log`, `models.effective.ini` (the copy the router reads, with context
clamped to each model's maximum), and caches for VRAM usage, MTP detection
and llama-server's flags.

## Development

```bash
# Clone and install
git clone https://github.com/thilomichael/llama-buddy.git
cd llama-buddy
uv sync

# Run
uv run llb <command>

# Test
uv run pytest

# Lint
uv run ruff check src/ tests/
```

## License

MIT
