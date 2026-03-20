# llama - CLI wrapper for llama.cpp

A Python CLI tool wrapping `llama-server` (installed via Homebrew) to provide an ollama-like experience.

## Subcommands

### `llama start`
Start llama-server in the background (daemonized) with sensible defaults.
- Flags passed to llama-server: `--jinja --sleep-idle-seconds 300 --models-preset <preset>`
- Writes a PID file (e.g. `~/.config/llama/server.pid`)
- Logs to `~/.config/llama/server.log`
- Extra args forwarded to llama-server: `llama start --port 9090`
- Should refuse to start if already running

### `llama stop`
Stop the running llama-server (via PID file).

### `llama status`
Show whether the server is running, which port, uptime, and which model (if any) is currently loaded.

### `llama models`
Query the running server's `/models` endpoint and display a table:
```
MODEL                                               ALIAS         STATUS      SIZE
----------------------------------------------------------------------------------
mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M  ministral-3b  unloaded    2.0G
unsloth/gpt-oss-20b-GGUF                            gpt-oss-20b   loaded      10.8G
```
- Aliases come from the server API (`aliases` field)
- Size is computed by summing matching `.gguf` files in `~/Library/Caches/llama.cpp` (supports multi-shard models, excludes mmproj files)
- Bare repo entries (e.g. `org/model-GGUF`) are hidden when a quant-specific variant (e.g. `org/model-GGUF:Q4_K_M`) exists

### `llama download <user/model[:quant]> [--alias NAME]`
Download a model and register it in the preset file.
- Uses `llama-cli -hf <repo>` to download into `~/Library/Caches/llama.cpp`
  - **Note**: `llama-cli` enters interactive chat mode after loading. The download itself happens during startup, so we need to kill the process after the model file appears in cache (or find a better download mechanism).
- Auto-generates an alias from the model name if `--alias` not provided:
  - Strip org prefix, quant tag, and common suffixes (`-GGUF`, `-Instruct`, `-it`)
  - Lowercase the result
- Appends `[repo]` + `alias = ...` to the preset INI file
- Skips if model already in preset

### `llama remove <model_id_or_alias>`
Remove a model from the preset file and optionally delete the cached `.gguf` files.

### `llama info <model_id_or_alias_or_path>`
Show GGUF metadata for a model, including:
- File path, total size (summed across shards)
- General metadata: name, architecture, size_label, context_length
- Embedded sampling parameters (`general.sampling.*` keys) if present, otherwise show hardcoded defaults
- Whether the model is currently loaded (query server if running)

### `llama logs`
Tail the server log file (`~/.config/llama/server.log`).

## Config files

### `~/.config/llama/models.ini`
llama-server preset file. Example:
```ini
version = 1

[*]
c = 0

[mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M]
alias = ministral-3b

[unsloth/gpt-oss-20b-GGUF]
alias = gpt-oss-20b
```

Section names are HF repo IDs (optionally with `:quant` tag). The `[*]` section sets global defaults. Per-model keys correspond to llama-server CLI args without `--` prefix.

### `~/.config/llama/server.pid`
PID of the running server process.

### `~/.config/llama/server.log`
Server stdout/stderr log.

## Architecture notes

- **llama-server is installed via Homebrew** (`brew install llama.cpp`). The CLI tool wraps it, never replaces it.
- **Router mode**: llama-server runs with `--models-preset` which enables multi-model routing. Models load/unload on demand. `--sleep-idle-seconds 300` frees VRAM after 5 minutes idle.
- **Cache directory**: `~/Library/Caches/llama.cpp` (macOS default). Models are stored as `.gguf` files. Multi-shard models have files like `*-00001-of-00003.gguf`. Multimodal projectors are `*mmproj*.gguf`.
- **GGUF sampling metadata**: Since [llama.cpp PR #17120](https://github.com/ggml-org/llama.cpp/pull/17120), models can embed recommended sampling params (`general.sampling.temp`, `general.sampling.top_k`, etc.) in GGUF metadata. llama-server reads these automatically at inference time (priority: user request params > GGUF metadata > hardcoded defaults).
- **Server API endpoints used**:
  - `GET /models` — list all models with status, aliases, tags
  - `GET /props?model=<name>` — default generation settings, chat template, modalities
  - `GET /slots?model=<name>` — per-slot status
  - `GET /health` — server health check
- **GGUF parsing**: Pure Python, reads KV metadata from the GGUF header (no external dependencies needed). Used for `llama info` to show embedded sampling params and model metadata.

## Existing code reference

The following functions in `~/.zshrc` (lines 219-323) contain working implementations that should be ported:

- **`llama-download()`** (line 221): Downloads via `llama-cli -hf`, auto-generates alias, appends to INI
- **`llama()`** (line 257): Starts llama-server with `--jinja --sleep-idle-seconds 300 --models-preset`
- **`llama-models()`** (line 265): Queries `/models`, builds cache size lookup, filters duplicates, prints table
- **GGUF metadata reader**: Used during this session (not in zshrc) — pure Python struct-based parser that reads `general.sampling.*` keys from GGUF files

## Open questions / improvements

- `llama download` currently uses `llama-cli -hf <repo>` which enters interactive chat after download. Need a clean way to download-only (kill after download completes, or use HuggingFace Hub API directly).
- Consider adding `llama chat <model>` as a quick interactive chat shortcut.
- `/props` endpoint shows hardcoded sampling defaults, not GGUF-embedded ones. The GGUF params apply at inference time but aren't visible via the API — `llama info` reading the GGUF directly is the only way to see them.
- No VRAM usage info available via server API. Could potentially use `powermetrics` or similar macOS tools.
