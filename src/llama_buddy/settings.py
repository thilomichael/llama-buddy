"""Settings editor for the preset's ``[*]`` section and per-model sections.

Every setting is a llama-server flag written into the preset file, so the
router applies it and nothing has to be passed on the command line, where
it would override per-model values. Only the port lives elsewhere: the
router owns it, so it goes in llama.cpp's own config.ini.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from rich.live import Live
from rich.text import Text

from llama_buddy.capabilities import filter_supported
from llama_buddy.config import (
    CONFIG_DIR,
    DEFAULT_PORT,
    get_port,
    key_names,
    read_preset,
    set_port,
    write_preset,
)
from llama_buddy.console import console
from llama_buddy.ini import IniFile
from llama_buddy.tui import read_key, require_tty

# Where releases before 0.1.16 kept the global settings.
SETTINGS_FILE = CONFIG_DIR / "settings.json"

PORT_META = {
    "label": "Port",
    "description": "Server listen port (stored in llama.cpp's config.ini)",
    "type": "int",
    "default": str(DEFAULT_PORT),
}

# Settings for every model, kept in the preset's [*] section. The defaults
# are llama-server's own; a setting left at its default is not written.
GLOBAL_SETTING_META: dict[str, dict] = {
    "sleep-idle-seconds": {
        "label": "Idle Timeout",
        "description": "Seconds before an idle model sleeps (-1 = never)",
        "type": "int",
        "default": "-1",
    },
    "c": {
        "label": "Context Size",
        "description": "Prompt context size (0 = from model, fit to memory)",
        "type": "int",
        "default": "0",
    },
    "jinja": {
        "label": "Jinja Templates",
        "description": "Use Jinja template engine for chat",
        "type": "bool",
        "default": "true",
    },
    "fa": {
        "label": "Flash Attention",
        "description": "Flash Attention mode",
        "type": "choice",
        "choices": ["auto", "on", "off"],
        "default": "auto",
    },
    "ngl": {
        "label": "GPU Layers",
        "description": "Layers to offload to GPU (0 = CPU only)",
        "type": "choice",
        "choices": ["auto", "all", "0"],
        "default": "auto",
    },
    "load-mode": {
        "label": "Load Mode",
        "description": "How model weights are read (replaces mmap/mlock)",
        "type": "choice",
        "choices": ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"],
        "default": "auto",
        "flag": "--load-mode",
    },
    "lazy-mode": {
        "label": "Lazy Mode",
        "description": "Read large tensors on demand instead of resident",
        "type": "choice",
        "choices": ["auto", "on", "off"],
        "default": "auto",
        "flag": "--lazy-mode",
    },
    "kv-unified-per-slot": {
        "label": "KV Per Slot",
        "description": "Context limit per parallel slot (0 = unset)",
        "type": "int",
        "default": "0",
        "flag": "--kv-unified-per-slot",
    },
}

# Per-model INI keys and their display metadata
MODEL_SETTING_META: dict[str, dict] = {
    "alias": {
        "label": "Alias",
        "description": "Custom name for this model",
        "type": "text",
        "default": "",
    },
    "c": {
        "label": "Context Size",
        "description": "Override context size (0 = use global/model default)",
        "type": "int",
        "default": "0",
    },
    "ngl": {
        "label": "GPU Layers",
        "description": "Override GPU layers (0 = CPU only)",
        "type": "choice",
        "choices": ["auto", "all", "0"],
        "default": "auto",
    },
    "fa": {
        "label": "Flash Attention",
        "description": "Override Flash Attention mode",
        "type": "choice",
        "choices": ["auto", "on", "off"],
        "default": "auto",
    },
    "spec-type": {
        "label": "Speculative Decoding",
        "description": "draft-mtp uses the model's own multi-token head",
        "type": "choice",
        "choices": ["", "draft-mtp", "none"],
        "default": "",
        "flag": "--spec-type",
    },
    "spec-draft-n-max": {
        "label": "Draft Tokens",
        "description": "Tokens to draft per step when speculating",
        "type": "int",
        "default": "3",
        "flag": "--spec-draft-n-max",
    },
    "ncmoe": {
        "label": "CPU MoE Layers",
        "description": "Keep MoE expert weights of the first N layers on CPU",
        "type": "int",
        "default": "0",
        "flag": "--n-cpu-moe",
    },
    "ncffn": {
        "label": "CPU FFN Layers",
        "description": "Keep dense FFN weights of the first N layers on CPU",
        "type": "int",
        "default": "0",
        "flag": "--n-cpu-ffn",
    },
}


# settings.json field → [*] key, and the value llb used when the field was
# missing. A value equal to llama-server's own default is not written.
_SETTINGS_JSON_KEYS: dict[str, tuple[str, object]] = {
    "idle_timeout": ("sleep-idle-seconds", 300),
    "ctx_size": ("c", 8192),
    "jinja": ("jinja", True),
    "flash_attention": ("fa", "auto"),
    "gpu_layers": ("ngl", "auto"),
    "load_mode": ("load-mode", "auto"),
    "lazy_mode": ("lazy-mode", "auto"),
    "kv_unified_per_slot": ("kv-unified-per-slot", 0),
}


def migrate_settings_file(preset: IniFile) -> bool:
    """Move settings.json into the preset's ``[*]`` and config.ini, once.

    llb used to pass these as router flags, which override every preset
    value, so they win over whatever ``[*]`` said before. Returns True when
    the preset was changed.
    """
    if not SETTINGS_FILE.exists():
        return False
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    changed = False
    for field, (key, old_default) in _SETTINGS_JSON_KEYS.items():
        value = data.get(field, old_default)
        if isinstance(value, bool):
            value = "true" if value else "false"
        value = str(value)
        # 0 used to mean "never sleep", which llama-server rejects.
        if key == "sleep-idle-seconds" and value == "0":
            value = "-1"
        # llb offered "none", which llama-server cannot parse.
        if key == "ngl" and value == "none":
            value = "0"
        names = key_names(key)
        if value == GLOBAL_SETTING_META[key]["default"]:
            changed = preset.remove_options("*", names) or changed
        elif preset.get_any("*", names) != value:
            preset.remove_options("*", names)
            preset.set("*", key, value)
            changed = True

    try:
        port = int(data.get("port", DEFAULT_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    if port != get_port():
        set_port(port)

    SETTINGS_FILE.unlink(missing_ok=True)
    console.print(
        "Moved settings from settings.json into the preset's [*] section.",
        style="dim",
    )
    return changed


def _set_option(
    preset: IniFile, section: str, key: str, value: str, default: str,
) -> None:
    """Write a setting under llb's spelling, or drop it when it is the default."""
    preset.remove_options(section, key_names(key))
    if value != default:
        preset.set(section, key, value)


# ---------------------------------------------------------------------------
# Generic key-value editor used by both global and model settings
# ---------------------------------------------------------------------------


def _render_kv_menu(
    title: str,
    items: list[tuple[str, dict, str]],
    selected: int,
    editing: bool,
    edit_buffer: str,
    extra_items: list[str] | None = None,
) -> Text:
    """Render a key-value settings menu.

    items: list of (key, meta_dict, current_value_str)
    extra_items: additional menu entries after the settings (e.g. "Model Settings >")
    """
    text = Text()
    text.append(f"  {title}\n\n", style="bold")

    for i, (key, meta, value) in enumerate(items):
        is_selected = i == selected
        if is_selected:
            text.append("  > ", style="bold cyan")
        else:
            text.append("    ")

        label_style = "bold cyan" if is_selected else ""
        text.append(f"{meta['label']}: ", style=label_style)

        if is_selected and editing:
            text.append(edit_buffer, style="bold underline")
            text.append("_", style="bold blink")
        elif meta["type"] == "bool":
            if value in ("True", "true", "1", "on"):
                text.append("on", style="green")
            else:
                text.append("off", style="red")
        else:
            val_style = "bold cyan" if is_selected else "dim"
            text.append(value, style=val_style)

        text.append(f"  {meta['description']}", style="dim italic")
        text.append("\n")

    if extra_items:
        for j, label in enumerate(extra_items):
            idx = len(items) + j
            is_selected = idx == selected
            if is_selected:
                text.append("  > ", style="bold cyan")
                text.append(f"{label}\n", style="bold cyan")
            else:
                text.append(f"    {label}\n", style="dim")

    text.append("\n")
    if editing:
        item_meta = items[selected][1]
        if item_meta["type"] == "choice":
            choices = ", ".join(item_meta["choices"])
            text.append(
                f"  Type a value ({choices}) then Enter  Esc cancel",
                style="dim italic",
            )
        else:
            text.append(
                "  Type a value then Enter  Esc cancel",
                style="dim italic",
            )
    else:
        text.append(
            "  ↑/↓ navigate  Enter edit  q save & quit",
            style="dim italic",
        )
    return text


def _run_kv_editor(
    title: str,
    meta_map: dict[str, dict],
    get_value: Callable[[str], str],
    set_value: Callable[[str, str], None],
    on_save: Callable[[], None],
    extra_items: list[str] | None = None,
    on_extra: Callable[[int], None] | None = None,
) -> None:
    """Generic interactive key-value editor loop.

    get_value(key) -> str: get the current value for a key
    set_value(key, value_str) -> None: set a value
    on_save() -> None: called when user quits
    extra_items: labels for extra menu entries after settings
    on_extra(index) -> None: called when an extra item is selected
    """
    require_tty()
    selected = 0
    editing = False
    edit_buffer = ""

    def build_items():
        keys = list(meta_map.keys())
        return keys, [(k, meta_map[k], get_value(k)) for k in keys]

    while True:
        pending_extra: int | None = None
        keys, items = build_items()
        total = len(keys) + (len(extra_items) if extra_items else 0)
        selected = min(selected, max(total - 1, 0))

        with Live(
            _render_kv_menu(
                title, items, selected, editing, edit_buffer, extra_items
            ),
            console=console,
            auto_refresh=False,
            transient=True,
        ) as live:
            while True:
                key = read_key()

                if editing:
                    setting_key = keys[selected]
                    meta = meta_map[setting_key]

                    if key == "enter":
                        if meta["type"] == "int":
                            try:
                                set_value(setting_key, str(int(edit_buffer)))
                            except ValueError:
                                pass
                        elif meta["type"] == "choice":
                            if edit_buffer in meta["choices"]:
                                set_value(setting_key, edit_buffer)
                        elif meta["type"] == "text":
                            set_value(setting_key, edit_buffer)
                        editing = False
                        edit_buffer = ""
                    elif key in ("\x1b", "ctrl-c"):
                        editing = False
                        edit_buffer = ""
                    elif key == "backspace":
                        edit_buffer = edit_buffer[:-1]
                    elif len(key) == 1 and key.isprintable():
                        edit_buffer += key
                else:
                    if key == "up" and selected > 0:
                        selected -= 1
                    elif key == "down" and selected < total - 1:
                        selected += 1
                    elif key in ("enter", "right"):
                        if selected < len(keys):
                            setting_key = keys[selected]
                            meta = meta_map[setting_key]
                            if meta["type"] == "bool":
                                current = get_value(setting_key)
                                new_val = (
                                    "false"
                                    if current in ("True", "true", "1", "on")
                                    else "true"
                                )
                                set_value(setting_key, new_val)
                            else:
                                editing = True
                                edit_buffer = get_value(setting_key)
                        elif extra_items and on_extra:
                            pending_extra = selected - len(keys)
                            break  # Exit Live context before running sub-menu
                    elif key == "left":
                        continue
                    elif key in ("q", "ctrl-c"):
                        on_save()
                        return

                keys, items = build_items()
                total = len(keys) + (len(extra_items) if extra_items else 0)
                live.update(
                    _render_kv_menu(
                        title,
                        items,
                        selected,
                        editing,
                        edit_buffer,
                        extra_items,
                    ),
                    refresh=True,
                )

        # Live context is fully closed here — safe to run sub-menus
        if pending_extra is not None:
            on_extra(pending_extra)
        else:
            return


# ---------------------------------------------------------------------------
# Global settings editor
# ---------------------------------------------------------------------------


def edit_settings() -> bool:
    """Interactive settings editor with model settings sub-menu.

    Returns True when the port changed, which takes a restart rather than a
    reload of the model list.
    """
    preset = read_preset()
    port = get_port()
    meta_map = {"port": PORT_META, **filter_supported(GLOBAL_SETTING_META)}
    # The model sub-menu writes the preset too, so global edits are kept
    # apart and applied to a fresh read on save.
    pending: dict[str, str] = {}

    def get_value(key: str) -> str:
        if key in pending:
            return pending[key]
        if key == "port":
            return str(port)
        default = meta_map[key]["default"]
        return preset.get_any("*", key_names(key), default) or default

    def set_value(key: str, value: str) -> None:
        # llama-server rejects 0 here; -1 is "never".
        if key == "sleep-idle-seconds" and value == "0":
            value = "-1"
        pending[key] = value

    def on_save() -> None:
        current = read_preset()
        for key, value in pending.items():
            if key != "port":
                _set_option(current, "*", key, value, meta_map[key]["default"])
        write_preset(current)
        if "port" in pending:
            set_port(int(pending["port"]))
        console.print("Settings saved.", style="green")

    def on_extra(idx: int) -> None:
        if idx == 0:
            _edit_model_settings()

    _run_kv_editor(
        title="Settings",
        meta_map=meta_map,
        get_value=get_value,
        set_value=set_value,
        on_save=on_save,
        extra_items=["Model Settings >"],
        on_extra=on_extra,
    )
    return "port" in pending and int(pending["port"]) != port


# ---------------------------------------------------------------------------
# Per-model settings editor
# ---------------------------------------------------------------------------


def _edit_model_settings() -> None:
    """Select a model, then edit its per-model settings in the INI file."""
    from llama_buddy.select import select_model

    model_id = select_model(title="Select a model to configure")
    _edit_single_model(model_id)


def _build_model_meta(model_id: str, preset) -> dict[str, dict]:
    """Build meta map for a model, including any custom keys from the INI."""
    meta = filter_supported(MODEL_SETTING_META)
    known = {name for key in MODEL_SETTING_META for name in key_names(key)}
    # Add any existing keys not in the standard set
    if preset.has_section(model_id):
        for key in preset.options(model_id):
            if key not in known:
                meta[key] = {
                    "label": key,
                    "description": "Custom setting",
                    "type": "text",
                    "default": "",
                }
    return meta


def _sync_draft_model_path(model_id: str, preset) -> None:
    """Keep ``spec-draft-model`` in step with the MTP setting.

    A model whose head sits in a sidecar GGUF needs the path to it. The path
    is only meaningful while MTP is on: left behind, it would make
    llama-server load a draft model that nothing drafts with.

    Adding the path may probe the pair, which costs a model load, so this
    runs on save rather than inside the editor's Live.
    """
    from llama_buddy.config import find_model_mtp_file
    from llama_buddy.models import draft_model_loads

    path_names = key_names("spec-draft-model")
    if preset.get_any(model_id, key_names("spec-type")) != "draft-mtp":
        preset.remove_options(model_id, path_names)
        return
    if preset.get_any(model_id, path_names) is not None:
        return
    draft = find_model_mtp_file(model_id)
    if draft is None:
        return
    with console.status(
        "Checking the head loads with this llama.cpp build…", spinner="dots",
    ):
        ok = draft_model_loads(model_id, draft)
    if ok:
        preset.set(model_id, "spec-draft-model", str(draft))
    else:
        console.print(
            "The installed llama-server cannot load this model's MTP head, "
            "so it is not wired in.",
            style="yellow",
        )


def _edit_single_model(model_id: str) -> None:
    """Edit settings for a single model in the preset INI file."""
    from llama_buddy.models import get_model_name

    preset = read_preset()
    display_name = get_model_name(model_id) or model_id
    meta_map = _build_model_meta(model_id, preset)

    def get_value(key: str) -> str:
        default = meta_map[key].get("default", "")
        return preset.get_any(model_id, key_names(key), default) or default

    def set_value(key: str, value: str) -> None:
        default = meta_map[key].get("default", "")
        _set_option(preset, model_id, key, value, default)

    def on_save() -> None:
        _sync_draft_model_path(model_id, preset)
        write_preset(preset)
        console.print(
            f"Model settings for [bold]{display_name}[/bold] saved.",
            style="green",
        )

    def on_extra(idx: int) -> None:
        if idx == 0:
            _add_custom_setting(model_id, preset, meta_map)

    _run_kv_editor(
        title=f"Model Settings: {display_name}",
        meta_map=meta_map,
        get_value=get_value,
        set_value=set_value,
        on_save=on_save,
        extra_items=["Add custom setting >"],
        on_extra=on_extra,
    )


def _add_custom_setting(
    model_id: str, preset, meta_map: dict[str, dict]
) -> None:
    """Prompt for a custom key=value pair and add it to the model."""
    require_tty()

    # Prompt for key
    key_buffer = ""
    value_buffer = ""
    phase = "key"  # "key" -> "value"

    def render() -> Text:
        text = Text()
        text.append("  Add Custom Setting\n\n", style="bold")
        text.append("  Key: ", style="bold cyan" if phase == "key" else "")
        if phase == "key":
            text.append(key_buffer, style="bold underline")
            text.append("_", style="bold blink")
        else:
            text.append(key_buffer)
        text.append("\n")
        text.append("  Value: ", style="bold cyan" if phase == "value" else "dim")
        if phase == "value":
            text.append(value_buffer, style="bold underline")
            text.append("_", style="bold blink")
        text.append("\n\n")
        text.append("  Enter confirm  Esc cancel", style="dim italic")
        return text

    with Live(render(), console=console, auto_refresh=False, transient=True) as live:
        while True:
            key = read_key()
            if key in ("\x1b", "ctrl-c"):
                return
            elif key == "enter":
                if phase == "key":
                    if key_buffer.strip():
                        phase = "value"
                else:
                    k = key_buffer.strip()
                    if k and k not in meta_map:
                        preset.set(model_id, k, value_buffer)
                        meta_map[k] = {
                            "label": k,
                            "description": "Custom setting",
                            "type": "text",
                            "default": "",
                        }
                    return
            elif key == "backspace":
                if phase == "key":
                    key_buffer = key_buffer[:-1]
                else:
                    value_buffer = value_buffer[:-1]
            elif len(key) == 1 and key.isprintable():
                if phase == "key":
                    key_buffer += key
                else:
                    value_buffer += key
            else:
                continue
            live.update(render(), refresh=True)
