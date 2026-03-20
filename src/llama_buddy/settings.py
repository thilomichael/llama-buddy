"""Settings management and interactive settings editor."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

from rich.live import Live
from rich.text import Text

from llama_buddy.config import CONFIG_DIR, ensure_config_dir, read_preset, write_preset
from llama_buddy.console import console
from llama_buddy.tui import read_key, require_tty

SETTINGS_FILE = CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    port: int = 8080
    idle_timeout: int = 300
    jinja: bool = True
    flash_attention: str = "auto"
    ctx_size: int = 0
    gpu_layers: str = "auto"

    def to_server_args(self) -> list[str]:
        """Convert settings to llama-server CLI arguments."""
        args: list[str] = []
        args.extend(["--port", str(self.port)])
        args.extend(["--sleep-idle-seconds", str(self.idle_timeout)])
        if self.jinja:
            args.append("--jinja")
        else:
            args.append("--no-jinja")
        args.extend(["--flash-attn", self.flash_attention])
        if self.ctx_size > 0:
            args.extend(["--ctx-size", str(self.ctx_size)])
        args.extend(["--gpu-layers", self.gpu_layers])
        return args


# Display metadata for each global setting
SETTING_META: dict[str, dict] = {
    "port": {
        "label": "Port",
        "description": "Server listen port",
        "type": "int",
    },
    "idle_timeout": {
        "label": "Idle Timeout",
        "description": "Seconds before unloading idle models (0 = never)",
        "type": "int",
    },
    "jinja": {
        "label": "Jinja Templates",
        "description": "Use Jinja template engine for chat",
        "type": "bool",
    },
    "flash_attention": {
        "label": "Flash Attention",
        "description": "Flash Attention mode",
        "type": "choice",
        "choices": ["auto", "on", "off"],
    },
    "ctx_size": {
        "label": "Context Size",
        "description": "Prompt context size (0 = use model default)",
        "type": "int",
    },
    "gpu_layers": {
        "label": "GPU Layers",
        "description": "Layers to offload to GPU",
        "type": "choice",
        "choices": ["auto", "all", "none"],
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
        "description": "Override GPU layers",
        "type": "choice",
        "choices": ["auto", "all", "none"],
        "default": "auto",
    },
    "fa": {
        "label": "Flash Attention",
        "description": "Override Flash Attention mode",
        "type": "choice",
        "choices": ["auto", "on", "off"],
        "default": "auto",
    },
}


def load_settings() -> Settings:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            known = {f.name for f in fields(Settings)}
            return Settings(**{k: v for k, v in data.items() if k in known})
        except (json.JSONDecodeError, TypeError):
            pass
    return Settings()


def save_settings(settings: Settings) -> None:
    ensure_config_dir()
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2) + "\n")


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


def edit_settings() -> None:
    """Interactive settings editor with model settings sub-menu."""
    settings = load_settings()

    def get_value(key: str) -> str:
        return str(getattr(settings, key))

    def set_value(key: str, value: str) -> None:
        meta = SETTING_META[key]
        if meta["type"] == "int":
            setattr(settings, key, int(value))
        elif meta["type"] == "bool":
            setattr(settings, key, value in ("True", "true", "1", "on"))
        else:
            setattr(settings, key, value)

    def on_save() -> None:
        save_settings(settings)
        console.print("Settings saved.", style="green")

    def on_extra(idx: int) -> None:
        if idx == 0:
            _edit_model_settings()

    _run_kv_editor(
        title="Settings",
        meta_map=SETTING_META,
        get_value=get_value,
        set_value=set_value,
        on_save=on_save,
        extra_items=["Model Settings >"],
        on_extra=on_extra,
    )


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
    meta = dict(MODEL_SETTING_META)
    # Add any existing keys not in the standard set
    if preset.has_section(model_id):
        for key in preset.options(model_id):
            if key not in meta and key != "version":
                meta[key] = {
                    "label": key,
                    "description": "Custom setting",
                    "type": "text",
                    "default": "",
                }
    return meta


def _edit_single_model(model_id: str) -> None:
    """Edit settings for a single model in the preset INI file."""
    from llama_buddy.models import get_model_name

    preset = read_preset()
    display_name = get_model_name(model_id) or model_id
    meta_map = _build_model_meta(model_id, preset)

    def get_value(key: str) -> str:
        default = meta_map[key].get("default", "")
        return preset.get(model_id, key, fallback=default)

    def set_value(key: str, value: str) -> None:
        default = meta_map[key].get("default", "")
        if value == default:
            preset.remove_option(model_id, key)
        else:
            preset.set(model_id, key, value)

    def on_save() -> None:
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
