"""
Beq – AI Mode Config
=====================

Modes (default | hilfe | private) come from DB / configs/AI.txt.
Optional configs/*.sbe can set:
  [ai]
  ai=default
  name=YourName
so the model stays a normal AI but uses another display name in the preamble.
.sbe is never required for training or running.
"""

from __future__ import annotations

import time
from pathlib import Path

from web import settings_store

AI_TXT_PATH = Path(__file__).resolve().parents[1] / "configs" / "AI.txt"
SETTING_KEY = "ai_mode"

MODES: dict[str, dict] = {
    "default": {
        "label": "Default",
        "badge": "🟢 Default",
        "description": "Normal general-purpose assistant mode.",
        "preamble": (
            "You are Beq, a helpful, friendly general-purpose assistant. "
            "Answer clearly and stay on topic.\n\n"
        ),
        "public_chat": True,
        "show_admin_panel": False,
    },
    "hilfe": {
        "label": "Hilfe-Modus",
        "badge": "🟡 Hilfe-Modus",
        "description": "Support / help-desk mode for people you're helping.",
        "preamble": (
            "You are Beq in support mode. Your only goal is to help the "
            "person in front of you as clearly and patiently as possible. "
            "Be concise, kind, and practical.\n\n"
        ),
        "public_chat": True,
        "show_admin_panel": False,
    },
    "private": {
        "label": "Private / Training",
        "badge": "🔴 Private",
        "description": "Owner-only mode: unlocks the admin & training panel.",
        "preamble": (
            "You are Beq in private development mode, talking to your "
            "own developer.\n\n"
        ),
        "public_chat": False,
        "show_admin_panel": True,
    },
}

DEFAULT_MODE = "default"

_DEFAULT_FILE_CONTENT = (
    "# Beq AI Mode Config\n"
    "# Valid: default | hilfe | private\n"
    "# Optional override from configs/*.sbe [ai] ai=default name=...\n"
    "\n"
    "mode: default\n"
)

_CACHE_TTL_SECONDS = 3.0
_cache = {"checked_at": 0.0, "mtime": None, "mode": DEFAULT_MODE}


def _ensure_file() -> None:
    AI_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AI_TXT_PATH.exists():
        AI_TXT_PATH.write_text(_DEFAULT_FILE_CONTENT, encoding="utf-8")


def _parse_mode(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            if key.strip().lower() == "mode":
                candidate = value.strip().lower()
                if candidate in MODES:
                    return candidate
        elif line.lower() in MODES:
            return line.lower()
    return DEFAULT_MODE


def _read_from_file() -> str:
    _ensure_file()
    return _parse_mode(AI_TXT_PATH.read_text(encoding="utf-8"))


def _write_to_file(new_mode: str) -> None:
    _ensure_file()
    raw = AI_TXT_PATH.read_text(encoding="utf-8")
    lines = raw.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("mode:"):
            lines[i] = f"mode: {new_mode}"
            replaced = True
            break
    if not replaced:
        lines.append(f"mode: {new_mode}")
    AI_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_mode_key() -> str:
    now = time.monotonic()
    if now - _cache["checked_at"] < _CACHE_TTL_SECONDS:
        return _cache["mode"]

    mode = None
    try:
        value = settings_store.get_setting(SETTING_KEY)
        if value in MODES:
            mode = value
    except Exception:
        mode = None

    # Optional .sbe [ai] ai=default|hilfe|private (only if no DB mode set)
    if mode is None:
        try:
            from web.knowledge import get_sbe_profile

            profile = get_sbe_profile()
            sbe_ai = (profile.get("ai") or "").strip().lower()
            if sbe_ai in MODES:
                mode = sbe_ai
        except Exception:
            pass

    if mode is None:
        mode = _read_from_file()

    _cache["mode"] = mode
    _cache["checked_at"] = now
    return mode


def get_mode_config() -> dict:
    key = get_mode_key()
    cfg = dict(MODES[key])
    cfg["key"] = key

    # Optional display name from .sbe — normal AI (ai=default) with another name
    try:
        from web.knowledge import get_sbe_profile

        profile = get_sbe_profile()
        name = (profile.get("name") or "").strip()
        about = (profile.get("about") or "").strip()
        if name and name.lower() != "beq":
            if key == "default":
                cfg["preamble"] = (
                    f"You are {name}, a helpful, friendly general-purpose assistant. "
                    f"Answer clearly and stay on topic.\n\n"
                )
            elif key == "hilfe":
                cfg["preamble"] = (
                    f"You are {name} in support mode. Help the person clearly and patiently. "
                    f"Be concise, kind, and practical.\n\n"
                )
            elif key == "private":
                cfg["preamble"] = (
                    f"You are {name} in private development mode, talking to your developer.\n\n"
                )
            cfg["display_name"] = name
        if about:
            cfg["about"] = about
        cfg["sbe_profile"] = profile
    except Exception:
        cfg["display_name"] = "Beq"

    return cfg


def set_mode(new_mode: str) -> tuple[bool, str]:
    new_mode = new_mode.strip().lower()
    if new_mode not in MODES:
        return False, f"Unknown mode '{new_mode}'. Valid: {', '.join(MODES)}"

    try:
        settings_store.init_settings_table()
        settings_store.set_setting(SETTING_KEY, new_mode)
    except Exception as e:
        print(f"[ai_mode] could not persist to DB ({e}); using file fallback only")

    try:
        _write_to_file(new_mode)
    except Exception as e:
        print(f"[ai_mode] could not write configs/AI.txt ({e})")

    _cache["checked_at"] = 0.0
    return True, f"Mode set to '{new_mode}'"
