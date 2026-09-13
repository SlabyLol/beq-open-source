"""
Beq – AI Mode Config
=====================

Beq's behavior is switched between three modes, either by editing
`configs/AI.txt` (local dev) or, better, through the HTTP config API
(`/api/config`) or the admin panel (`/admin`) — those write to the database
via web/settings_store.py, so the mode survives Render redeploys.

Just change the mode, save/submit it, done — no restart needed. Beq re-reads
the current value automatically (cached for a few seconds).

Modes
-----
default  Normal general-purpose assistant persona. This is what a fresh
         install ships with.
hilfe    "Hilfe-Modus" — a support/help-desk persona meant for helping
         other people who use the chat. Friendlier, more patient framing.
private  Your own dev/training mode. Chat access is restricted to the admin
         account, and the admin panel (mode switch + training controls)
         becomes visible.

Important honesty note (please read)
-------------------------------------
Beq is a small, from-scratch, *character-level* language model (see
model/transformer.py) — it is NOT an instruction-following model like
ChatGPT/Claude. These "modes" change:
  1) what text is silently prepended to the prompt before generation
     (a soft steering hint), and
  2) how the *web app itself* behaves (who can chat, whether the admin
     panel is shown, etc).
They do NOT magically make the model understand instructions it was never
trained on. If you want the model to actually behave differently per mode
(e.g. answer in a support-agent tone), the most reliable way is to include
examples of that style in `data/input.txt` and retrain.
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
    "#\n"
    "# Valid values for `mode`: default | hilfe | private\n"
    "#   default -> normal assistant, open to everyone\n"
    "#   hilfe   -> support/help persona, open to everyone\n"
    "#   private -> owner-only, unlocks the admin/training panel\n"
    "#\n"
    "# NOTE: on Render (and most free hosts) this file resets on every\n"
    "# redeploy. The real, persistent source of truth is the database\n"
    "# (see web/settings_store.py) once DATABASE_URL is set. This file is\n"
    "# still read as a fallback for local dev without a database.\n"
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
    """
    Active mode key.

    Priority: database setting (persists across Render redeploys) -> local
    configs/AI.txt (fallback for local dev / no DB configured).
    Cached for a few seconds so we don't hit the DB on every keystroke.
    """
    now = time.monotonic()
    if now - _cache["checked_at"] < _CACHE_TTL_SECONDS:
        return _cache["mode"]

    mode = None
    try:
        value = settings_store.get_setting(SETTING_KEY)
        if value in MODES:
            mode = value
    except Exception:
        mode = None  # DB not reachable/initialized yet -> fall back to file

    if mode is None:
        mode = _read_from_file()

    _cache["mode"] = mode
    _cache["checked_at"] = now
    return mode


def get_mode_config() -> dict:
    key = get_mode_key()
    cfg = dict(MODES[key])
    cfg["key"] = key
    return cfg


def set_mode(new_mode: str) -> tuple[bool, str]:
    new_mode = new_mode.strip().lower()
    if new_mode not in MODES:
        return False, f"Unknown mode '{new_mode}'. Valid: {', '.join(MODES)}"

    # Persistent copy (survives Render redeploys)
    try:
        settings_store.init_settings_table()
        settings_store.set_setting(SETTING_KEY, new_mode)
    except Exception as e:
        # Still fine locally without a DB - file fallback below covers it.
        print(f"[ai_mode] could not persist to DB ({e}); using file fallback only")

    # Human-readable mirror on disk (nice for local dev)
    try:
        _write_to_file(new_mode)
    except Exception as e:
        print(f"[ai_mode] could not write configs/AI.txt ({e})")

    _cache["checked_at"] = 0.0  # force reload on next get_mode_key()
    return True, f"Mode set to '{new_mode}'"
