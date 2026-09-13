"""
Beq Knowledge (.sbe) — OPTIONAL
===============================
.sbe files are NOT required to train or run the model.
Training uses only data/input.txt (+ optional input_extra.txt).

When .sbe files exist, they can:
  1) Store exact FAQ answers (Q: / A:)
  2) Set display name / about via [identity] or [ai]
  3) Select base persona: ai=default (normal AI, maybe other name),
     ai=hilfe, or ai=private

Example minimal "normal AI, other name":

  [ai]
  ai=default
  name=DarkFox
  role=a helpful general-purpose assistant
  about=I am DarkFox, a normal AI assistant with my own name.

No Q:/A: pairs required. Without any .sbe files, Beq still runs
with built-in default mode and the name "Beq".
"""

from __future__ import annotations

import re
import time
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_CACHE_TTL = 2.0
_cache: dict = {
    "loaded_at": 0.0,
    "entries": [],
    "profile": None,
    "mtime": 0.0,
}

_Q_LINE = re.compile(r"^Q:\s*(.+)$", re.I)
_A_LINE = re.compile(r"^A:\s*(.*)$", re.I)
_SECTION = re.compile(r"^\[(\w+)\]\s*$")
_KV = re.compile(r"^([^=#]+)=(.*)$")

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "for", "please", "can", "you", "me", "my", "your",
    "tell", "say", "just", "hey", "hi", "hello", "ok", "okay",
}

# Valid ai= values (same keys as web.ai_mode.MODES)
VALID_AI = {"default", "hilfe", "private"}


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9äöüáéíóúñß\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t and t not in _STOP}


def _split_aliases(q: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\|\s*", q) if p.strip()]
    return parts or [q]


def _default_profile() -> dict:
    return {
        "ai": "default",  # normal general-purpose AI
        "name": "Beq",
        "role": "a helpful general-purpose assistant",
        "about": "I am Beq, a helpful general-purpose assistant.",
        "has_sbe": False,
    }


def _parse_sbe(text: str) -> tuple[list[dict], dict]:
    """Return (faq_entries, profile_updates from [ai]/[identity])."""
    entries: list[dict] = []
    pending_qs: list[str] = []
    pending_a_lines: list[str] = []
    collecting_a = False
    section = "faq"
    identity: dict[str, str] = {}
    ai_block: dict[str, str] = {}

    def flush() -> None:
        nonlocal pending_qs, pending_a_lines, collecting_a
        if pending_qs and pending_a_lines:
            answer = "\n".join(pending_a_lines).strip()
            if answer:
                for q in pending_qs:
                    entries.append({
                        "q": q,
                        "q_norm": _norm(q),
                        "q_tokens": _tokens(q),
                        "a": answer,
                        "section": section,
                        "source": "explicit",
                    })
        pending_qs = []
        pending_a_lines = []
        collecting_a = False

    for raw in text.splitlines():
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            if collecting_a and not stripped and pending_a_lines:
                flush()
            continue

        m = _SECTION.match(stripped)
        if m:
            flush()
            section = m.group(1).lower()
            collecting_a = False
            continue

        if section in ("identity", "ai"):
            qm = _Q_LINE.match(stripped)
            am = _A_LINE.match(stripped)
            if not (qm or am):
                km = _KV.match(stripped)
                if km:
                    key = km.group(1).strip().lower()
                    val = km.group(2).strip()
                    if section == "ai":
                        ai_block[key] = val
                    else:
                        identity[key] = val
                continue
            section = "faq"

        qm = _Q_LINE.match(stripped)
        if qm:
            flush()
            pending_qs = _split_aliases(qm.group(1).strip())
            collecting_a = False
            continue

        am = _A_LINE.match(stripped)
        if am and pending_qs:
            collecting_a = True
            first = am.group(1)
            pending_a_lines = [first] if first.strip() else []
            continue

        if collecting_a and pending_qs:
            pending_a_lines.append(stripped)
            continue

    flush()

    # Merge profile: [ai] wins over [identity] for same keys
    profile_bits: dict[str, str] = {}
    profile_bits.update(identity)
    profile_bits.update(ai_block)

    name = profile_bits.get("name") or profile_bits.get("display_name")
    role = profile_bits.get("role")
    about = profile_bits.get("about")
    ai_mode = (profile_bits.get("ai") or profile_bits.get("mode") or "").strip().lower()

    if name or role or about or ai_mode:
        resolved_name = name or "Beq"
        resolved_role = role or "a helpful general-purpose assistant"
        resolved_about = about or f"I am {resolved_name}, {resolved_role}."
        if ai_mode not in VALID_AI:
            ai_mode = "default"

        auto = [
            ("Who are you?", resolved_about),
            ("What are you?", resolved_about),
            ("What is your name?", f"My name is {resolved_name}."),
            (f"Who is {resolved_name}?", resolved_about),
            (f"What is {resolved_name}?", resolved_about),
        ]
        if resolved_name.lower() != "beq":
            auto.extend([
                ("Who is Beq?", resolved_about),
                ("What is Beq?", resolved_about),
            ])
        else:
            auto.extend([
                ("Who is Beq?", resolved_about),
                ("What is Beq?", resolved_about),
            ])

        existing = {e["q_norm"] for e in entries}
        for q, a in auto:
            qn = _norm(q)
            if qn in existing:
                continue
            entries.append({
                "q": q,
                "q_norm": qn,
                "q_tokens": _tokens(q),
                "a": a,
                "section": "identity",
                "source": "identity",
            })

        profile_update = {
            "ai": ai_mode or "default",
            "name": resolved_name,
            "role": resolved_role,
            "about": resolved_about,
            "has_sbe": True,
        }
    else:
        profile_update = {}

    return entries, profile_update


def _sbe_files() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.sbe"))


def _load_all(force: bool = False) -> tuple[list[dict], dict]:
    now = time.monotonic()
    files = _sbe_files()
    mtime = max((f.stat().st_mtime for f in files), default=0.0)
    if (
        not force
        and _cache["profile"] is not None
        and now - _cache["loaded_at"] < _CACHE_TTL
        and _cache["mtime"] == mtime
    ):
        return _cache["entries"], _cache["profile"]

    entries: list[dict] = []
    profile = _default_profile()

    for path in files:
        try:
            file_entries, file_profile = _parse_sbe(path.read_text(encoding="utf-8"))
            entries.extend(file_entries)
            if file_profile:
                profile.update(file_profile)
                profile["has_sbe"] = True
        except Exception as e:
            print(f"[knowledge] failed to load {path.name}: {e}")

    entries.sort(key=lambda e: 0 if e.get("source") == "explicit" else 1)
    _cache["entries"] = entries
    _cache["profile"] = profile
    _cache["loaded_at"] = now
    _cache["mtime"] = mtime
    return entries, profile


def get_sbe_profile() -> dict:
    """
    Profile from optional .sbe files.

    Always returns a dict with keys: ai, name, role, about, has_sbe.
    If no .sbe files exist, has_sbe=False and defaults are used (normal AI named Beq).
    ai=default means normal general-purpose AI (optionally with another name).
    """
    _, profile = _load_all()
    return dict(profile)


def try_knowledge_answer(prompt: str) -> str | None:
    """Return EXACT A: text if a Q matches; None if no .sbe or no match (optional)."""
    if not prompt or not prompt.strip():
        return None

    raw = prompt.strip().split("\n")[0].strip()
    raw = re.sub(r"^(user|human|prompt)\s*:\s*", "", raw, flags=re.I)
    qn = _norm(raw)
    if not qn:
        return None

    entries, _ = _load_all()
    if not entries:
        return None

    for e in entries:
        if qn == e["q_norm"]:
            return e["a"]

    qn_stripped = re.sub(r"\b(please|thanks|thank you)\b", "", qn)
    qn_stripped = re.sub(r"\s+", " ", qn_stripped).strip()
    for e in entries:
        if qn_stripped and qn_stripped == e["q_norm"]:
            return e["a"]

    for e in entries:
        eq = e["q_norm"]
        if len(eq) < 4:
            continue
        if eq in qn:
            return e["a"]
        if len(qn) >= 3 and qn in eq and len(qn) / max(len(eq), 1) >= 0.5:
            return e["a"]

    q_tokens = _tokens(raw)
    if not q_tokens:
        return None

    best_a = None
    best_score = 0.0
    best_explicit = False
    for e in entries:
        t = e.get("q_tokens") or _tokens(e["q"])
        if not t:
            continue
        overlap = len(q_tokens & t)
        recall = overlap / len(t)
        precision = overlap / max(len(q_tokens), 1)
        score = 0.65 * recall + 0.35 * precision
        explicit = e.get("source") == "explicit"
        if recall < 0.85:
            continue
        if score > best_score or (score == best_score and explicit and not best_explicit):
            best_score = score
            best_a = e["a"]
            best_explicit = explicit

    if best_a is not None and best_score >= 0.80:
        return best_a
    return None


def list_knowledge() -> list[dict]:
    entries, _ = _load_all(force=True)
    return [
        {
            "q": e["q"],
            "a": e["a"],
            "section": e["section"],
            "source": e.get("source", "explicit"),
        }
        for e in entries
    ]
