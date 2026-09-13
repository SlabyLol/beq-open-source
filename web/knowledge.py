"""
Beq Knowledge (.sbe) — STRICT compliance
=======================================
Answers come ONLY from configs/*.sbe. When a question matches, the A: text
is returned EXACTLY as written (no model, no paraphrase).

Format:

  # comment
  Q: Who are you?
  A: I am Beq.

  # aliases (same answer for several phrasings)
  Q: What is Beq? | What's Beq? | Explain Beq
  A: Beq is an open-source pure-PyTorch language model.

  # multi-line answers (until next Q: or blank section)
  Q: How do I train?
  A: 1) Put text in data/input.txt
     2) Run train/train.py or Beq-Trainer
     3) Deploy the checkpoint

  [identity]
  name=Beq
  role=...
  about=...
"""

from __future__ import annotations

import re
import time
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_CACHE_TTL = 2.0
_cache: dict = {"loaded_at": 0.0, "entries": [], "mtime": 0.0}

_Q_LINE = re.compile(r"^Q:\s*(.+)$", re.I)
_A_LINE = re.compile(r"^A:\s*(.*)$", re.I)
_SECTION = re.compile(r"^\[(\w+)\]\s*$")
_KV = re.compile(r"^([^=#]+)=(.*)$")

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and",
    "or", "in", "on", "for", "please", "can", "you", "me", "my", "your",
    "tell", "say", "just", "hey", "hi", "hello", "ok", "okay",
}


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


def _parse_sbe(text: str) -> list[dict]:
    entries: list[dict] = []
    pending_qs: list[str] = []
    pending_a_lines: list[str] = []
    collecting_a = False
    section = "faq"
    identity: dict[str, str] = {}

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
        line = raw.rstrip()
        stripped = line.strip()

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

        if section == "identity":
            qm = _Q_LINE.match(stripped)
            am = _A_LINE.match(stripped)
            if not (qm or am):
                km = _KV.match(stripped)
                if km:
                    identity[km.group(1).strip().lower()] = km.group(2).strip()
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

    if identity:
        name = identity.get("name", "Beq")
        role = identity.get("role", "an open-source language model")
        about = identity.get("about", f"I am {name}, {role}.")
        auto = [
            ("Who are you?", about),
            ("What are you?", about),
            ("What is your name?", f"My name is {name}."),
            ("Who is Beq?", about),
            ("What is Beq?", about),
            (f"Who is {name}?", about),
            (f"What is {name}?", about),
        ]
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

    return entries


def _sbe_files() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(CONFIGS_DIR.glob("*.sbe"))


def _load_entries(force: bool = False) -> list[dict]:
    now = time.monotonic()
    files = _sbe_files()
    mtime = max((f.stat().st_mtime for f in files), default=0.0)
    if (
        not force
        and _cache["entries"] is not None
        and now - _cache["loaded_at"] < _CACHE_TTL
        and _cache["mtime"] == mtime
    ):
        return _cache["entries"]

    entries: list[dict] = []
    for path in files:
        try:
            entries.extend(_parse_sbe(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[knowledge] failed to load {path.name}: {e}")

    entries.sort(key=lambda e: 0 if e.get("source") == "explicit" else 1)
    _cache["entries"] = entries
    _cache["loaded_at"] = now
    _cache["mtime"] = mtime
    return entries


def try_knowledge_answer(prompt: str) -> str | None:
    """Return the EXACT A: text from .sbe if the prompt matches a Q."""
    if not prompt or not prompt.strip():
        return None

    raw = prompt.strip().split("\n")[0].strip()
    raw = re.sub(r"^(user|human|prompt)\s*:\s*", "", raw, flags=re.I)
    qn = _norm(raw)
    if not qn:
        return None

    entries = _load_entries()
    if not entries:
        return None

    # 1) Exact match → verbatim answer
    for e in entries:
        if qn == e["q_norm"]:
            return e["a"]

    # 2) Ignore trailing please/thanks
    qn_stripped = re.sub(r"\b(please|thanks|thank you)\b", "", qn)
    qn_stripped = re.sub(r"\s+", " ", qn_stripped).strip()
    for e in entries:
        if qn_stripped and qn_stripped == e["q_norm"]:
            return e["a"]

    # 3) Full FAQ question contained in prompt
    for e in entries:
        eq = e["q_norm"]
        if len(eq) < 4:
            continue
        if eq in qn:
            return e["a"]
        if len(qn) >= 3 and qn in eq and len(qn) / max(len(eq), 1) >= 0.5:
            return e["a"]

    # 4) High token recall only (strict)
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
    return [
        {
            "q": e["q"],
            "a": e["a"],
            "section": e["section"],
            "source": e.get("source", "explicit"),
        }
        for e in _load_entries(force=True)
    ]
