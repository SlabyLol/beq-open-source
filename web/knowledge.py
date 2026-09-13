"""
Beq Knowledge (.sbe files)
=========================
Load FAQ / identity / facts from configs/*.sbe and answer exact or fuzzy matches
before the tiny model generates gibberish.

.sbe format (simple Beq entries):

  # comment
  Q: Who are you?
  A: I am Beq, your own open-source AI.

  Q: What is 2+2?
  A: 4

  [identity]
  name=Beq
  role=Open-source PyTorch language model

Matching is case-insensitive; punctuation is stripped for comparison.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_CACHE_TTL = 5.0
_cache: dict = {"loaded_at": 0.0, "entries": [], "mtime": 0.0}

_Q_LINE = re.compile(r"^Q:\s*(.+)$", re.I)
_A_LINE = re.compile(r"^A:\s*(.+)$", re.I)
_SECTION = re.compile(r"^\[(\w+)\]\s*$")
_KV = re.compile(r"^([^=#]+)=(.*)$")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9äöüáéíóúñß\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_sbe(text: str) -> list[dict]:
    entries: list[dict] = []
    pending_q: str | None = None
    section = "faq"
    identity: dict[str, str] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SECTION.match(line)
        if m:
            section = m.group(1).lower()
            continue
        if section == "identity":
            km = _KV.match(line)
            if km:
                identity[km.group(1).strip().lower()] = km.group(2).strip()
            continue
        qm = _Q_LINE.match(line)
        if qm:
            pending_q = qm.group(1).strip()
            continue
        am = _A_LINE.match(line)
        if am and pending_q:
            entries.append({
                "q": pending_q,
                "q_norm": _norm(pending_q),
                "a": am.group(1).strip(),
                "section": section,
            })
            pending_q = None
            continue
        # multi-line answers: continue previous A if indented or plain
        if pending_q is None and entries and not line.startswith("Q:"):
            # continuation of last answer
            if line.startswith("A:"):
                continue

    # Auto identity FAQs from [identity] block
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
        ]
        for q, a in auto:
            entries.append({"q": q, "q_norm": _norm(q), "a": a, "section": "identity"})

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
            text = path.read_text(encoding="utf-8")
            entries.extend(_parse_sbe(text))
        except Exception as e:
            print(f"[knowledge] failed to load {path.name}: {e}")

    _cache["entries"] = entries
    _cache["loaded_at"] = now
    _cache["mtime"] = mtime
    return entries


def try_knowledge_answer(prompt: str) -> str | None:
    """Return FAQ/identity answer or None."""
    if not prompt or not prompt.strip():
        return None
    qn = _norm(prompt)
    if not qn:
        return None

    entries = _load_entries()
    # exact
    for e in entries:
        if qn == e["q_norm"]:
            return e["a"]
    # contains / soft match
    for e in entries:
        if len(e["q_norm"]) >= 6 and (e["q_norm"] in qn or qn in e["q_norm"]):
            return e["a"]
    # token overlap
    q_tokens = set(qn.split())
    best = None
    best_score = 0.0
    for e in entries:
        t = set(e["q_norm"].split())
        if not t:
            continue
        score = len(q_tokens & t) / max(len(t), 1)
        if score > best_score and score >= 0.7:
            best_score = score
            best = e["a"]
    return best


def list_knowledge() -> list[dict]:
    return [
        {"q": e["q"], "a": e["a"], "section": e["section"]}
        for e in _load_entries(force=True)
    ]
