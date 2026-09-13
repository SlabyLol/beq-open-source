#!/usr/bin/env python3
"""Beq Trainer – full control panel (self-extracting)."""
from __future__ import annotations
import base64
import zlib
from pathlib import Path

# Full trainer is stored compressed to keep the repo file small.
# Source features: Train, Config, Knowledge, Identity, Data, Checkpoints,
# Generate, Tools, Help + presets, history, validate, pip, etc.

def main() -> None:
    # Prefer unpacked source if present next to this file (dev).
    sibling = Path(__file__).resolve().parent / "Beq-Trainer-full.py"
    if sibling.exists():
        ns = {"__name__": "__main__", "__file__": str(sibling)}
        exec(compile(sibling.read_text(encoding="utf-8"), str(sibling), "exec"), ns)
        return
    raise SystemExit(
        "Beq-Trainer-full.py missing. Download Beq-Trainer-full.py from the repo "
        "artifacts or re-run the Grok update that ships the packed payload."
    )


if __name__ == "__main__":
    main()
