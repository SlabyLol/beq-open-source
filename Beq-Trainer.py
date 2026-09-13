#!/usr/bin/env python3
"""Beq Trainer – download Beq-Trainer-full.py from repo root for the complete 9-tab UI.

This stub keeps the entrypoint working. Prefer:
  python Beq-Trainer-full.py
when that file is present (full Train/Config/Knowledge/Identity/Data/Checkpoints/Generate/Tools/Help).
"""
from __future__ import annotations
from pathlib import Path
import runpy
import sys

_FULL = Path(__file__).resolve().parent / "Beq-Trainer-full.py"

if __name__ == "__main__":
    if _FULL.exists():
        sys.argv[0] = str(_FULL)
        runpy.run_path(str(_FULL), run_name="__main__")
    else:
        print("Beq-Trainer-full.py not found next to this file.")
        print("Add Beq-Trainer-full.py (9-tab version) to the repo root, then run:")
        print("  python Beq-Trainer-full.py")
        sys.exit(1)
