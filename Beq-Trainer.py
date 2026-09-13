#!/usr/bin/env python3
"""Beq Trainer launcher.

The COMPLETE 9-tab trainer is the file:
  Beq-Trainer-full.py

Upload that file to the repo root (GitHub website → Add file → Upload),
then run:

  python Beq-Trainer.py
  # or
  python Beq-Trainer-full.py

Tabs in the full version:
  Train | Config & name | Knowledge | Identity | Data | Checkpoints | Generate | Tools | Help
"""
from __future__ import annotations
from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parent
_CANDIDATES = [
    _ROOT / "Beq-Trainer-full.py",
    _ROOT / "Beq-Trainer-complete.py",
]


def main() -> None:
    for path in _CANDIDATES:
        if path.exists():
            sys.argv[0] = str(path)
            runpy.run_path(str(path), run_name="__main__")
            return
    print("=" * 60)
    print("Beq-Trainer-full.py is missing in this folder.")
    print()
    print("1) Download Beq-Trainer-full.py (complete 9-tab UI)")
    print("2) Put it next to this file in the repo root")
    print("3) Run:  python Beq-Trainer.py")
    print()
    print("Or open the file from your Grok/build artifacts and")
    print("upload it on GitHub: Add file → Upload files")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
