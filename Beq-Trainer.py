#!/usr/bin/env python3
"""Beq Trainer — complete 9-tab control panel.

Tabs: Train | Config & name | Knowledge | Identity | Data | Checkpoints | Generate | Tools | Help
Requires beq_trainer_p0.txt … beq_trainer_p3.txt next to this file (same folder).
"""
from __future__ import annotations
import base64
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parts = []
    for i in range(4):
        path = _ROOT / f"beq_trainer_p{i}.txt"
        if not path.exists():
            raise SystemExit(
                f"Missing {path.name}. Clone the full repo so all beq_trainer_p*.txt files are present."
            )
        parts.append(path.read_text(encoding="utf-8").strip())
    code = zlib.decompress(base64.b64decode("".join(parts))).decode("utf-8")
    ns = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, "Beq-Trainer.py", "exec"), ns)


if __name__ == "__main__":
    main()
