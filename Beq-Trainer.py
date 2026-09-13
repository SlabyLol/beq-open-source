#!/usr/bin/env python3
"""Beq Trainer — complete 9-tab control panel.

Joins beq_trainer_src_0.txt … beq_trainer_src_7.txt and runs the full app.
"""
from __future__ import annotations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> None:
    chunks = []
    for i in range(8):
        path = _ROOT / f"beq_trainer_src_{i}.txt"
        if not path.exists():
            raise SystemExit(
                f"Missing {path.name}. Re-clone the repo so all beq_trainer_src_*.txt files exist."
            )
        chunks.append(path.read_text(encoding="utf-8"))
    code = "".join(chunks)
    ns = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, "Beq-Trainer.py", "exec"), ns)


if __name__ == "__main__":
    main()
