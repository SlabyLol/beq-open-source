#!/usr/bin/env python3
"""Beq Trainer – full desktop control panel."""
from __future__ import annotations
import base64
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_A = (_ROOT / "beq_trainer_payload_a.txt").read_text(encoding="utf-8").strip()
_B = (_ROOT / "beq_trainer_payload_b.txt").read_text(encoding="utf-8").strip()


def main() -> None:
    code = zlib.decompress(base64.b64decode(_A + _B)).decode()
    ns = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, "Beq-Trainer.py", "exec"), ns)


if __name__ == "__main__":
    main()
