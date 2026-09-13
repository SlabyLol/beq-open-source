#!/usr/bin/env python3
"""Beq Trainer — complete 9-tab control panel (self-extracting)."""
from __future__ import annotations
import base64
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PARTS = [(_ROOT / f"beq_trainer_p{i}.txt").read_text(encoding="utf-8").strip() for i in range(3)]


def main() -> None:
    code = zlib.decompress(base64.b64decode("".join(_PARTS))).decode("utf-8")
    ns = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(code, "Beq-Trainer.py", "exec"), ns)


if __name__ == "__main__":
    main()
