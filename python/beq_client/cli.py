"""CLI: beq generate / status / languages"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .client import Beq, BeqError
from .languages import list_languages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="beq", description="Beq API client")
    p.add_argument("--url", default=os.environ.get("BEQ_URL", "http://127.0.0.1:8000"))
    p.add_argument("--key", default=os.environ.get("BEQ_API_KEY", ""))
    p.add_argument("--lang", "-l", default=os.environ.get("BEQ_LANG", "en"))
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate text")
    g.add_argument("prompt", nargs="+")
    g.add_argument("--max-tokens", type=int, default=80)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--json", action="store_true")

    sub.add_parser("status", help="Server status")
    sub.add_parser("languages", help="List language codes")

    args = p.parse_args(argv)
    client = Beq(args.url, api_key=args.key or None, language=args.lang)

    try:
        if args.cmd == "languages":
            for row in list_languages():
                print(f"{row['code']:4}  {row['name']}")
            return 0
        if args.cmd == "status":
            print(json.dumps(client.status(), indent=2))
            return 0
        if args.cmd == "generate":
            prompt = " ".join(args.prompt)
            if args.json:
                print(json.dumps(
                    client.generate_full(prompt, max_tokens=args.max_tokens, temperature=args.temperature),
                    indent=2, ensure_ascii=False,
                ))
            else:
                print(client.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature))
            return 0
    except BeqError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
