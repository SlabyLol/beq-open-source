"""
Simple character-level tokenizer for Beq.
Fully self-contained. No external dependencies beyond Python.
"""

from __future__ import annotations
import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer. Easy to understand and fully yours."""

    def __init__(self, text: str | None = None):
        if text is not None:
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
        else:
            self.stoi = {}
            self.itos = {}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in tokens)

    def save(self, path: str | Path):
        path = Path(path)
        data = {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


class SimpleBPETokenizer:
    """
    Very simple BPE-style tokenizer (educational version).
    Still fully self-contained.
    """

    def __init__(self):
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def train(self, text: str, vocab_size: int = 500):
        """Train a tiny BPE vocabulary."""
        # Start with characters
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Simple frequency-based merges (educational, not production BPE)
        from collections import Counter
        words = text.split()
        tokens = list(text)

        while len(self.stoi) < vocab_size:
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1

            if not pairs:
                break

            best = pairs.most_common(1)[0][0]
            new_token = best[0] + best[1]
            if new_token in self.stoi:
                break

            idx = len(self.stoi)
            self.stoi[new_token] = idx
            self.itos[idx] = new_token
            self.merges.append(best)

            # Apply merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        # Greedy longest-match encoding
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i : i + length]
                if substr in self.stoi:
                    tokens.append(self.stoi[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                # Unknown character → skip or use a special token
                i += 1
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in token_ids)

    def save(self, path: str | Path):
        path = Path(path)
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "merges": self.merges,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleBPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        return tok
