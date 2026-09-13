"""
Beq Math Shortcut
==================

Beq's language model is a tiny, from-scratch, character-level transformer
trained on ~11KB of text. It has no arithmetic ability whatsoever — asking
it "what's 109 * 100" will always produce plausible-looking nonsense,
because it never learned arithmetic (and realistically never will at this
scale/data size).

Rather than pretend the model can do this, we detect simple arithmetic
expressions before generation and answer them directly and correctly with
real Python arithmetic. This only handles clean, single expressions
(+ - * / and parentheses, digits only) — never arbitrary code execution.
"""

from __future__ import annotations

import re

# Only digits, whitespace, + - * / . ( ) and x/× as multiply — nothing else.
_MATH_CHARS = re.compile(r"^[\d\s\.\+\-\*/x×÷()]+$")
_HAS_DIGIT_AND_OP = re.compile(r"\d.*[\+\-\*/x×÷].*\d|\d.*[\+\-\*/x×÷]")


def _extract_expression(prompt: str) -> str | None:
    """Pull out something that looks like a pure arithmetic expression."""
