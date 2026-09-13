"""
Beq Math Shortcut
=================
Tiny character-level Beq cannot do arithmetic. Detect simple math questions
and answer with real Python math before generation.
"""

from __future__ import annotations

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Pure expression: digits, whitespace, + - * / ** % . ( ) x × ÷
_PURE_EXPR = re.compile(r"^[\d\s\.\+\-\*/%x×÷()]+$")
_PERCENT_OF = re.compile(
    r"(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)",
    re.I,
)
_SIMPLE_EQ = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(\d+(?:\.\d+)?)\s*=?\s*$",
    re.I,
)
_STRIP_WORDS = re.compile(
    r"^(?:what\s+is|what's|calculate|compute|solve|eval(?:uate)?)\s+",
    re.I,
)


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # py < 3.8 compat
        return float(node.n)
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported op")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary")
        return op(_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.10g}"


def _eval_expr(expr: str) -> str | None:
    expr = expr.strip()
    expr = expr.replace("×", "*").replace("x", "*").replace("÷", "/")
    expr = expr.replace("**", "^").replace("^", "**")  # allow 2^3
    if not expr or not re.search(r"\d", expr):
        return None
    # only allow safe characters after normalize
    if not re.match(r"^[\d\s\.\+\-\*/%()]+$", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval(tree)
        if val != val or abs(val) == float("inf"):
            return None
        return _format_num(val)
    except Exception:
        return None


def try_math_answer(prompt: str) -> str | None:
    """
    Return a direct math answer string, or None to fall through to the model.
    Handles:
      - 2+2=
      - 109 * 100
      - What is 25% of 80?
      - calculate 3*(4+5)
    """
    if not prompt or not prompt.strip():
        return None
    text = prompt.strip()

    # Percent: "25% of 80", "what is 25% of 80"
    m = _PERCENT_OF.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return _format_num(a / 100.0 * b)

    # Simple a op b
    m = _SIMPLE_EQ.match(text)
    if m:
        a, op, b = m.group(1), m.group(2), m.group(3)
        op = op.replace("×", "*").replace("x", "*").replace("÷", "/")
        return _eval_expr(f"{a}{op}{b}")

    # Strip leading "what is" / "calculate"
    cleaned = _STRIP_WORDS.sub("", text).strip().rstrip("?").strip()

    # Pure expression left
    if _PURE_EXPR.match(cleaned.replace("**", "*")):
        ans = _eval_expr(cleaned)
        if ans is not None:
            return ans

    # Expression embedded in short prompt (e.g. "solve 2+2")
    m = re.search(r"([\d\(][\d\s\.\+\-\*/x×÷()%]{0,40}[\d\)])", cleaned)
    if m and len(cleaned) < 80:
        ans = _eval_expr(m.group(1))
        if ans is not None:
            return ans

    return None
