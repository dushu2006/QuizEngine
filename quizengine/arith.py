"""Safe arithmetic evaluation shared by the local rule solver and the offline model.

No ``eval()`` anywhere in the codebase: expressions are parsed with ``ast`` and
walked with an explicit operator whitelist.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Optional

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

#: ``12 * 7``, ``3.5 + 2``, ``(40 - 32) / 2``
EXPRESSION_RE = re.compile(
    r"\(?\s*-?\d+(?:\.\d+)?\s*(?:[-+*/%]|\*\*|\^)\s*-?\d+(?:\.\d+)?\s*(?:(?:[-+*/%]|\*\*|\^)\s*-?\d+(?:\.\d+)?\s*)*\)?"
)


def safe_eval(expression: str, *, max_exponent: float = 16.0) -> Optional[float]:
    """Evaluate a plain arithmetic expression, or return ``None``."""
    text = (expression or "").strip().replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    text = text.strip("()") if text.startswith("(") and text.endswith(")") and _balanced(text) else text
    if not text:
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > max_exponent:
                raise ValueError("exponent out of bounds")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ZeroDivisionError("division by zero")
            return float(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return float(_UNARY_OPS[type(node.op)](walk(node.operand)))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        return walk(tree)
    except Exception:
        return None


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def find_expression(text: str) -> Optional[str]:
    """First arithmetic-looking expression in a question."""
    match = EXPRESSION_RE.search(text or "")
    return match.group(0) if match else None


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


def numbers_match(value: float, candidate: str, tolerance: float = 1e-6) -> bool:
    """True when ``candidate`` (an option's text) states ``value``."""
    for number in numbers_in(candidate):
        if abs(number - value) <= tolerance:
            return True
        # "1,024" / "1024 units" / "$12.50" all normalize to the same number.
        if abs(number - round(value)) <= tolerance and float(value).is_integer():
            return True
    return False


__all__ = ["EXPRESSION_RE", "find_expression", "numbers_in", "numbers_match", "safe_eval"]
