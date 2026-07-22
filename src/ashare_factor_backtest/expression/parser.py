"""Parse a tiny function-only expression language without executing Python."""

from __future__ import annotations

import ast
import math

from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.expression.model import CallNode, ConstantNode, ExprNode, FieldNode


MAX_DEPTH = 8
MAX_NODES = 64


def parse_expression(text: str) -> ExprNode:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("PARSE_ERROR", "expression must not be empty", text)
    try:
        parsed = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ExpressionError("PARSE_ERROR", f"invalid expression syntax: {error.msg}", text) from error
    count = 0

    def convert(node: ast.AST, depth: int) -> ExprNode:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise ExpressionError("RESOURCE_LIMIT", "expression exceeds node limit", text)
        if depth > MAX_DEPTH:
            raise ExpressionError("RESOURCE_LIMIT", "expression exceeds depth limit", text)
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ExpressionError("UNSAFE_SYNTAX", "private names are forbidden", text)
            return FieldNode(node.id)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExpressionError("UNSAFE_SYNTAX", "only finite numeric constants are allowed", text)
            if isinstance(value, float) and not math.isfinite(value):
                raise ExpressionError("UNSAFE_SYNTAX", "only finite numeric constants are allowed", text)
            return ConstantNode(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = convert(node.operand, depth + 1)
            if isinstance(operand, ConstantNode):
                return ConstantNode(-operand.value)
            raise ExpressionError("UNSAFE_SYNTAX", "unary minus only accepts constants", text)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords or node.func.id.startswith("_"):
                raise ExpressionError("UNSAFE_SYNTAX", "only direct positional function calls are allowed", text)
            return CallNode(node.func.id, tuple(convert(arg, depth + 1) for arg in node.args))
        raise ExpressionError(
            "UNSAFE_SYNTAX", f"forbidden syntax node: {type(node).__name__}", text
        )

    return convert(parsed.body, 1)


def referenced_fields(text: str) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(node: ExprNode) -> None:
        if isinstance(node, FieldNode):
            names.add(node.name)
        elif isinstance(node, CallNode):
            for argument in node.arguments:
                visit(argument)

    visit(parse_expression(text))
    return tuple(sorted(names))
