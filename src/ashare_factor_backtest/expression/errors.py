"""Stable expression-engine errors suitable for batch rejection ledgers."""

from __future__ import annotations


class ExpressionError(ValueError):
    def __init__(self, code: str, message: str, expression: str | None = None) -> None:
        if not code or not message:
            raise ValueError("expression error code and message must be non-empty")
        self.code = code
        self.expression = expression
        super().__init__(message)
