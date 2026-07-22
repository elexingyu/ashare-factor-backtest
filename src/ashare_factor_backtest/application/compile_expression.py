"""Compile expressions without depending on a CLI or data supplier."""

from __future__ import annotations

from ashare_factor_backtest.expression.compiler import explain_expression
from ashare_factor_backtest.expression.operators.registry import build_production_operator_catalog
from ashare_factor_backtest.expression.production_fields import (
    build_production_field_catalog,
)


class CompileExpressionService:
    def __init__(self) -> None:
        self.operators, _ = build_production_operator_catalog()
        self.fields = build_production_field_catalog()

    @property
    def operator_catalog_version(self) -> str:
        return self.operators.version

    @property
    def field_catalog_version(self) -> str:
        return self.fields.version

    def execute(self, expression: str) -> dict[str, object]:
        return explain_expression(expression, self.operators, self.fields)

    def schema(self) -> dict[str, object]:
        return {
            "field_catalog": self.fields.version_payload(),
            "operator_catalog": self.operators.export_payload(),
        }
