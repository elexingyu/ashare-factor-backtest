"""Compile expressions without depending on a CLI or data supplier."""

from __future__ import annotations

from pathlib import Path

from ashare_factor_backtest.expression.compiler import explain_expression
from ashare_factor_backtest.expression.operators.registry import build_production_operator_catalog
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.expression.production_fields import (
    build_production_field_catalog,
)
from ashare_factor_backtest.application.production_job_field_catalog import (
    build_job_field_catalog,
)
from ashare_factor_backtest.application.production_job import load_production_job


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

    def execute_for_job(self, expression: str, job_path: Path) -> dict[str, object]:
        fields = build_job_field_catalog(job_path, referenced_fields(expression))
        return explain_expression(expression, self.operators, fields)

    def schema(self) -> dict[str, object]:
        return {
            "field_catalog": self.fields.version_payload(),
            "operator_catalog": self.operators.export_payload(),
        }

    def schema_for_job(self, job_path: Path) -> dict[str, object]:
        job = load_production_job(job_path)
        field_names = (
            *(spec.name for spec in self.fields.specs()),
            *(field for binding in job.plugins for field in binding.fields),
        )
        fields = build_job_field_catalog(job_path, field_names)
        return {
            "field_catalog": fields.version_payload(),
            "operator_catalog": self.operators.export_payload(),
        }
