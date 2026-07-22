"""Batch evaluation for compiled factor expressions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.cache import PanelLRU
from ashare_factor_backtest.expression.catalog import FieldCatalog, OperatorCatalog
from ashare_factor_backtest.expression.compiler import canonical_expression, compile_expression
from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.expression.model import (
    CompiledExpression,
    ConstantNode,
    ExprNode,
    FieldNode,
    ValueType,
)


@dataclass(frozen=True)
class EvaluationContext:
    fields: dict[str, pd.DataFrame]
    dataset_versions: dict[str, str]
    universe_policy: str
    date_range: tuple[str, str]
    universe_size: pd.Series
    universe_mask: pd.DataFrame | None = None
    evaluation_price_basis: PriceBasis = PriceBasis.HFQ_PIT


@dataclass(frozen=True)
class EvaluationResult:
    factor_id: str
    experiment_id: str
    canonical: str
    values: pd.DataFrame
    coverage_rate: float
    warnings: tuple[str, ...]
    cache_stats: dict[str, int]
    price_basis: PriceBasis
    universe_size: pd.Series


@dataclass(frozen=True)
class RejectedExpression:
    expression: str
    code: str
    message: str


class BatchEvaluator:
    def __init__(
        self,
        operators: OperatorCatalog,
        fields: FieldCatalog,
        functions: Mapping[str, Callable[..., object]],
        *,
        cache_max_bytes: int,
    ) -> None:
        self.operators = operators
        self.fields = fields
        self.functions = dict(functions)
        self.cache = PanelLRU(cache_max_bytes)

    def evaluate(self, expression: str, context: EvaluationContext) -> EvaluationResult:
        results, rejected = self.evaluate_many((expression,), context)
        if rejected:
            error = rejected[0]
            raise ExpressionError(error.code, error.message, error.expression)
        return results[0]

    def evaluate_many(
        self, expressions: Sequence[str], context: EvaluationContext
    ) -> tuple[list[EvaluationResult], list[RejectedExpression]]:
        template = _validate_context(context, self.fields)
        universe_mask = _validated_universe_mask(context, template)
        context_id = _context_identity(context)
        self.cache.clear(reset_stats=True)
        results: list[EvaluationResult] = []
        rejected: list[RejectedExpression] = []
        for expression in expressions:
            try:
                compiled = compile_expression(expression, self.operators, self.fields)
                value = self._evaluate_node(
                    compiled.root,
                    context,
                    template,
                    context_id,
                    expression,
                    universe_mask,
                )
                if compiled.output_type is not ValueType.PANEL_FLOAT:
                    raise ExpressionError(
                        "TYPE_MISMATCH", "top-level expression must return panel_float", expression
                    )
                assert isinstance(value, pd.DataFrame)
                if universe_mask is not None:
                    value = value.where(universe_mask)
                finite_count = int(np.isfinite(value.to_numpy(dtype=float, na_value=np.nan)).sum())
                coverage = finite_count / value.size if value.size else 0.0
                results.append(
                    EvaluationResult(
                        factor_id=compiled.factor_id,
                        experiment_id=_experiment_identity(
                            compiled,
                            context_id,
                            self.operators,
                            self.fields,
                        ),
                        canonical=compiled.canonical,
                        values=value,
                        coverage_rate=coverage,
                        warnings=compiled.warnings,
                        cache_stats=self.cache.stats().as_dict(),
                        price_basis=compiled.price_basis or context.evaluation_price_basis,
                        universe_size=context.universe_size.copy(),
                    )
                )
            except ExpressionError as error:
                rejected.append(RejectedExpression(expression, error.code, str(error)))
        return results, rejected

    def _evaluate_node(
        self,
        node: ExprNode,
        context: EvaluationContext,
        template: pd.DataFrame,
        context_id: str,
        expression: str,
        universe_mask: pd.DataFrame | None,
    ) -> object:
        if isinstance(node, FieldNode):
            return context.fields[node.name]
        if isinstance(node, ConstantNode):
            return node.value

        key = f"{context_id}:{canonical_expression(node)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        arguments = [
            self._evaluate_node(
                child,
                context,
                template,
                context_id,
                expression,
                universe_mask,
            )
            for child in node.arguments
        ]
        function = self.functions.get(node.operator)
        if function is None:
            raise ExpressionError("UNKNOWN_OPERATOR", f"operator has no implementation: {node.operator}", expression)
        spec = self.operators.resolve(node.operator)
        if spec.category == "cross_section" and universe_mask is not None:
            arguments = [
                argument.where(universe_mask)
                if isinstance(argument, pd.DataFrame)
                else argument
                for argument in arguments
            ]
        try:
            result = function(*arguments)
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ExpressionError("EVALUATION_ERROR", str(error), expression) from error
        result = _validate_result(result, spec.output_type, template, expression)
        if isinstance(result, pd.DataFrame):
            self.cache.put(key, result)
        return result


def _validate_context(context: EvaluationContext, catalog: FieldCatalog) -> pd.DataFrame:
    if not context.fields:
        raise ValueError("evaluation context must contain fields")
    expected_names = {spec.name for spec in catalog.specs()}
    if not expected_names.issubset(context.fields):
        missing = sorted(expected_names.difference(context.fields))
        raise ValueError(f"evaluation context is missing fields: {missing}")
    template = next(iter(context.fields.values()))
    if not isinstance(template, pd.DataFrame) or template.empty:
        raise ValueError("field panels must be non-empty DataFrames")
    if not template.index.is_unique or not template.columns.is_unique:
        raise ValueError("field panel labels must be unique")
    if not template.index.is_monotonic_increasing:
        raise ValueError("field panel index must be monotonic increasing")
    for name, panel in context.fields.items():
        if not isinstance(panel, pd.DataFrame):
            raise ValueError(f"field {name} must be a DataFrame")
        if not panel.index.equals(template.index) or not panel.columns.equals(template.columns):
            raise ValueError("all field panels must be exactly aligned")
        try:
            values = panel.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as error:
            raise ValueError(f"field {name} must be numeric") from error
        if np.isinf(values).any():
            raise ValueError(f"field {name} contains infinity")
    if not context.universe_size.index.equals(template.index):
        raise ValueError("universe_size must be aligned with field index")
    return template


def _validated_universe_mask(
    context: EvaluationContext,
    template: pd.DataFrame,
) -> pd.DataFrame | None:
    mask = context.universe_mask
    if mask is None:
        return None
    if not isinstance(mask, pd.DataFrame):
        raise ValueError("universe_mask must be a DataFrame")
    if not mask.index.equals(template.index) or not mask.columns.equals(template.columns):
        raise ValueError("universe_mask must be exactly aligned with field panels")
    if mask.isna().to_numpy().any() or not all(
        pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes
    ):
        raise ValueError("universe_mask must contain nonmissing boolean values")
    return mask.astype(bool)


def _validate_result(
    result: object, output_type: ValueType, template: pd.DataFrame, expression: str
) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame):
        raise ExpressionError("EVALUATION_ERROR", "operator did not return a DataFrame", expression)
    if not result.index.equals(template.index) or not result.columns.equals(template.columns):
        raise ExpressionError("EVALUATION_ERROR", "operator returned a misaligned panel", expression)
    if output_type is ValueType.PANEL_BOOL:
        try:
            return result.astype("boolean")
        except (TypeError, ValueError) as error:
            raise ExpressionError("EVALUATION_ERROR", "boolean operator returned invalid values", expression) from error
    try:
        values = result.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ExpressionError("EVALUATION_ERROR", "numeric operator returned invalid values", expression) from error
    if np.isinf(values).any():
        raise ExpressionError("NONFINITE_OUTPUT", "operator returned infinity", expression)
    return result.astype(float)


def _context_identity(context: EvaluationContext) -> str:
    payload = {
        "dataset_versions": sorted(context.dataset_versions.items()),
        "universe_policy": context.universe_policy,
        "universe_mask": _universe_mask_identity(context.universe_mask),
        "date_range": context.date_range,
        "price_basis": context.evaluation_price_basis.value,
    }
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:16]


def _experiment_identity(
    compiled: CompiledExpression,
    context_id: str,
    operators: OperatorCatalog,
    fields: FieldCatalog,
) -> str:
    payload = {
        "factor_id": compiled.factor_id,
        "operator_catalog_version": operators.version,
        "field_catalog_version": fields.version,
        "context": context_id,
    }
    return "e_" + hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:16]


def _universe_mask_identity(mask: pd.DataFrame | None) -> str | None:
    if mask is None:
        return None
    digest = hashlib.sha256()
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(
        pd.util.hash_pandas_object(mask.index, index=False).to_numpy().tobytes()
    )
    digest.update(
        pd.util.hash_pandas_object(mask.columns, index=False).to_numpy().tobytes()
    )
    digest.update(np.packbits(mask.to_numpy(dtype=bool), bitorder="little").tobytes())
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
