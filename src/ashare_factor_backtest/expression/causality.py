"""Dynamic prefix-invariance audit for factor expressions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ashare_factor_backtest.expression.catalog import FieldCatalog, OperatorCatalog
from ashare_factor_backtest.expression.evaluator import (
    BatchEvaluator,
    EvaluationContext,
)


class FactorValueResult(Protocol):
    factor_id: str
    canonical: str
    values: pd.DataFrame


@dataclass(frozen=True)
class PrefixInvarianceReport:
    factor_id: str
    canonical: str
    cutoff: str
    compared_cells: int
    mismatch_cells: int
    max_abs_error: float
    first_mismatch_date: str | None

    @property
    def passed(self) -> bool:
        return self.mismatch_cells == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "audit": "prefix_invariance",
            "passed": self.passed,
            "factor_id": self.factor_id,
            "canonical": self.canonical,
            "cutoff": self.cutoff,
            "compared_cells": self.compared_cells,
            "mismatch_cells": self.mismatch_cells,
            "max_abs_error": self.max_abs_error,
            "first_mismatch_date": self.first_mismatch_date,
        }


def audit_prefix_invariance(
    expression: str,
    context: EvaluationContext,
    *,
    operators: OperatorCatalog,
    fields: FieldCatalog,
    functions: Mapping[str, Callable[..., object]],
    cache_max_bytes: int,
    cutoff: str | pd.Timestamp,
    tolerance: float = 1e-12,
) -> PrefixInvarianceReport:
    """Verify that rows after ``cutoff`` cannot change earlier factor values."""
    if tolerance < 0:
        raise ValueError("causality tolerance must be nonnegative")
    template = next(iter(context.fields.values()))
    cutoff_value = pd.Timestamp(cutoff)
    if cutoff_value not in template.index:
        raise ValueError("causality cutoff must be an evaluation date")
    if cutoff_value == template.index[-1]:
        raise ValueError("causality cutoff must leave at least one future date")

    prefix_context = _prefix_context(context, cutoff_value)
    full = BatchEvaluator(
        operators,
        fields,
        functions,
        cache_max_bytes=cache_max_bytes,
    ).evaluate(expression, context)
    prefix = BatchEvaluator(
        operators,
        fields,
        functions,
        cache_max_bytes=cache_max_bytes,
    ).evaluate(expression, prefix_context)
    return compare_prefix_results(full, prefix, cutoff=cutoff_value, tolerance=tolerance)


def compare_prefix_results(
    full: FactorValueResult,
    prefix: FactorValueResult,
    *,
    cutoff: str | pd.Timestamp,
    tolerance: float = 1e-12,
) -> PrefixInvarianceReport:
    if full.factor_id != prefix.factor_id or full.canonical != prefix.canonical:
        raise ValueError("full and prefix expression identities must match")
    if tolerance < 0:
        raise ValueError("causality tolerance must be nonnegative")
    cutoff_value = pd.Timestamp(cutoff)
    expected = full.values.loc[prefix.values.index, prefix.values.columns]
    expected_values = expected.to_numpy(dtype=float, na_value=np.nan)
    actual_values = prefix.values.to_numpy(dtype=float, na_value=np.nan)

    expected_finite = np.isfinite(expected_values)
    actual_finite = np.isfinite(actual_values)
    mismatch = expected_finite ^ actual_finite
    jointly_finite = expected_finite & actual_finite
    differences = np.zeros_like(expected_values, dtype=float)
    differences[jointly_finite] = np.abs(
        expected_values[jointly_finite] - actual_values[jointly_finite]
    )
    mismatch |= jointly_finite & (differences > tolerance)
    mismatch_cells = int(mismatch.sum())
    mismatch_rows = np.flatnonzero(mismatch.any(axis=1))
    first_mismatch = (
        pd.Timestamp(expected.index[int(mismatch_rows[0])]).date().isoformat()
        if mismatch_rows.size
        else None
    )
    max_abs_error = (
        float(differences[jointly_finite].max()) if jointly_finite.any() else 0.0
    )
    return PrefixInvarianceReport(
        factor_id=full.factor_id,
        canonical=full.canonical,
        cutoff=cutoff_value.date().isoformat(),
        compared_cells=int(expected_values.size),
        mismatch_cells=mismatch_cells,
        max_abs_error=max_abs_error,
        first_mismatch_date=first_mismatch,
    )


def _prefix_context(
    context: EvaluationContext,
    cutoff: pd.Timestamp,
) -> EvaluationContext:
    first_panel = next(iter(context.fields.values()))
    keep = first_panel.index <= cutoff
    index = first_panel.index[keep]
    return EvaluationContext(
        fields={name: panel.loc[index].copy() for name, panel in context.fields.items()},
        dataset_versions=dict(context.dataset_versions),
        universe_policy=context.universe_policy,
        date_range=(context.date_range[0], cutoff.date().isoformat()),
        universe_size=context.universe_size.loc[index].copy(),
        universe_mask=(
            context.universe_mask.loc[index].copy()
            if context.universe_mask is not None
            else None
        ),
        evaluation_price_basis=context.evaluation_price_basis,
    )
