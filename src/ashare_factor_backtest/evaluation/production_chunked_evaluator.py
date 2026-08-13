"""Memory-bounded expression evaluation over yearly production-universe chunks."""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ashare_factor_backtest.expression.compiler import compile_expression
from ashare_factor_backtest.expression.evaluator import BatchEvaluator
from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.expression.catalog import OperatorCatalog
from ashare_factor_backtest.expression.operators.registry import (
    build_production_operator_catalog,
)
from ashare_factor_backtest.evaluation.production_context import (
    build_production_evaluation_context,
    build_production_evaluation_context_from_batches,
    price_carry_state_before,
)
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


@dataclass(frozen=True)
class EvaluatedChunk:
    year: int
    calculation_start: str
    calculation_end: str
    load_start: str
    load_end: str
    date_count: int
    security_count: int
    finite_cells: int


@dataclass(frozen=True)
class ChunkedEvaluationResult:
    factor_id: str
    canonical: str
    lookback: int
    values: pd.DataFrame
    chunks: tuple[EvaluatedChunk, ...]


FrameLoader = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


def evaluate_expression_by_year(
    expression: str,
    *,
    chunks: Sequence[YearChunk],
    frame_loader: FrameLoader,
    dataset_version: str,
    view: str,
    cache_max_bytes: int,
    required_fields: set[str] | None = None,
    spill_to_disk: bool = False,
    operator_catalog_builder: Callable[
        [], tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]
    ] = build_production_operator_catalog,
) -> ChunkedEvaluationResult:
    if not chunks:
        raise ValueError("production expression chunks must not be empty")
    if cache_max_bytes <= 0:
        raise ValueError("expression cache size must be positive")

    operators, functions = operator_catalog_builder()
    factor_id: str | None = None
    canonical: str | None = None
    expression_lookback: int | None = None
    spill = (
        tempfile.TemporaryDirectory(prefix="ashare-backtest-chunks-")
        if spill_to_disk
        else None
    )
    outputs: list[pd.DataFrame | Path] = []
    reports: list[EvaluatedChunk] = []
    calculation_dates: set[pd.Timestamp] = set()
    additional_field_specs = getattr(frame_loader, "additional_field_specs", {})
    additional_dataset_versions = getattr(
        frame_loader, "additional_dataset_versions", {}
    )
    initial_price_values = None

    for position, chunk in enumerate(chunks, start=1):
        if chunk.calculation_start > chunk.calculation_end:
            raise ValueError(f"production chunk {chunk.year} has an empty calculation period")
        iterator = getattr(frame_loader, "iter_frames", None)
        if iterator is None:
            frame = frame_loader(chunk.load_start, chunk.load_end)
            if frame.empty:
                raise ValueError(f"production chunk {chunk.year} loader returned no rows")
            if "forward_return" in frame.columns:
                raise ValueError("production expression chunks must not contain forward_return")
            catalog, context = build_production_evaluation_context(
                frame,
                dataset_version=dataset_version,
                view=view,
                required_fields=required_fields,
                additional_field_specs=additional_field_specs,
                additional_dataset_versions=additional_dataset_versions,
                initial_price_values=initial_price_values,
            )
        else:
            frame = None
            catalog, context = build_production_evaluation_context_from_batches(
                iterator(chunk.load_start, chunk.load_end),
                dataset_version=dataset_version,
                view=view,
                required_fields=required_fields,
                additional_field_specs=additional_field_specs,
                additional_dataset_versions=additional_dataset_versions,
                initial_price_values=initial_price_values,
            )
        compiled = compile_expression(expression, operators, catalog)
        if chunk.max_lookback < compiled.lookback:
            raise ValueError(
                f"production chunk {chunk.year} lookback {chunk.max_lookback} "
                f"is below expression lookback {compiled.lookback}"
            )
        if factor_id is None:
            factor_id = compiled.factor_id
            canonical = compiled.canonical
            expression_lookback = compiled.lookback
        elif compiled.factor_id != factor_id:
            raise RuntimeError("expression identity changed between production chunks")

        evaluator = BatchEvaluator(
            operators, catalog, functions, cache_max_bytes=cache_max_bytes
        )
        evaluated = evaluator.evaluate(expression, context)
        values = evaluated.values.loc[
            evaluated.values.index.to_series().between(
                chunk.calculation_start, chunk.calculation_end
            )
        ].copy()
        if values.empty:
            raise ValueError(f"production chunk {chunk.year} has no calculation dates")
        duplicate_dates = calculation_dates.intersection(values.index)
        if duplicate_dates:
            first = min(duplicate_dates).date().isoformat()
            raise ValueError(f"production chunks contain duplicate calculation date: {first}")
        calculation_dates.update(pd.Timestamp(date) for date in values.index)
        if spill is None:
            outputs.append(values)
        else:
            path = Path(spill.name) / f"chunk-{position:03d}.parquet"
            values.to_parquet(path)
            outputs.append(path)
        reports.append(
            EvaluatedChunk(
                year=chunk.year,
                calculation_start=chunk.calculation_start.date().isoformat(),
                calculation_end=chunk.calculation_end.date().isoformat(),
                load_start=chunk.load_start.date().isoformat(),
                load_end=chunk.load_end.date().isoformat(),
                date_count=len(values),
                security_count=len(values.columns),
                finite_cells=int(values.notna().sum().sum()),
            )
        )
        print(
            f"[production-chunk] expression={compiled.factor_id} "
            f"chunk={position}/{len(chunks)} year={chunk.year} dates={len(values)}",
            flush=True,
        )
        if position < len(chunks):
            initial_price_values = price_carry_state_before(
                context,
                before=chunks[position].load_start,
            )
        del evaluated, evaluator, context, catalog, frame, values
        gc.collect()
        pa.default_memory_pool().release_unused()

    materialized = (
        [pd.read_parquet(path) for path in outputs]
        if spill is not None
        else outputs
    )
    combined = pd.concat(materialized, axis=0, join="outer").sort_index().sort_index(axis=1)
    if spill is not None:
        spill.cleanup()
    if combined.index.duplicated().any():
        raise RuntimeError("production chunk concatenation created duplicate dates")
    assert factor_id is not None
    assert canonical is not None
    assert expression_lookback is not None
    return ChunkedEvaluationResult(
        factor_id=factor_id,
        canonical=canonical,
        lookback=expression_lookback,
        values=combined,
        chunks=tuple(reports),
    )


def evaluate_expressions_by_year(
    expressions: Sequence[str],
    *,
    chunks: Sequence[YearChunk],
    frame_loader: FrameLoader,
    dataset_version: str,
    view: str,
    cache_max_bytes: int,
    required_fields: set[str] | None = None,
    spill_to_disk: bool = False,
    operator_catalog_builder: Callable[
        [], tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]
    ] = build_production_operator_catalog,
) -> tuple[ChunkedEvaluationResult, ...]:
    """Evaluate a canonical-unique batch while loading each time chunk once."""

    requested = tuple(str(expression).strip() for expression in expressions)
    if not requested or any(not expression for expression in requested):
        raise ValueError("production expression batch must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("production expression batch must contain unique expressions")
    expression_positions = {
        expression: position for position, expression in enumerate(requested)
    }
    if not chunks:
        raise ValueError("production expression chunks must not be empty")
    if cache_max_bytes <= 0:
        raise ValueError("expression cache size must be positive")

    operators, functions = operator_catalog_builder()
    spill = (
        tempfile.TemporaryDirectory(prefix="ashare-backtest-batch-chunks-")
        if spill_to_disk
        else None
    )
    outputs: dict[str, list[pd.DataFrame | Path]] = {
        expression: [] for expression in requested
    }
    reports: dict[str, list[EvaluatedChunk]] = {
        expression: [] for expression in requested
    }
    identities: dict[str, tuple[str, str, int]] = {}
    calculation_dates: dict[str, set[pd.Timestamp]] = {
        expression: set() for expression in requested
    }
    additional_field_specs = getattr(frame_loader, "additional_field_specs", {})
    additional_dataset_versions = getattr(
        frame_loader, "additional_dataset_versions", {}
    )
    initial_price_values = None

    for position, chunk in enumerate(chunks, start=1):
        if chunk.calculation_start > chunk.calculation_end:
            raise ValueError(
                f"production chunk {chunk.year} has an empty calculation period"
            )
        iterator = getattr(frame_loader, "iter_frames", None)
        if iterator is None:
            frame = frame_loader(chunk.load_start, chunk.load_end)
            if frame.empty:
                raise ValueError(
                    f"production chunk {chunk.year} loader returned no rows"
                )
            if "forward_return" in frame.columns:
                raise ValueError(
                    "production expression chunks must not contain forward_return"
                )
            catalog, context = build_production_evaluation_context(
                frame,
                dataset_version=dataset_version,
                view=view,
                required_fields=required_fields,
                additional_field_specs=additional_field_specs,
                additional_dataset_versions=additional_dataset_versions,
                initial_price_values=initial_price_values,
            )
        else:
            frame = None
            catalog, context = build_production_evaluation_context_from_batches(
                iterator(chunk.load_start, chunk.load_end),
                dataset_version=dataset_version,
                view=view,
                required_fields=required_fields,
                additional_field_specs=additional_field_specs,
                additional_dataset_versions=additional_dataset_versions,
                initial_price_values=initial_price_values,
            )

        compiled_by_canonical: dict[str, str] = {}
        for expression in requested:
            compiled = compile_expression(expression, operators, catalog)
            if chunk.max_lookback < compiled.lookback:
                raise ValueError(
                    f"production chunk {chunk.year} lookback {chunk.max_lookback} "
                    f"is below expression lookback {compiled.lookback}"
                )
            prior = identities.setdefault(
                expression,
                (compiled.factor_id, compiled.canonical, compiled.lookback),
            )
            if prior != (compiled.factor_id, compiled.canonical, compiled.lookback):
                raise RuntimeError(
                    "expression identity changed between production chunks"
                )
            if compiled.canonical in compiled_by_canonical:
                raise ValueError(
                    "production expression batch contains duplicate canonical formulas"
                )
            compiled_by_canonical[compiled.canonical] = expression

        evaluator = BatchEvaluator(
            operators, catalog, functions, cache_max_bytes=cache_max_bytes
        )
        evaluated_batch, rejected = evaluator.evaluate_many(requested, context)
        if rejected:
            first = rejected[0]
            raise ExpressionError(first.code, first.message, first.expression)
        if len(evaluated_batch) != len(requested):
            raise RuntimeError("production batch evaluator omitted a formula")

        for evaluated in evaluated_batch:
            expression = compiled_by_canonical[evaluated.canonical]
            values = evaluated.values.loc[
                evaluated.values.index.to_series().between(
                    chunk.calculation_start, chunk.calculation_end
                )
            ].copy()
            if values.empty:
                raise ValueError(
                    f"production chunk {chunk.year} has no calculation dates"
                )
            duplicate_dates = calculation_dates[expression].intersection(values.index)
            if duplicate_dates:
                first = min(duplicate_dates).date().isoformat()
                raise ValueError(
                    f"production chunks contain duplicate calculation date: {first}"
                )
            calculation_dates[expression].update(
                pd.Timestamp(date) for date in values.index
            )
            if spill is None:
                outputs[expression].append(values)
            else:
                formula_index = expression_positions[expression]
                path = (
                    Path(spill.name)
                    / f"factor-{formula_index:04d}-chunk-{position:03d}.parquet"
                )
                values.to_parquet(path)
                outputs[expression].append(path)
            reports[expression].append(
                EvaluatedChunk(
                    year=chunk.year,
                    calculation_start=chunk.calculation_start.date().isoformat(),
                    calculation_end=chunk.calculation_end.date().isoformat(),
                    load_start=chunk.load_start.date().isoformat(),
                    load_end=chunk.load_end.date().isoformat(),
                    date_count=len(values),
                    security_count=len(values.columns),
                    finite_cells=int(values.notna().sum().sum()),
                )
            )
        print(
            f"[production-batch-chunk] formulas={len(requested)} "
            f"chunk={position}/{len(chunks)} year={chunk.year}",
            flush=True,
        )
        if position < len(chunks):
            initial_price_values = price_carry_state_before(
                context,
                before=chunks[position].load_start,
            )
        del evaluated_batch, evaluator, context, catalog, frame
        gc.collect()
        pa.default_memory_pool().release_unused()

    results: list[ChunkedEvaluationResult] = []
    for expression in requested:
        materialized = (
            [pd.read_parquet(path) for path in outputs[expression]]
            if spill is not None
            else outputs[expression]
        )
        combined = (
            pd.concat(materialized, axis=0, join="outer")
            .sort_index()
            .sort_index(axis=1)
        )
        if combined.index.duplicated().any():
            raise RuntimeError("production chunk concatenation created duplicate dates")
        factor_id, canonical, lookback = identities[expression]
        results.append(
            ChunkedEvaluationResult(
                factor_id=factor_id,
                canonical=canonical,
                lookback=lookback,
                values=combined,
                chunks=tuple(reports[expression]),
            )
        )
    if spill is not None:
        spill.cleanup()
    return tuple(results)
