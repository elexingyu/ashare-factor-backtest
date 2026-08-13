"""Return-blind validity audit for one production factor expression."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ashare_factor_backtest.application.audit_causality import CausalityAuditService
from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.expression.compiler import compile_expression
from ashare_factor_backtest.expression.evaluator import BatchEvaluator
from ashare_factor_backtest.expression.operators.registry import (
    build_production_operator_catalog,
)
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.evaluation.production_context import (
    build_production_evaluation_context_from_batches,
    price_carry_state_before,
)


_FORBIDDEN_COLUMNS = frozenset(
    {"forward_return", "future_return", "target", "target_return"}
)


class FactorValidityAuditService:
    """Audit expression execution, PIT coverage, and prefix invariance without returns."""

    def audit(
        self,
        path: Path,
        expression: str,
        *,
        work_root: Path,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        total_start = perf_counter()
        prepared = ProductionJobService().prepare(
            path,
            expression=expression,
            validation_cache_root=Path(work_root) / "cache" / "plugin_validation",
        )
        operators, functions = build_production_operator_catalog()
        requested = set(referenced_fields(expression))
        field_counts = {
            name: {"finite": 0, "universe": 0} for name in sorted(requested)
        }
        totals = {
            "universe": 0,
            "all_inputs_finite": 0,
            "factor_finite": 0,
            "duplicate_rows": 0,
            "forbidden_columns": set(),
        }
        yearly: list[dict[str, object]] = []
        canonical: str | None = None
        factor_id: str | None = None
        lookback: int | None = None
        output_type: str | None = None
        price_basis: str | None = None
        unit_lineage: str | None = None
        field_semantics: dict[str, dict[str, object]] = {}
        initial_price_values = None

        evaluation_start = perf_counter()
        for position, chunk in enumerate(prepared.chunks):
            frame_stats: dict[str, Any] = {
                "duplicate_rows": 0,
                "forbidden_columns": set(),
            }

            def audited_frames() -> Iterable[pd.DataFrame]:
                for frame in prepared.frame_loader.iter_frames(
                    chunk.load_start, chunk.load_end
                ):
                    duplicate_rows = int(
                        frame.duplicated(["ts_code", "trade_date"]).sum()
                    )
                    forbidden = _FORBIDDEN_COLUMNS.intersection(frame.columns)
                    frame_stats["duplicate_rows"] += duplicate_rows
                    frame_stats["forbidden_columns"].update(forbidden)
                    if duplicate_rows:
                        raise ValueError(
                            "factor validity input contains duplicate security-date keys"
                        )
                    if forbidden:
                        raise ValueError(
                            "factor validity input exposes forbidden target columns: "
                            + ", ".join(sorted(forbidden))
                        )
                    yield frame

            catalog, context = build_production_evaluation_context_from_batches(
                audited_frames(),
                dataset_version=(
                    f"{prepared.job.dataset_version}_{prepared.job_identity[:16]}"
                ),
                view=prepared.job.view,
                required_fields=requested,
                additional_field_specs=prepared.frame_loader.additional_field_specs,
                additional_dataset_versions=(
                    prepared.frame_loader.additional_dataset_versions
                ),
                initial_price_values=initial_price_values,
            )
            compiled = compile_expression(expression, operators, catalog)
            if compiled.lookback > prepared.job.evaluation.max_lookback:
                raise ValueError("factor lookback exceeds the production job limit")
            current_identity = (compiled.canonical, compiled.factor_id, compiled.lookback)
            if canonical is None:
                canonical, factor_id, lookback = current_identity
                output_type = compiled.output_type.value
                price_basis = (
                    compiled.price_basis.value if compiled.price_basis is not None else None
                )
                unit_lineage = compiled.unit_lineage
                field_semantics = {
                    name: _field_semantics(catalog.resolve(name))
                    for name in sorted(requested)
                }
            elif current_identity != (canonical, factor_id, lookback):
                raise RuntimeError("factor implementation identity changed between chunks")

            evaluator = BatchEvaluator(
                operators,
                catalog,
                functions,
                cache_max_bytes=prepared.job.evaluation.cache_mib * 1024 * 1024,
            )
            evaluated = evaluator.evaluate(expression, context)
            in_period = context.universe_mask.loc[
                context.universe_mask.index.to_series().between(
                    chunk.calculation_start, chunk.calculation_end
                )
            ]
            universe = in_period.to_numpy(dtype=bool)
            values = evaluated.values.reindex_like(in_period)
            factor_finite = np.isfinite(
                values.to_numpy(dtype=float, na_value=np.nan)
            )
            denominator = int(universe.sum())
            factor_finite_count = int((factor_finite & universe).sum())
            all_inputs = universe.copy()
            yearly_fields: dict[str, float] = {}
            for name in sorted(requested):
                panel = context.fields[name].reindex_like(in_period)
                finite = np.isfinite(
                    panel.to_numpy(dtype=float, na_value=np.nan)
                )
                count = int((finite & universe).sum())
                field_counts[name]["finite"] += count
                field_counts[name]["universe"] += denominator
                yearly_fields[name] = _ratio(count, denominator)
                all_inputs &= finite
            all_inputs_count = int(all_inputs.sum())
            totals["universe"] += denominator
            totals["all_inputs_finite"] += all_inputs_count
            totals["factor_finite"] += factor_finite_count
            totals["duplicate_rows"] += int(frame_stats["duplicate_rows"])
            totals["forbidden_columns"].update(frame_stats["forbidden_columns"])
            yearly.append(
                {
                    "year": chunk.year,
                    "universe_cells": denominator,
                    "factor_finite_cells": factor_finite_count,
                    "factor_coverage": _ratio(factor_finite_count, denominator),
                    "all_input_coverage": _ratio(all_inputs_count, denominator),
                    "field_coverage": yearly_fields,
                }
            )
            if position + 1 < len(prepared.chunks):
                initial_price_values = price_carry_state_before(
                    context,
                    before=prepared.chunks[position + 1].load_start,
                )
            print(
                f"[factor-validity] year={chunk.year} "
                f"coverage={_ratio(factor_finite_count, denominator):.6f} "
                f"universe_cells={denominator:,}",
                flush=True,
            )
        evaluation_seconds = perf_counter() - evaluation_start

        assert canonical is not None
        assert factor_id is not None
        assert lookback is not None
        causality, causality_warnings = CausalityAuditService().audit(
            path,
            expression,
            work_root=work_root,
        )
        denominator = int(totals["universe"])
        factor_coverage = _ratio(int(totals["factor_finite"]), denominator)
        input_coverage = _ratio(int(totals["all_inputs_finite"]), denominator)
        pit_clock_passed = all(
            _available_by_signal_close(value["available_at"])
            for value in field_semantics.values()
        )
        report = {
            "schema_version": "ashare-factor-validity-audit.v1",
            "status": (
                "factor_validity_verified"
                if denominator > 0
                and not totals["duplicate_rows"]
                and not totals["forbidden_columns"]
                and pit_clock_passed
                and bool(causality["passed"])
                else "factor_validity_failed"
            ),
            "return_data_read": False,
            "expression": {
                "requested": expression,
                "canonical": canonical,
                "factor_id": factor_id,
                "lookback": lookback,
                "output_type": output_type,
                "price_basis": price_basis,
                "unit_lineage": unit_lineage,
                "fields": sorted(requested),
            },
            "job": {
                "job_id": prepared.job.job_id,
                "job_identity": prepared.job_identity,
                "dataset_version": prepared.job.dataset_version,
                "asset_identities": prepared.inspected["asset_identities"],
                "plugins": prepared.inspected["plugins"],
                "universe_view": prepared.job.view,
                "evaluation_start": prepared.job.evaluation.start.isoformat(),
                "evaluation_end": prepared.job.evaluation.end.isoformat(),
            },
            "field_semantics": field_semantics,
            "checks": {
                "pit_clock": {
                    "passed": pit_clock_passed,
                    "decision_clock": "after_close_t_for_next_open",
                },
                "duplicate_keys": {
                    "passed": not totals["duplicate_rows"],
                    "duplicate_rows": totals["duplicate_rows"],
                },
                "target_separation": {
                    "passed": not totals["forbidden_columns"],
                    "forbidden_columns_seen": sorted(totals["forbidden_columns"]),
                },
                "causality": {
                    "passed": bool(causality["passed"]),
                    "certificate_identity": causality["certificate_identity"],
                    "certificate_path": causality["certificate_path"],
                    "cutoff": causality["cutoff"],
                    "compared_cells": causality["compared_cells"],
                    "mismatch_cells": causality["mismatch_cells"],
                },
            },
            "coverage": {
                "denominator": "finite cells inside the job's PIT universe mask",
                "universe_cells": denominator,
                "factor_finite_cells": totals["factor_finite"],
                "factor_coverage": factor_coverage,
                "all_input_coverage": input_coverage,
                "lookback_or_operator_additional_loss": max(
                    0.0, input_coverage - factor_coverage
                ),
                "field_coverage": {
                    name: {
                        "finite_cells": counts["finite"],
                        "universe_cells": counts["universe"],
                        "coverage": _ratio(counts["finite"], counts["universe"]),
                    }
                    for name, counts in field_counts.items()
                },
                "by_year": yearly,
            },
            "timings_seconds": {
                "validity_evaluation": evaluation_seconds,
                "causality": causality["timings_seconds"]["total"],
                "total": perf_counter() - total_start,
            },
            "limitations": [
                "No return, target, IC, Sharpe, turnover, or portfolio evidence was read.",
                "Prefix invariance audits expression execution, not upstream vendor truth.",
                "Coverage thresholds and research promotion remain caller responsibilities.",
            ],
        }
        identity_checks = dict(report["checks"])
        identity_checks["causality"] = {
            key: value
            for key, value in report["checks"]["causality"].items()
            if key != "certificate_path"
        }
        identity = _identity(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "return_data_read": report["return_data_read"],
                "expression": report["expression"],
                "job": report["job"],
                "field_semantics": report["field_semantics"],
                "checks": identity_checks,
                "coverage": report["coverage"],
            }
        )
        target = Path(work_root).resolve() / "factor_validity" / f"{identity}.json"
        report["audit_identity"] = identity
        report["artifact_path"] = str(target)
        _atomic_json(target, report)
        warnings = tuple(prepared.warnings) + tuple(causality_warnings)
        return report, warnings


def _field_semantics(spec: Any) -> dict[str, object]:
    return {
        "available_at": spec.available_at,
        "coverage_note": spec.coverage_note,
        "dataset_version": spec.dataset_version,
        "price_basis": spec.price_basis.value if spec.price_basis else None,
        "unit_lineage": spec.unit_lineage,
    }


def _available_by_signal_close(value: object) -> bool:
    text = str(value).lower()
    return not text.startswith("day+") or text.startswith("day+0_")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
