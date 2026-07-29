"""Public prefix-invariance audit for one production factor expression."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import pandas as pd

from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.expression.causality import (
    compare_prefix_results,
)
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


class CausalityAuditService:
    """Check one expression without running portfolio simulation."""

    def audit(
        self,
        path: Path,
        expression: str,
        *,
        work_root: Path,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        total_start = perf_counter()
        prepare_start = perf_counter()
        prepared = ProductionJobService().prepare(
            path,
            expression=expression,
            validation_cache_root=Path(work_root) / "cache" / "plugin_validation",
        )
        if len(prepared.calculation_dates) < 2:
            raise ValueError("causality audit requires at least two evaluation dates")
        cutoff = pd.Timestamp(
            prepared.calculation_dates[(len(prepared.calculation_dates) - 1) // 2]
        )
        prepare_seconds = perf_counter() - prepare_start
        common = {
            "frame_loader": prepared.frame_loader,
            "dataset_version": (
                f"{prepared.job.dataset_version}_{prepared.job_identity[:16]}"
            ),
            "view": prepared.job.view,
            "cache_max_bytes": prepared.job.evaluation.cache_mib * 1024 * 1024,
            "required_fields": set(referenced_fields(expression)),
            "spill_to_disk": True,
        }

        full_start = perf_counter()
        full = evaluate_expression_by_year(
            expression,
            chunks=prepared.chunks,
            **common,
        )
        full_seconds = perf_counter() - full_start

        prefix_start = perf_counter()
        prefix = evaluate_expression_by_year(
            expression,
            chunks=_truncate_chunks(prepared.chunks, cutoff),
            **common,
        )
        prefix_seconds = perf_counter() - prefix_start

        compare_start = perf_counter()
        report = compare_prefix_results(
            full,
            prefix,
            cutoff=cutoff,
        )
        compare_seconds = perf_counter() - compare_start
        certificate = {
            "schema_version": "causality-certificate.v1",
            "status": (
                "prefix_invariance_verified"
                if report.passed
                else "causality_violation_detected"
            ),
            "job_id": prepared.job.job_id,
            "job_identity": prepared.job_identity,
            **report.as_dict(),
        }
        identity = hashlib.sha256(
            json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        target = (
            Path(work_root).resolve()
            / "causality"
            / f"{identity}.json"
        )
        certificate["certificate_identity"] = identity
        _atomic_json(target, certificate)
        result = {
            **certificate,
            "certificate_path": str(target),
            "timings_seconds": {
                "prepare": prepare_seconds,
                "full_factor": full_seconds,
                "prefix_factor": prefix_seconds,
                "compare": compare_seconds,
                "total": perf_counter() - total_start,
            },
        }
        warnings = tuple(prepared.warnings) + (
            "Prefix invariance checks expression execution, not upstream data truth.",
        )
        return result, warnings


def _truncate_chunks(
    chunks: tuple[YearChunk, ...],
    cutoff: pd.Timestamp,
) -> tuple[YearChunk, ...]:
    truncated: list[YearChunk] = []
    for chunk in chunks:
        if chunk.calculation_start > cutoff:
            break
        calculation_end = min(chunk.calculation_end, cutoff)
        truncated.append(
            YearChunk(
                year=chunk.year,
                calculation_start=chunk.calculation_start,
                calculation_end=calculation_end,
                load_start=chunk.load_start,
                load_end=min(chunk.load_end, cutoff),
                max_lookback=chunk.max_lookback,
                forward_tail=0,
                partition_axis=chunk.partition_axis,
            )
        )
    if not truncated:
        raise ValueError("causality cutoff precedes every evaluation chunk")
    return tuple(truncated)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
