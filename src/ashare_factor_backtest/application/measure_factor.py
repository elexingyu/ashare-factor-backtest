"""Public service for one immutable daily factor measurement artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.evaluation.daily_factor_measurement import (
    MEASUREMENT_ENGINE_VERSION,
    measure_daily_factor,
)
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    ExecutionCapturingFrameLoader,
)
from ashare_factor_backtest.expression.parser import referenced_fields


DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 20, 60)
DEFAULT_ROLLING_WINDOWS = (252, 504)


class FactorMeasurementService:
    """Compute facts once without selecting, gating, or promoting a factor."""

    def measure(
        self,
        path: Path,
        expression: str,
        *,
        direction: str,
        work_root: Path,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        total_start = perf_counter()
        prepared = ProductionJobService().prepare(
            path,
            expression=expression,
            validation_cache_root=Path(work_root) / "cache" / "plugin_validation",
        )
        capturing_loader = ExecutionCapturingFrameLoader(
            prepared.frame_loader,
            prepared.chunks,
            price_storage_dtype="float32",
            eligibility_column=prepared.job.view,
        )
        evaluated = evaluate_expression_by_year(
            expression,
            chunks=prepared.chunks,
            frame_loader=capturing_loader,
            dataset_version=(
                f"{prepared.job.dataset_version}_{prepared.job_identity[:16]}"
            ),
            view=prepared.job.view,
            cache_max_bytes=prepared.job.evaluation.cache_mib * 1024 * 1024,
            required_fields=set(referenced_fields(expression)),
            spill_to_disk=True,
        )
        if evaluated.lookback > prepared.job.evaluation.max_lookback:
            raise ValueError("expression lookback exceeds production job max_lookback")
        fixed_horizons = tuple(int(value) for value in horizons)
        fixed_rolling = tuple(int(value) for value in rolling_windows)
        identity_payload = {
            "schema_version": "ashare-factor-daily-measurement-identity.v1",
            "engine_version": MEASUREMENT_ENGINE_VERSION,
            "canonical": evaluated.canonical,
            "factor_id": evaluated.factor_id,
            "job_identity": prepared.job_identity,
            "direction": direction,
            "horizons": list(fixed_horizons),
            "rolling_windows": list(fixed_rolling),
        }
        identity = _identity(identity_payload)
        root = Path(work_root).resolve() / "daily_measurement"
        report_path = root / f"{identity}.json"
        if report_path.is_file():
            cached = _json_object(report_path)
            _verify_cached(cached, identity)
            cached["reused"] = True
            cached["timings_seconds"] = {"total": perf_counter() - total_start}
            return cached, tuple(prepared.warnings)

        context = capturing_loader.execution_context()
        summary, trace, memberships = measure_daily_factor(
            evaluated.values,
            context,
            direction=direction,
            horizons=fixed_horizons,
            rolling_windows=fixed_rolling,
        )
        root.mkdir(parents=True, exist_ok=True)
        trace_path = root / f"{identity}.daily.parquet"
        membership_path = root / f"{identity}.top20.parquet"
        _atomic_parquet(trace_path, trace)
        _atomic_parquet(membership_path, memberships)
        report: dict[str, object] = {
            "schema_version": "ashare-factor-daily-measurement.v1",
            "measurement_identity": identity,
            "measurement_contract": identity_payload,
            "return_data_read": True,
            "promotion_authority": False,
            "expression": {
                "requested": expression,
                "canonical": evaluated.canonical,
                "factor_id": evaluated.factor_id,
                "lookback": evaluated.lookback,
            },
            "job": {
                "job_id": prepared.job.job_id,
                "job_identity": prepared.job_identity,
                "dataset_version": prepared.job.dataset_version,
                "universe_view": prepared.job.view,
                "evaluation_start": prepared.job.evaluation.start.isoformat(),
                "evaluation_end": prepared.job.evaluation.end.isoformat(),
                "evidence_mode": (
                    prepared.job.research.evidence_mode
                    if prepared.job.research is not None
                    else "unspecified"
                ),
                "symbol_cap": prepared.job.evaluation.symbol_cap,
            },
            "summary": summary,
            "artifacts": {
                "daily_trace": {
                    "path": str(trace_path),
                    "sha256": _file_sha256(trace_path),
                    "rows": int(len(trace)),
                },
                "top20_membership": {
                    "path": str(membership_path),
                    "sha256": _file_sha256(membership_path),
                    "rows": int(len(memberships)),
                },
            },
            "limitations": [
                "This is a zero-net unit-gross factor measurement, not an executable account.",
                "No cost, capacity, portfolio increment, library correlation, or admission gate is applied.",
                "Direction and diagnostic horizons are caller-frozen and are not selected from returns.",
            ],
            "reused": False,
            "timings_seconds": {"total": perf_counter() - total_start},
            "artifact_path": str(report_path),
        }
        _atomic_json(report_path, report)
        return report, tuple(prepared.warnings)


def _verify_cached(payload: Mapping[str, Any], identity: str) -> None:
    if payload.get("measurement_identity") != identity:
        raise ValueError("cached daily measurement identity drift")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("cached daily measurement has no artifact index")
    for label in ("daily_trace", "top20_membership"):
        item = artifacts.get(label)
        if not isinstance(item, Mapping):
            raise ValueError(f"cached daily measurement lacks {label}")
        path = Path(str(item.get("path", ""))).resolve(strict=True)
        if _file_sha256(path) != item.get("sha256"):
            raise ValueError(f"cached daily measurement {label} hash drift")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


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


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("daily measurement artifact must be an object")
    return value


def _identity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
