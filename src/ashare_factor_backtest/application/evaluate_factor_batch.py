"""Public shared-context evaluation for a small frozen expression batch."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from ashare_factor_backtest.application.evaluate_factor import (
    _load_or_compute_stage,
    _peak_rss_mib,
    _rolling_stage,
)
from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expressions_by_year,
)
from ashare_factor_backtest.evaluation.production_context import (
    PRODUCTION_FACTOR_EVALUATION_SEMANTICS,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    ExecutionCapturingFrameLoader,
)
from ashare_factor_backtest.evaluation.production_screen import screen_production_values
from ashare_factor_backtest.expression.parser import referenced_fields


_COVERAGE_REJECTION = "no discovery variant satisfies coverage and period gates"


class FactorBatchEvaluationService:
    """Evaluate formulas together without changing single-factor economics."""

    def evaluate(
        self,
        path: Path,
        expressions: tuple[str, ...],
        *,
        through: str = "rolling",
        work_root: Path,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        total_start = perf_counter()
        requested = tuple(str(expression).strip() for expression in expressions)
        if through not in {"screen", "rolling"}:
            raise ValueError(f"unsupported public factor stage: {through}")
        if not requested or any(not expression for expression in requested):
            raise ValueError("factor evaluation batch must not be empty")
        if len(requested) != len(set(requested)):
            raise ValueError("factor evaluation batch expressions must be unique")

        root = Path(work_root).resolve()
        prepared = ProductionJobService().prepare_batch(
            path,
            expressions=requested,
            validation_cache_root=root / "cache" / "plugin_validation",
        )
        job = prepared.job
        if job.research is None:
            raise ValueError("factor evaluation requires a research contract")
        required_fields = set().union(
            *(set(referenced_fields(expression)) for expression in requested)
        )
        capturing_loader = ExecutionCapturingFrameLoader(
            prepared.frame_loader,
            prepared.chunks,
            price_storage_dtype="float32",
            eligibility_column=job.view,
        )
        factor_start = perf_counter()
        evaluated_batch = evaluate_expressions_by_year(
            requested,
            chunks=prepared.chunks,
            frame_loader=capturing_loader,
            dataset_version=f"{job.dataset_version}_{prepared.job_identity[:16]}",
            view=job.view,
            cache_max_bytes=job.evaluation.cache_mib * 1024 * 1024,
            required_fields=required_fields,
            spill_to_disk=True,
        )
        factor_seconds = perf_counter() - factor_start
        execution = capturing_loader.execution_context()
        candidates: list[dict[str, object]] = []
        completed_count = 0
        rejected_count = 0

        for expression, evaluated in zip(requested, evaluated_batch, strict=True):
            if evaluated.lookback > job.evaluation.max_lookback:
                raise ValueError("expression lookback exceeds production job max_lookback")
            common: dict[str, object] = {
                "canonical": evaluated.canonical,
                "evidence_mode": job.research.evidence_mode,
                "factor_evaluation_semantics": PRODUCTION_FACTOR_EVALUATION_SEMANTICS,
                "factor_id": evaluated.factor_id,
                "job_id": job.job_id,
                "job_identity": prepared.job_identity,
                "lookback": evaluated.lookback,
                "return_data_read": True,
            }
            artifacts: dict[str, str] = {}
            try:
                screen_payload, screen_target, _ = _load_or_compute_stage(
                    work_root=root,
                    stage="screen",
                    upstream=(),
                    common=common,
                    compute=lambda values=evaluated.values: {
                        "screen": screen_production_values(
                            values,
                            execution,
                            policy=job.research.screen,
                        ),
                        "promotable": job.research.evidence_mode == "production",
                    },
                    memory_limit_mib=job.evaluation.memory_limit_mib,
                )
            except ValueError as error:
                if str(error) != _COVERAGE_REJECTION:
                    raise
                candidates.append(
                    {
                        **common,
                        "candidate_status": "coverage_or_period_gate_unsatisfied",
                        "expression": expression,
                        "reason": str(error),
                    }
                )
                rejected_count += 1
                continue
            artifacts["screen"] = str(screen_target)
            final_payload = screen_payload
            final_target = screen_target

            if through == "rolling":
                try:
                    rolling_payload, rolling_target, _ = _load_or_compute_stage(
                        work_root=root,
                        stage="rolling",
                        upstream=(str(screen_payload["artifact_identity"]),),
                        common=common,
                        compute=lambda values=evaluated.values: _rolling_stage(
                            values,
                            execution,
                            research=job.research,
                        ),
                        memory_limit_mib=job.evaluation.memory_limit_mib,
                    )
                except ValueError as error:
                    if str(error) != _COVERAGE_REJECTION:
                        raise
                    partial = dict(screen_payload)
                    partial.update(
                        {
                            "artifact_path": str(screen_target),
                            "candidate_status": (
                                "rolling_coverage_or_period_gate_unsatisfied"
                            ),
                            "expression": expression,
                            "gate": None,
                            "reason": str(error),
                            "rolling": None,
                            "screen": screen_payload["screen"],
                            "stage_artifacts": artifacts,
                        }
                    )
                    candidates.append(partial)
                    rejected_count += 1
                    continue
                artifacts["rolling"] = str(rolling_target)
                final_payload = rolling_payload
                final_target = rolling_target

            result = dict(final_payload)
            if through == "rolling":
                result["screen"] = screen_payload["screen"]
            result.update(
                {
                    "artifact_path": str(final_target),
                    "candidate_status": "completed",
                    "expression": expression,
                    "stage_artifacts": artifacts,
                }
            )
            candidates.append(result)
            completed_count += 1

        result = {
            "candidate_count": len(requested),
            "candidates": candidates,
            "completed_count": completed_count,
            "factor_seconds": factor_seconds,
            "job_identity": prepared.job_identity,
            "peak_rss_mib": _peak_rss_mib(),
            "rejected_count": rejected_count,
            "schema": "public_factor_batch_evaluation_v1",
            "through": through,
            "total_seconds": perf_counter() - total_start,
        }
        warnings = list(prepared.warnings)
        if job.research.evidence_mode == "engineering":
            warnings.append(
                "Engineering evidence mode is non-promotable and cannot enter admission."
            )
        return result, tuple(warnings)
