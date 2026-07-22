"""Public single-factor screen and rolling evaluation service."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from typing import Any

from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_context import (
    PRODUCTION_FACTOR_EVALUATION_SEMANTICS,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    build_chunked_production_execution_context,
)
from ashare_factor_backtest.evaluation.production_rolling import (
    audit_production_rolling,
    evaluate_production_rolling,
)
from ashare_factor_backtest.evaluation.production_screen import screen_production_values


class FactorEvaluationService:
    """Evaluate one expression without candidate search or statistical admission."""

    def screen(
        self,
        path: Path,
        expression: str,
        *,
        work_root: Path,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        return self.evaluate(path, expression, through="screen", work_root=work_root)

    def evaluate(
        self,
        path: Path,
        expression: str,
        *,
        through: str = "rolling",
        work_root: Path,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        if through not in {"screen", "rolling"}:
            raise ValueError(f"unsupported public factor stage: {through}")
        prepared = ProductionJobService().prepare(
            path,
            expression=expression,
            validation_cache_root=Path(work_root) / "cache" / "plugin_validation",
        )
        job = prepared.job
        if job.research is None:
            raise ValueError("factor evaluation requires a research contract")
        evaluated = evaluate_expression_by_year(
            expression,
            chunks=prepared.chunks,
            frame_loader=prepared.frame_loader,
            dataset_version=f"{job.dataset_version}_{prepared.job_identity[:16]}",
            view=job.view,
            cache_max_bytes=job.evaluation.cache_mib * 1024 * 1024,
            required_fields=set(referenced_fields(expression)),
            spill_to_disk=True,
        )
        if evaluated.lookback > job.evaluation.max_lookback:
            raise ValueError("expression lookback exceeds production job max_lookback")
        execution = build_chunked_production_execution_context(
            chunks=prepared.chunks,
            frame_loader=prepared.frame_loader,
            price_storage_dtype="float32",
            eligibility_column=job.view,
        )
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
        artifact_paths: dict[str, str] = {}
        reused: list[str] = []

        screen_payload, screen_target, was_reused = _load_or_compute_stage(
            work_root=work_root,
            stage="screen",
            upstream=(),
            common=common,
            compute=lambda: {
                "screen": screen_production_values(
                    evaluated.values,
                    execution,
                    policy=job.research.screen,
                ),
                "promotable": job.research.evidence_mode == "production",
            },
            memory_limit_mib=job.evaluation.memory_limit_mib,
        )
        artifact_paths["screen"] = str(screen_target)
        if was_reused:
            reused.append("screen")
        final_payload = screen_payload

        if through == "rolling":
            rolling_payload, rolling_target, was_reused = _load_or_compute_stage(
                work_root=work_root,
                stage="rolling",
                upstream=(str(screen_payload["artifact_identity"]),),
                common=common,
                compute=lambda: _rolling_stage(
                    evaluated.values,
                    execution,
                    research=job.research,
                ),
                memory_limit_mib=job.evaluation.memory_limit_mib,
            )
            artifact_paths["rolling"] = str(rolling_target)
            if was_reused:
                reused.append("rolling")
            final_payload = rolling_payload

        result = dict(final_payload)
        result["artifact_path"] = artifact_paths[through]
        result["stage_artifacts"] = artifact_paths
        result["reused_stages"] = reused
        warnings = list(prepared.warnings)
        if job.research.evidence_mode == "engineering":
            warnings.append(
                "Engineering evidence mode is non-promotable and cannot enter admission."
            )
        return result, tuple(warnings)


def _rolling_stage(values: Any, execution: Any, *, research: Any) -> dict[str, object]:
    rolling = evaluate_production_rolling(
        values,
        execution,
        policy=research.rolling,
    )
    gate = audit_production_rolling(rolling, research.rolling_gate)
    return {
        "gate": gate,
        "promotable": (
            research.evidence_mode == "production"
            and gate["status"] == "rolling_survivor"
        ),
        "rolling": rolling,
    }


def _load_or_compute_stage(
    *,
    work_root: Path,
    stage: str,
    upstream: tuple[str, ...],
    common: dict[str, object],
    compute: Any,
    memory_limit_mib: float,
) -> tuple[dict[str, object], Path, bool]:
    identity = _artifact_identity(
        job_identity=str(common["job_identity"]),
        factor_id=str(common["factor_id"]),
        stage=stage,
        upstream=upstream,
    )
    target = Path(work_root).resolve() / "stages" / stage / f"{identity}.json"
    existing = _read_valid_stage(
        target,
        identity=identity,
        stage=stage,
        common=common,
        upstream=upstream,
    )
    if existing is not None:
        return existing, target, True
    stage_result = compute()
    peak = _peak_rss_mib()
    if peak > memory_limit_mib:
        raise MemoryError(f"factor {stage} exceeded memory limit: {peak:.1f} MiB")
    payload = {
        **common,
        **stage_result,
        "artifact_identity": identity,
        "peak_rss_mib": peak,
        "stage": stage,
        "upstream_artifact_identities": list(upstream),
    }
    payload["content_digest"] = _content_digest(payload)
    _atomic_json(target, payload)
    return payload, target, False


def _read_valid_stage(
    path: Path,
    *,
    identity: str,
    stage: str,
    common: dict[str, object],
    upstream: tuple[str, ...],
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    digest = payload.get("content_digest")
    if not isinstance(digest, str) or digest != _content_digest(
        {key: value for key, value in payload.items() if key != "content_digest"}
    ):
        return None
    expected = {
        "artifact_identity": identity,
        "canonical": common["canonical"],
        "factor_evaluation_semantics": common["factor_evaluation_semantics"],
        "factor_id": common["factor_id"],
        "job_identity": common["job_identity"],
        "stage": stage,
        "upstream_artifact_identities": list(upstream),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    return payload


def _artifact_identity(
    *,
    job_identity: str,
    factor_id: str,
    stage: str,
    upstream: tuple[str, ...] = (),
) -> str:
    payload = {
        "factor_evaluation_semantics": PRODUCTION_FACTOR_EVALUATION_SEMANTICS,
        "factor_id": factor_id,
        "job_identity": job_identity,
        "schema": "production-job-stage.v2",
        "stage": stage,
    }
    if upstream:
        payload["upstream_artifact_identities"] = list(upstream)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _content_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
