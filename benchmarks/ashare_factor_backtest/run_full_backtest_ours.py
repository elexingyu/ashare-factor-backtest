"""Run the public rolling research path as a complete backtest workload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any

from ashare_factor_backtest.application.evaluate_factor import FactorEvaluationService


ROOT = Path(__file__).resolve().parents[2]


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    repetitions: int = 1,
    warmup_repetitions: int = 0,
) -> dict[str, Any]:
    if repetitions <= 0 or warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    def evaluate(run_label: str) -> tuple[dict[str, object], tuple[str, ...]]:
        work_root = target / "work" / run_label
        shutil.rmtree(work_root, ignore_errors=True)
        return FactorEvaluationService().evaluate(
            Path(manifest["production_job_path"]),
            str(manifest["expression"]),
            through="rolling",
            work_root=work_root,
        )

    for position in range(warmup_repetitions):
        print(f"[full-backtest] warmup={position + 1}/{warmup_repetitions}", flush=True)
        evaluate(f"warmup_{position}")

    walls: list[float] = []
    cpus: list[float] = []
    stage_timings: list[dict[str, float]] = []
    final_result: dict[str, object] = {}
    warnings: tuple[str, ...] = ()
    for position in range(repetitions):
        print(f"[full-backtest] repetition={position + 1}/{repetitions}", flush=True)
        cpu_start = process_time()
        wall_start = perf_counter()
        final_result, warnings = evaluate(f"measured_{position}")
        walls.append(perf_counter() - wall_start)
        cpus.append(process_time() - cpu_start)
        stage_timings.append(
            {name: float(value) for name, value in final_result["timings_seconds"].items()}
        )

    fold_count = int(manifest["rolling_fold_count"])
    horizon_count = 3
    strategy_runs_per_screen = 2 * horizon_count + 3
    benchmark_runs_per_screen = horizon_count + 3
    payload = {
        "schema_version": "ashare-factor-full-backtest-result.v1",
        "benchmark_id": str(manifest["dataset_identity"])[:16],
        "engine": {
            "name": "ashare-factor-backtest-public-rolling",
            "commit": _git_revision(),
        },
        "environment": _environment(),
        "workload": {
            "dataset_identity": manifest["dataset_identity"],
            "date_count": manifest["date_count"],
            "security_count": manifest["security_count"],
            "expression": manifest["expression"],
            "rolling_fold_count": fold_count,
            "strategy_portfolio_runs": strategy_runs_per_screen * (fold_count + 1),
            "benchmark_portfolio_runs": benchmark_runs_per_screen * (fold_count + 1),
            "rank_ic_runs": 2 * (fold_count + 1),
            "includes": [
                "persistent_data_read",
                "expression_evaluation",
                "pit_universe",
                "st_suspension_limit_execution",
                "two_sided_costs",
                "nav_metrics_turnover",
                "rank_ic",
                "screen_selection",
                "rolling_evaluation",
                "atomic_json_artifacts",
            ],
        },
        "measurements": {
            "repetitions": repetitions,
            "warmup_repetitions": warmup_repetitions,
            "wall_seconds": walls,
            "cpu_seconds": cpus,
            "stage_timings_seconds": stage_timings,
            "peak_rss_mib": _peak_rss_mib(),
        },
        "evidence": {
            "screen": final_result.get("screen"),
            "rolling": final_result.get("rolling"),
            "gate": final_result.get("gate"),
            "warnings": list(warnings),
        },
    }
    (target / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _git_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "."], cwd=ROOT, text=True
    )
    return revision + ("+dirty" if status.strip() else "")


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.machine(),
        "logical_cpu_count": os.cpu_count() or 1,
        "memory_gib": _physical_memory_bytes() / (1024**3),
    }


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup-repetitions", type=int, default=0)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(
        arguments.manifest,
        arguments.output_dir,
        repetitions=arguments.repetitions,
        warmup_repetitions=arguments.warmup_repetitions,
    )
