"""Retention gate for production-path performance optimizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks.ashare_factor_backtest.common_backtest import write_json


_ENVIRONMENT_KEYS = (
    "python",
    "platform",
    "processor",
    "logical_cpu_count",
    "memory_gib",
)


def compare(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    maximum_median_seconds: float,
    maximum_peak_rss_mib: float,
) -> dict[str, Any]:
    if maximum_median_seconds <= 0 or maximum_peak_rss_mib <= 0:
        raise ValueError("full research performance limits must be positive")
    baseline = _read_result(baseline_path)
    candidate = _read_result(candidate_path)
    baseline_walls = _wall_seconds(baseline)
    candidate_walls = _wall_seconds(candidate)
    baseline_median = median(baseline_walls)
    candidate_median = median(candidate_walls)
    candidate_peak = float(candidate["measurements"]["peak_rss_mib"])
    environment_exact = all(
        baseline["environment"].get(key) == candidate["environment"].get(key)
        for key in _ENVIRONMENT_KEYS
    )
    checks = {
        "environment_exact": environment_exact,
        "workload_exact": baseline["workload"] == candidate["workload"],
        "semantic_evidence_exact": baseline["evidence"] == candidate["evidence"],
        "median_wall_within_limit": candidate_median <= maximum_median_seconds,
        "peak_rss_within_limit": candidate_peak <= maximum_peak_rss_mib,
    }
    payload = {
        "schema_version": "ashare-factor-full-research-comparison.v1",
        **checks,
        "retention_allowed": all(checks.values()),
        "limits": {
            "maximum_median_seconds": maximum_median_seconds,
            "maximum_peak_rss_mib": maximum_peak_rss_mib,
        },
        "baseline": {
            "commit": baseline["engine"]["commit"],
            "median_wall_seconds": baseline_median,
            "peak_rss_mib": float(baseline["measurements"]["peak_rss_mib"]),
        },
        "candidate": {
            "commit": candidate["engine"]["commit"],
            "median_wall_seconds": candidate_median,
            "peak_rss_mib": candidate_peak,
        },
        "wall_time_improvement_fraction": 1.0
        - candidate_median / baseline_median,
    }
    write_json(output_path, payload)
    return payload


def _read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("engine", "environment", "workload", "measurements", "evidence"):
        if key not in payload:
            raise ValueError(f"full research result is missing {key}")
    return payload


def _wall_seconds(payload: dict[str, Any]) -> list[float]:
    values = [float(value) for value in payload["measurements"].get("wall_seconds", [])]
    if not values or any(value <= 0 for value in values):
        raise ValueError("full research wall times must be positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-median-seconds", type=float, required=True)
    parser.add_argument("--maximum-peak-rss-mib", type=float, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    print(
        json.dumps(
            compare(
                arguments.baseline,
                arguments.candidate,
                arguments.output,
                maximum_median_seconds=arguments.maximum_median_seconds,
                maximum_peak_rss_mib=arguments.maximum_peak_rss_mib,
            ),
            sort_keys=True,
        )
    )
