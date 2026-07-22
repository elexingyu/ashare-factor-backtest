"""Parity gate for the fixed-policy complete backtest benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from benchmarks.ashare_factor_backtest.common_backtest import write_json


TOLERANCES = {
    "gross_return_max_abs": 1e-10,
    "net_return_max_abs": 1e-6,
    "turnover_rate_max_abs": 1e-6,
    "cost_rate_max_abs": 1e-6,
    "total_return_abs": 1e-4,
    "sharpe_abs": 1e-3,
    "rank_ic_mean_abs": 1e-6,
}


def compare(ours_dir: Path, qlib_dir: Path, output_path: Path) -> dict[str, Any]:
    ours = _read_result(ours_dir)
    qlib = _read_result(qlib_dir)
    if ours["workload"] != qlib["workload"]:
        raise ValueError("common benchmark workload contracts differ")
    for key in ("python", "platform", "processor", "logical_cpu_count"):
        if ours["environment"][key] != qlib["environment"][key]:
            raise ValueError(f"common benchmark environments differ: {key}")
    ours_arrays = np.load(Path(ours_dir) / "outputs.npz")
    qlib_arrays = np.load(Path(qlib_dir) / "outputs.npz")
    dates_exact = np.array_equal(
        ours_arrays["return_dates_ns"], qlib_arrays["return_dates_ns"]
    )
    ours_factor = ours_arrays["factor"]
    qlib_factor = qlib_arrays["factor"]
    factor_masks_exact = np.array_equal(
        np.isfinite(ours_factor), np.isfinite(qlib_factor)
    )
    factor_values_exact = factor_masks_exact and np.array_equal(
        ours_factor[np.isfinite(ours_factor)], qlib_factor[np.isfinite(qlib_factor)]
    )
    selections_exact = (
        ours["evidence"]["target_selection_digest"]
        == qlib["evidence"]["target_selection_digest"]
    )
    gross_error = _max_abs(ours_arrays["gross_returns"], qlib_arrays["gross_returns"])
    net_error = _max_abs(ours_arrays["net_returns"], qlib_arrays["net_returns"])
    turnover_error = _max_abs(
        ours_arrays["turnover_rates"], qlib_arrays["turnover_rates"]
    )
    cost_error = _max_abs(ours_arrays["cost_rates"], qlib_arrays["cost_rates"])
    total_return_error = abs(
        float(ours["evidence"]["strategy_metrics"]["total_return"])
        - float(qlib["evidence"]["strategy_metrics"]["total_return"])
    )
    sharpe_error = abs(
        float(ours["evidence"]["strategy_metrics"]["sharpe"])
        - float(qlib["evidence"]["strategy_metrics"]["sharpe"])
    )
    rank_ic_error = abs(
        float(ours["evidence"]["rank_ic"]["rank_ic_mean"])
        - float(qlib["evidence"]["rank_ic"]["rank_ic_mean"])
    )
    checks = {
        "dates_exact": dates_exact,
        "factor_finite_masks_exact": factor_masks_exact,
        "factor_finite_values_exact": factor_values_exact,
        "target_selections_exact": selections_exact,
        "gross_return_within_tolerance": gross_error <= TOLERANCES["gross_return_max_abs"],
        "net_return_within_tolerance": net_error <= TOLERANCES["net_return_max_abs"],
        "turnover_rate_within_tolerance": turnover_error
        <= TOLERANCES["turnover_rate_max_abs"],
        "cost_rate_within_tolerance": cost_error <= TOLERANCES["cost_rate_max_abs"],
        "total_return_within_tolerance": total_return_error <= TOLERANCES["total_return_abs"],
        "sharpe_within_tolerance": sharpe_error <= TOLERANCES["sharpe_abs"],
        "rank_ic_within_tolerance": rank_ic_error <= TOLERANCES["rank_ic_mean_abs"],
    }
    comparable = all(checks.values())
    ours_median = median(float(value) for value in ours["measurements"]["wall_seconds"])
    qlib_median = median(float(value) for value in qlib["measurements"]["wall_seconds"])
    payload = {
        "schema_version": "ashare-factor-common-backtest-comparison.v1",
        "comparable": comparable,
        "checks": checks,
        "tolerances": TOLERANCES,
        "errors": {
            "gross_return_max_abs": gross_error,
            "net_return_max_abs": net_error,
            "turnover_rate_max_abs": turnover_error,
            "cost_rate_max_abs": cost_error,
            "total_return_abs": total_return_error,
            "sharpe_abs": sharpe_error,
            "rank_ic_mean_abs": rank_ic_error,
        },
        "first_mismatches": {
            name: mismatch
            for name, mismatch in (
                (
                    "gross_returns",
                    _first_mismatch(
                        ours_arrays["gross_returns"],
                        qlib_arrays["gross_returns"],
                        TOLERANCES["gross_return_max_abs"],
                    ),
                ),
                (
                    "net_returns",
                    _first_mismatch(
                        ours_arrays["net_returns"],
                        qlib_arrays["net_returns"],
                        TOLERANCES["net_return_max_abs"],
                    ),
                ),
                (
                    "turnover_rates",
                    _first_mismatch(
                        ours_arrays["turnover_rates"],
                        qlib_arrays["turnover_rates"],
                        TOLERANCES["turnover_rate_max_abs"],
                    ),
                ),
                (
                    "cost_rates",
                    _first_mismatch(
                        ours_arrays["cost_rates"],
                        qlib_arrays["cost_rates"],
                        TOLERANCES["cost_rate_max_abs"],
                    ),
                ),
            )
            if mismatch is not None
        },
        "ours_median_wall_seconds": ours_median,
        "qlib_median_wall_seconds": qlib_median,
        "qlib_over_ours_wall_ratio": qlib_median / ours_median if comparable else None,
        "speed_claim_allowed": comparable,
    }
    write_json(output_path, payload)
    return payload


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads((Path(path) / "result.json").read_text(encoding="utf-8"))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _first_mismatch(
    left: np.ndarray, right: np.ndarray, tolerance: float
) -> dict[str, float | int | str] | None:
    if left.shape != right.shape:
        return {"position": 0, "reason": f"shape {left.shape} != {right.shape}"}
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    errors = np.abs(left_values - right_values)
    indexes = np.flatnonzero(errors > tolerance)
    if not len(indexes):
        return None
    position = int(indexes[0])
    return {
        "position": position,
        "ours": float(left_values.flat[position]),
        "qlib": float(right_values.flat[position]),
        "absolute_error": float(errors.flat[position]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-dir", type=Path, required=True)
    parser.add_argument("--qlib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    print(
        json.dumps(
            compare(arguments.ours_dir, arguments.qlib_dir, arguments.output),
            sort_keys=True,
        )
    )
