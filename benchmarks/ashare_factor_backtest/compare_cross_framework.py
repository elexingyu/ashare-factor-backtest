"""Validate numerical parity and produce a cross-framework summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.ashare_factor_backtest.contract import validate_result
from benchmarks.ashare_factor_backtest.cross_framework import read_outputs


def compare(
    ours_dir: Path,
    qlib_dir: Path,
    output_path: Path,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> dict[str, object]:
    ours = read_outputs(Path(ours_dir) / "outputs.npz")
    qlib = read_outputs(Path(qlib_dir) / "outputs.npz")
    if len(ours) != len(qlib):
        raise ValueError("frameworks produced different expression counts")
    masks_equal = True
    maximum_error = 0.0
    values_close = True
    expression_parity: list[dict[str, object]] = []
    for position, (expected, actual) in enumerate(zip(ours, qlib, strict=True)):
        if expected.shape != actual.shape:
            raise ValueError("frameworks produced different output shapes")
        expected_finite = np.isfinite(expected)
        actual_finite = np.isfinite(actual)
        expression_masks_equal = np.array_equal(expected_finite, actual_finite)
        masks_equal = masks_equal and expression_masks_equal
        common = expected_finite & actual_finite
        expression_maximum_error = 0.0
        expression_values_close = True
        if np.any(common):
            expression_maximum_error = float(
                np.max(np.abs(expected[common] - actual[common]))
            )
            maximum_error = max(maximum_error, expression_maximum_error)
            expression_values_close = bool(
                np.allclose(
                    expected[common],
                    actual[common],
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            )
            values_close = values_close and expression_values_close
        expression_parity.append(
            {
                "position": position,
                "finite_masks_equal": expression_masks_equal,
                "values_close": expression_values_close,
                "maximum_absolute_error": expression_maximum_error,
            }
        )
    comparable = masks_equal and values_close
    reason = (
        "same output shapes, finite masks and values within declared tolerance"
        if comparable
        else "output parity failed; speed ratio is not publishable"
    )

    result_paths = (Path(ours_dir) / "result.json", Path(qlib_dir) / "result.json")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    for payload, reference in zip(
        payloads,
        (payloads[1]["engine"]["name"], payloads[0]["engine"]["name"]),
        strict=True,
    ):
        payload["parity"] = {
            "reference_engine": reference,
            "comparable": comparable,
            "exact": maximum_error == 0.0 and masks_equal,
            "maximum_absolute_error": maximum_error,
            "reason": reason,
        }
        validate_result(payload)
    for path, payload in zip(result_paths, payloads, strict=True):
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    ours_median = median(payloads[0]["measurements"]["wall_seconds"])
    qlib_median = median(payloads[1]["measurements"]["wall_seconds"])
    summary = {
        "schema_version": "ashare-factor-cross-framework-summary.v1",
        "comparable": comparable,
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "expression_parity": expression_parity,
        "ours_median_wall_seconds": ours_median,
        "qlib_median_wall_seconds": qlib_median,
        "qlib_over_ours_wall_ratio": qlib_median / ours_median if comparable else None,
        "warning": "expression-layer result only; not a full backtest comparison",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-dir", type=Path, required=True)
    parser.add_argument("--qlib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    result = compare(arguments.ours_dir, arguments.qlib_dir, arguments.output)
    print(json.dumps(result, sort_keys=True))
