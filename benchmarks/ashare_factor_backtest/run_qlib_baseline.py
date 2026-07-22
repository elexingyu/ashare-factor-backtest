"""Run Qlib expressions that have an explicit mapping to the public evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd

from benchmarks.ashare_factor_backtest.contract import validate_result
from benchmarks.ashare_factor_backtest.cross_framework import (
    benchmark_result,
    normalized_outputs,
    write_outputs,
)


def run(
    manifest_path: Path,
    qlib_data_dir: Path,
    qlib_commit: str,
    output_dir: Path,
    *,
    repetitions: int = 5,
    warmup_repetitions: int = 1,
    kernels: int = 1,
) -> dict[str, object]:
    if repetitions <= 0 or warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    if warmup_repetitions == 0 and repetitions != 1:
        raise ValueError("a cold benchmark must run once in a fresh process")
    if kernels <= 0:
        raise ValueError("kernels must be positive")
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    panel = pd.read_parquet(manifest["panel_path"], columns=["date", "symbol"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    symbols = tuple(sorted(panel["symbol"].unique()))
    expressions = tuple(str(item["qlib"]) for item in manifest["expressions"])
    qlib.init(
        provider_uri=str(Path(qlib_data_dir).resolve()),
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
        kernels=kernels,
    )

    def evaluate() -> tuple[np.ndarray, ...]:
        frame = D.features(
            list(symbols),
            list(expressions),
            start_time=str(manifest["start_date"]),
            end_time=str(manifest["end_date"]),
            freq="day",
            disk_cache=0,
        )
        outputs: list[np.ndarray] = []
        for expression in expressions:
            matrix = (
                frame[expression]
                .unstack(level="instrument")
                .reindex(index=dates, columns=symbols)
            )
            outputs.append(matrix.to_numpy(dtype=float, na_value=np.nan))
        return normalized_outputs(tuple(outputs), manifest["expressions"])

    for _ in range(warmup_repetitions):
        evaluate()
    walls: list[float] = []
    cpus: list[float] = []
    outputs: tuple[np.ndarray, ...] = ()
    for _ in range(repetitions):
        cpu_start = process_time()
        wall_start = perf_counter()
        outputs = evaluate()
        walls.append(perf_counter() - wall_start)
        cpus.append(process_time() - cpu_start)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_outputs(target / "outputs.npz", outputs)
    payload = benchmark_result(
        manifest=manifest,
        engine_name=f"microsoft-qlib-local-provider-kernels-{kernels}",
        engine_version=getattr(qlib, "__version__", "source-checkout"),
        engine_commit=qlib_commit,
        wall_seconds=walls,
        cpu_seconds=cpus,
        outputs=outputs,
        cache_state="warm" if warmup_repetitions else "cold",
    )
    validate_result(payload)
    (target / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qlib-data-dir", type=Path, required=True)
    parser.add_argument("--qlib-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--kernels", type=int, default=1)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(
        arguments.manifest,
        arguments.qlib_data_dir,
        arguments.qlib_commit,
        arguments.output_dir,
        repetitions=arguments.repetitions,
        warmup_repetitions=arguments.warmup_repetitions,
        kernels=arguments.kernels,
    )
