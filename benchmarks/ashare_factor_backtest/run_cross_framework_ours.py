"""Run the public expression evaluator on the cross-framework panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter, process_time

import numpy as np
import pandas as pd

from benchmarks.ashare_factor_backtest.contract import validate_result
from benchmarks.ashare_factor_backtest.cross_framework import (
    benchmark_result,
    normalized_outputs,
    write_outputs,
)
from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.evaluator import BatchEvaluator, EvaluationContext
from ashare_factor_backtest.expression.operators.registry import build_production_operator_catalog
from ashare_factor_backtest.expression.production_fields import build_production_field_catalog


ROOT = Path(__file__).resolve().parents[2]


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    repetitions: int = 5,
    warmup_repetitions: int = 1,
) -> dict[str, object]:
    if repetitions <= 0 or warmup_repetitions < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    if warmup_repetitions == 0 and repetitions != 1:
        raise ValueError("a cold benchmark must run once in a fresh process")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    operators, functions = build_production_operator_catalog()
    field_catalog = build_production_field_catalog()
    expressions = tuple(str(item["ours"]) for item in manifest["expressions"])

    def evaluate() -> tuple[np.ndarray, ...]:
        panel = pd.read_parquet(manifest["panel_path"])
        close = (
            panel.pivot(index="date", columns="symbol", values="close")
            .sort_index()
            .sort_index(axis=1)
        )
        fields = {
            name: close
            for name in ("open", "high", "low", "close", "volume", "amount")
        }
        context = EvaluationContext(
            fields=fields,
            dataset_versions={
                name: str(manifest["dataset_identity"]) for name in fields
            },
            universe_policy="all_benchmark_symbols",
            date_range=(str(manifest["start_date"]), str(manifest["end_date"])),
            universe_size=pd.Series(len(close.columns), index=close.index, dtype=float),
            universe_mask=pd.DataFrame(True, index=close.index, columns=close.columns),
            evaluation_price_basis=PriceBasis.HFQ_PIT,
        )
        evaluator = BatchEvaluator(
            operators,
            field_catalog,
            functions,
            cache_max_bytes=2 << 30,
        )
        results, rejected = evaluator.evaluate_many(expressions, context)
        if rejected:
            raise RuntimeError(f"benchmark expressions were rejected: {rejected}")
        return normalized_outputs(
            tuple(
                result.values.to_numpy(dtype=float, na_value=np.nan)
                for result in results
            ),
            manifest["expressions"],
        )

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
        engine_name="ashare-factor-backtest-public-evaluator",
        engine_version=operators.version,
        engine_commit=_git_revision(),
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


def _git_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "."],
        cwd=ROOT,
        text=True,
    )
    return revision + ("+dirty" if status.strip() else "")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    run(
        arguments.manifest,
        arguments.output_dir,
        repetitions=arguments.repetitions,
        warmup_repetitions=arguments.warmup_repetitions,
    )
