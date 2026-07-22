"""Run one fixed-policy complete backtest with the public engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter, process_time
from typing import Any

import numpy as np
import pandas as pd

from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.evaluation.production_evaluator import (
    evaluate_production_staggered_long_only_context,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
)
from ashare_factor_backtest.evaluation.production_rank_ic import evaluate_production_rank_ic
from ashare_factor_backtest.expression.evaluator import BatchEvaluator, EvaluationContext
from ashare_factor_backtest.expression.operators.registry import (
    build_production_operator_catalog,
)
from ashare_factor_backtest.expression.production_fields import (
    build_production_field_catalog,
)
from benchmarks.ashare_factor_backtest.common_backtest import (
    array_digest,
    environment,
    target_selection_digest,
    write_arrays,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]


def run(
    manifest_path: Path,
    output_dir: Path,
    *,
    repetitions: int = 5,
    warmup_repetitions: int = 1,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    contract = manifest["common_contract"]
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    operators, functions = build_production_operator_catalog()
    fields = build_production_field_catalog()

    def evaluate() -> tuple[dict[str, Any], dict[str, float], dict[str, np.ndarray]]:
        total_start = perf_counter()
        data_start = perf_counter()
        panel = pd.read_parquet(manifest["panel_path"])
        close = (
            panel.pivot(index="date", columns="symbol", values="close")
            .sort_index()
            .sort_index(axis=1)
        )
        open_price = panel.pivot(index="date", columns="symbol", values="open").reindex_like(close)
        data_seconds = perf_counter() - data_start

        factor_start = perf_counter()
        source_fields = {name: close for name in ("high", "low", "close", "volume", "amount")}
        source_fields["open"] = open_price
        mask = pd.DataFrame(True, index=close.index, columns=close.columns)
        context = EvaluationContext(
            fields=source_fields,
            dataset_versions={name: str(manifest["dataset_identity"]) for name in source_fields},
            universe_policy="all_benchmark_symbols",
            date_range=(str(manifest["start_date"]), str(manifest["end_date"])),
            universe_size=pd.Series(len(close.columns), index=close.index, dtype=float),
            universe_mask=mask,
            evaluation_price_basis=PriceBasis.HFQ_PIT,
        )
        evaluated, rejected = BatchEvaluator(
            operators, fields, functions, cache_max_bytes=2 << 30
        ).evaluate_many((str(manifest["expression"]),), context)
        if rejected:
            raise RuntimeError(f"common benchmark expression rejected: {rejected}")
        factor = evaluated[0].values
        factor_seconds = perf_counter() - factor_start

        execution_start = perf_counter()
        effective_end = pd.Timestamp(open_price.index[-2])
        shape = open_price.shape
        execution = ProductionExecutionContext(
            dates=pd.DatetimeIndex(open_price.index),
            codes=pd.Index(open_price.columns),
            valuation_open=open_price.to_numpy(dtype=np.float64),
            buyable=np.ones(shape, dtype=bool),
            sellable=np.ones(shape, dtype=bool),
            signal_eligible=np.ones(shape, dtype=bool),
        )
        portfolio = evaluate_production_staggered_long_only_context(
            factor,
            execution,
            direction=str(contract["direction"]),
            horizon=int(contract["horizon"]),
            buy_cost=float(contract["buy_cost"]),
            sell_cost=float(contract["sell_cost"]),
            decision_start=str(manifest["start_date"]),
            decision_end=effective_end.date().isoformat(),
            top_fraction=float(contract["top_fraction"]),
            record_events=True,
        )
        rank_ic = evaluate_production_rank_ic(
            factor,
            execution,
            horizon=int(contract["horizon"]),
            signal_start=str(manifest["start_date"]),
            signal_end=effective_end.date().isoformat(),
        )
        execution_seconds = perf_counter() - execution_start
        gross_returns = np.asarray(portfolio.strategy.gross_returns)
        net_returns = np.asarray(portfolio.strategy.net_returns)
        cost_rates = gross_returns - net_returns
        turnover_by_date = {}
        buy_cost = float(contract["buy_cost"])
        for event in portfolio.strategy.order_events:
            cash_spend_notional = event.sell_notional + event.buy_notional
            if event.actual_turnover > 0.0 and cash_spend_notional > 0.0:
                pretrade_value = cash_spend_notional / event.actual_turnover
                traded_security_value = (
                    event.sell_notional + event.buy_notional * (1.0 - buy_cost)
                )
                turnover_by_date[event.entry_date.value] = (
                    traded_security_value / pretrade_value
                )
            else:
                turnover_by_date[event.entry_date.value] = 0.0
        turnover_rates = np.asarray(
            [
                turnover_by_date.get(value.value, 0.0) * (1.0 + gross_returns[position])
                for position, value in enumerate(portfolio.strategy.return_dates)
            ],
            dtype=np.float64,
        )
        sell_cost = float(contract["sell_cost"])
        if len(turnover_rates) and sell_cost:
            turnover_rates[-1] = cost_rates[-1] / sell_cost
        arrays = {
            "factor": factor.to_numpy(dtype=np.float64),
            "return_dates_ns": np.asarray(
                [value.value for value in portfolio.strategy.return_dates], dtype=np.int64
            ),
            "gross_returns": gross_returns,
            "net_returns": net_returns,
            "turnover_rates": turnover_rates,
            "cost_rates": cost_rates,
            "benchmark_returns": np.asarray(portfolio.benchmark.net_returns),
            "excess_returns": np.asarray(portfolio.excess_returns),
        }
        evidence = {
            "strategy_metrics": portfolio.strategy.metrics,
            "benchmark_metrics": portfolio.benchmark.metrics,
            "excess_metrics": portfolio.excess_metrics,
            "average_turnover": float(turnover_rates.mean()),
            "average_cash_spend_turnover": portfolio.strategy.average_turnover,
            "total_cost": portfolio.strategy.total_cost,
            "rank_ic": rank_ic,
            "target_selection_digest": target_selection_digest(
                factor.loc[:effective_end], top_fraction=float(contract["top_fraction"])
            ),
            "execution_adapter": "target_delta_continuous_value_v2",
            "turnover_coordinate": "traded_security_value_over_previous_portfolio_value",
            "effective_end": effective_end.date().isoformat(),
        }
        timings = {
            "data": data_seconds,
            "factor": factor_seconds,
            "portfolio_ic": execution_seconds,
            "total_before_artifact": perf_counter() - total_start,
        }
        return evidence, timings, arrays

    for _ in range(warmup_repetitions):
        evaluate()
    walls: list[float] = []
    cpus: list[float] = []
    stages: list[dict[str, float]] = []
    evidence: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for position in range(repetitions):
        print(f"[common-backtest-ours] repetition={position + 1}/{repetitions}", flush=True)
        cpu_start = process_time()
        wall_start = perf_counter()
        evidence, timings, arrays = evaluate()
        write_arrays(target / f"outputs_{position}.npz", **arrays)
        walls.append(perf_counter() - wall_start)
        cpus.append(process_time() - cpu_start)
        stages.append(timings)
    write_arrays(target / "outputs.npz", **arrays)
    payload = {
        "schema_version": "ashare-factor-common-backtest-result.v1",
        "engine": {"name": "ashare-factor-backtest", "commit": _git_revision()},
        "environment": environment(),
        "workload": {
            "dataset_identity": manifest["dataset_identity"],
            "date_count": manifest["date_count"],
            "security_count": manifest["security_count"],
            "contract": contract,
        },
        "measurements": {
            "repetitions": repetitions,
            "warmup_repetitions": warmup_repetitions,
            "wall_seconds": walls,
            "cpu_seconds": cpus,
            "stage_timings_seconds": stages,
            "peak_rss_mib": _peak_rss_mib(),
            "output_digest": array_digest(tuple(arrays.values())),
        },
        "evidence": evidence,
    }
    write_json(target / "result.json", payload)
    return payload


def _git_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=ROOT, text=True)
    return revision + ("+dirty" if status.strip() else "")


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


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
