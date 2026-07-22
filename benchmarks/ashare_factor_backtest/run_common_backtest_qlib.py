"""Run the fixed-policy complete backtest on Qlib's native backtest engine."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.ashare_factor_backtest.common_backtest import (
    array_digest,
    environment,
    target_selection_digest,
    write_arrays,
    write_json,
)
from benchmarks.ashare_factor_backtest.net_rebalance import (
    plan_target_delta_values,
    qlib_open_cost,
)


def run(
    manifest_path: Path,
    qlib_data_dir: Path,
    qlib_commit: str,
    output_dir: Path,
    *,
    repetitions: int = 5,
    warmup_repetitions: int = 1,
) -> dict[str, Any]:
    import qlib
    from qlib.constant import REG_CN

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    contract = manifest["common_contract"]
    qlib.init(
        provider_uri=str(Path(qlib_data_dir).resolve()),
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
        kernels=1,
    )
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    def evaluate() -> tuple[dict[str, Any], dict[str, float], dict[str, np.ndarray]]:
        from qlib.contrib.evaluate import backtest_daily
        from qlib.data import D

        total_start = perf_counter()
        data_start = perf_counter()
        symbols = list(manifest["symbols"])
        frame = D.features(
            symbols,
            [str(manifest["qlib_expression"]), "$open"],
            start_time=str(manifest["start_date"]),
            end_time=str(manifest["end_date"]),
            freq="day",
            disk_cache=0,
        )
        factor = (
            frame[str(manifest["qlib_expression"])]
            .unstack(level="instrument")
            .sort_index()
            .sort_index(axis=1)
        )
        open_price = frame["$open"].unstack(level="instrument").reindex_like(factor)
        data_seconds = perf_counter() - data_start

        backtest_start = perf_counter()
        effective_end = pd.Timestamp(factor.index[-2])
        strategy = _strategy(
            factor.stack(future_stack=True),
            top_fraction=float(contract["top_fraction"]),
            buy_cost=float(contract["buy_cost"]),
            sell_cost=float(contract["sell_cost"]),
            final_date=effective_end,
        )
        benchmark = pd.Series(0.0, index=factor.index)
        report, _ = backtest_daily(
            start_time=factor.index[1],
            end_time=effective_end,
            strategy=strategy,
            account=100_000_000.0,
            benchmark=benchmark,
            exchange_kwargs={
                "deal_price": "$open",
                "open_cost": qlib_open_cost(float(contract["buy_cost"])),
                "close_cost": float(contract["sell_cost"]),
                "min_cost": 0.0,
                "limit_threshold": None,
                "trade_unit": None,
            },
        )
        gross = report["return"].to_numpy(dtype=float)
        cost = report["cost"].to_numpy(dtype=float)
        net = gross - cost
        backtest_seconds = perf_counter() - backtest_start

        ic_start = perf_counter()
        rank_ic = _rank_ic(
            factor.loc[:effective_end],
            open_price.loc[:effective_end],
            horizon=int(contract["horizon"]),
        )
        ic_seconds = perf_counter() - ic_start
        arrays = {
            "factor": factor.to_numpy(dtype=np.float64),
            "return_dates_ns": np.asarray(report.index.view("i8"), dtype=np.int64),
            "gross_returns": gross,
            "net_returns": net,
            "turnover_rates": report["turnover"].to_numpy(dtype=float),
            "cost_rates": cost,
        }
        evidence = {
            "strategy_metrics": _metrics(net),
            "average_turnover": float(report["turnover"].mean()),
            "total_cost": float(report["total_cost"].iloc[-1] / 100_000_000.0),
            "rank_ic": rank_ic,
            "target_selection_digest": target_selection_digest(
                factor.loc[:effective_end], top_fraction=float(contract["top_fraction"])
            ),
            "execution_adapter": "target_delta_continuous_value_v2",
            "turnover_coordinate": "traded_security_value_over_previous_portfolio_value",
            "effective_end": effective_end.date().isoformat(),
        }
        timings = {
            "data_factor": data_seconds,
            "portfolio": backtest_seconds,
            "rank_ic": ic_seconds,
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
        print(f"[common-backtest-qlib] repetition={position + 1}/{repetitions}", flush=True)
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
        "engine": {
            "name": "microsoft-qlib",
            "version": getattr(qlib, "__version__", "source-checkout"),
            "commit": qlib_commit,
        },
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


def _strategy(
    signal: pd.Series,
    *,
    top_fraction: float,
    buy_cost: float,
    sell_cost: float,
    final_date: pd.Timestamp,
) -> Any:
    from qlib.backtest.decision import Order, OrderDir
    from qlib.contrib.strategy.order_generator import OrderGenerator
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

    class TargetDeltaOrderGenerator(OrderGenerator):
        def generate_order_list_from_target_weight_position(
            self,
            current,
            trade_exchange,
            target_weight_position,
            risk_degree,
            pred_start_time,
            pred_end_time,
            trade_start_time,
            trade_end_time,
        ):
            del risk_degree, pred_start_time, pred_end_time
            current_amounts = current.get_stock_amount_dict()
            symbols = sorted(set(current_amounts) | set(target_weight_position or {}))
            prices = {}
            for stock_id in symbols:
                price = trade_exchange.get_deal_price(
                    stock_id, trade_start_time, trade_end_time, direction=OrderDir.BUY
                )
                prices[stock_id] = float(price)
            plan = plan_target_delta_values(
                current_values={
                    stock_id: float(amount) * prices[stock_id]
                    for stock_id, amount in current_amounts.items()
                },
                cash=float(current.get_cash()),
                target_weights=target_weight_position or {},
                buy_cost=buy_cost,
                sell_cost=sell_cost,
            )
            sell_orders = [
                Order(
                    stock_id=stock_id,
                    amount=value / prices[stock_id],
                    direction=OrderDir.SELL,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                for stock_id, value in plan.sell_values.items()
            ]
            buy_orders = [
                Order(
                    stock_id=stock_id,
                    amount=value / prices[stock_id],
                    direction=OrderDir.BUY,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                for stock_id, value in plan.buy_values.items()
            ]
            return sell_orders + buy_orders

    class EqualWeightTopFraction(WeightStrategyBase):
        def generate_target_weight_position(
            self, score, current, trade_start_time, trade_end_time
        ):
            del current, trade_start_time
            if pd.Timestamp(trade_end_time) >= final_date:
                return {}
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            ranked = score.dropna().sort_index().sort_values(ascending=False, kind="stable")
            count = max(1, int(math.ceil(len(ranked) * top_fraction))) if len(ranked) else 0
            selected = ranked.iloc[:count].index
            return {str(stock_id): 1.0 / count for stock_id in selected} if count else {}

    return EqualWeightTopFraction(
        signal=signal,
        risk_degree=1.0,
        order_generator_cls_or_obj=TargetDeltaOrderGenerator(),
    )


def _rank_ic(factor: pd.DataFrame, open_price: pd.DataFrame, *, horizon: int) -> dict[str, Any]:
    values: list[float] = []
    counts: list[int] = []
    for position in range(len(factor) - horizon - 1):
        signal = factor.iloc[position]
        forward = open_price.iloc[position + horizon + 1] / open_price.iloc[position + 1] - 1.0
        valid = signal.notna() & forward.notna()
        if int(valid.sum()) < 2:
            continue
        signal_rank = signal.loc[valid].rank(method="average").to_numpy(dtype=float)
        return_rank = forward.loc[valid].rank(method="average").to_numpy(dtype=float)
        value = float(np.corrcoef(signal_rank, return_rank)[0, 1])
        if np.isfinite(value):
            values.append(value)
            counts.append(int(valid.sum()))
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    mean = float(array.mean()) if len(array) else None
    return {
        "semantics": "signal_t_to_open_t_plus_1_to_t_plus_1_plus_h",
        "horizon": horizon,
        "observation_count": len(array),
        "rank_ic_mean": mean,
        "rank_ic_std": std if len(array) else None,
        "rank_ic_ir": float(mean / std) if mean is not None and std > 0.0 else None,
        "positive_rate": float((array > 0.0).mean()) if len(array) else None,
        "average_cross_section_count": float(np.mean(counts)) if counts else 0.0,
    }


def _metrics(returns: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=float)
    nav = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    std = float(values.std())
    peak = np.maximum.accumulate(nav)
    return {
        "total_return": float(nav[-1] - 1.0),
        "annual_return": float(nav[-1] ** (252.0 / len(values)) - 1.0),
        "sharpe": float(values.mean() / std * np.sqrt(252.0)) if std > 0.0 else 0.0,
        "max_drawdown": float(((peak - nav) / peak).max()),
        "period_count": len(values),
    }


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qlib-data-dir", type=Path, required=True)
    parser.add_argument("--qlib-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
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
    )
