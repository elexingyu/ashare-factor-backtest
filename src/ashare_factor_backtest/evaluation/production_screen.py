"""Candidate-level production screen over chunked factors and shared execution state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable, Mapping
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.metrics import compute_periodic_metrics
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.expression.catalog import OperatorCatalog
from ashare_factor_backtest.expression.operators.registry import build_production_operator_catalog
from ashare_factor_backtest.evaluation.cheap_evaluator import (
    ScreenVariant,
    choose_discovery_variant,
)
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    FrameLoader,
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_evaluator import (
    ProductionLongOnlyResult,
    ProductionPortfolioResult,
    evaluate_production_staggered_long_only_context,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
)
from ashare_factor_backtest.evaluation.production_rank_ic import (
    evaluate_production_rank_ic,
)
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


@dataclass(frozen=True)
class ProductionScreenPolicy:
    discovery: tuple[str, str]
    validation: tuple[str, str]
    horizons: tuple[int, ...] = (5, 20, 60)
    mode: str = "staggered_horizon"
    fixed_direction: str | None = None
    decay_horizons: tuple[int, ...] = ()
    minimum_coverage: float = 0.70
    minimum_periods: int = 504
    top_fraction: float = 0.20
    real_buy_cost: float = 0.0003
    real_sell_cost: float = 0.0012
    stress_buy_cost: float = 0.0005
    stress_sell_cost: float = 0.0020

    def __post_init__(self) -> None:
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("production screen horizons must be positive")
        if self.mode not in {"staggered_horizon", "daily_factor"}:
            raise ValueError("production screen mode is unsupported")
        if self.fixed_direction not in {None, "high", "low"}:
            raise ValueError("production screen fixed_direction must be high or low")
        if any(value <= 0 for value in self.decay_horizons):
            raise ValueError("production screen decay_horizons must be positive")
        if self.mode == "daily_factor":
            if self.horizons != (1,):
                raise ValueError("daily_factor mode requires one daily account sleeve")
            if self.fixed_direction is None:
                raise ValueError("daily_factor mode requires a frozen direction")
        if not 0 < self.minimum_coverage <= 1 or self.minimum_periods <= 0:
            raise ValueError("production screen coverage and periods must be positive")
        if self.discovery[0] > self.discovery[1] or self.validation[0] > self.validation[1]:
            raise ValueError("production screen date windows are invalid")


def screen_production_candidate(
    candidate: dict[str, Any],
    *,
    chunks: Sequence[YearChunk],
    frame_loader: FrameLoader,
    execution_context: ProductionExecutionContext,
    dataset_version: str,
    view: str,
    execution_contract: str,
    cache_max_bytes: int,
    policy: ProductionScreenPolicy,
    operator_catalog_builder: Callable[
        [], tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]
    ] = build_production_operator_catalog,
) -> dict[str, Any]:
    evaluated = evaluate_expression_by_year(
        str(candidate["expression"]),
        chunks=chunks,
        frame_loader=frame_loader,
        dataset_version=dataset_version,
        view=view,
        cache_max_bytes=cache_max_bytes,
        required_fields=set(referenced_fields(str(candidate["expression"]))),
        spill_to_disk=True,
        operator_catalog_builder=operator_catalog_builder,
    )
    expected_canonical = str(candidate["canonical"])
    if evaluated.canonical != expected_canonical:
        raise RuntimeError(
            f"production canonical expression changed for {candidate['candidate_id']}"
        )

    core = screen_production_values(evaluated.values, execution_context, policy=policy)
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "factor_id": evaluated.factor_id,
        "source_formula_id": str(candidate["factor_id"]),
        "expression": str(candidate["expression"]),
        "canonical": evaluated.canonical,
        "status": "research_proxy",
        "execution_contract": execution_contract,
        "view": view,
        **core,
        "chunk_count": len(evaluated.chunks),
        "factor_lookback": evaluated.lookback,
    }


def screen_production_values(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionScreenPolicy,
    benchmark_cache: dict[tuple[Any, ...], ProductionPortfolioResult] | None = None,
    result_observer: Callable[
        [str, str, str, int, ProductionLongOnlyResult], object
    ]
    | None = None,
) -> dict[str, Any]:
    """Apply the frozen production selection contract to an existing factor panel."""

    if policy.mode == "daily_factor":
        return evaluate_production_daily_factor_values(
            values,
            execution_context,
            policy=policy,
            benchmark_cache=benchmark_cache,
        )

    if _discovery_average_coverage(values, execution_context, policy) < policy.minimum_coverage:
        raise ValueError("no discovery variant satisfies coverage and period gates")

    cache = {} if benchmark_cache is None else benchmark_cache

    def evaluate(
        *, direction: str, horizon: int, buy_cost: float, sell_cost: float,
        decision_start: str, decision_end: str, segment: str, cost_label: str,
    ) -> ProductionLongOnlyResult:
        trade_start, fully_warmed = _warmup_start(
            execution_context.dates, score_start=decision_start, horizon=horizon
        )
        metric_window = (
            {"score_start": decision_start, "score_end": decision_end}
            if fully_warmed
            else {}
        )
        key = (
            horizon,
            buy_cost,
            sell_cost,
            trade_start,
            decision_end,
            decision_start if fully_warmed else None,
            decision_end if fully_warmed else None,
        )
        result = evaluate_production_staggered_long_only_context(
            values,
            execution_context,
            direction=direction,
            horizon=horizon,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            decision_start=trade_start,
            decision_end=decision_end,
            top_fraction=policy.top_fraction,
            record_events=False,
            benchmark_result=cache.get(key),
            **metric_window,
        )
        cache.setdefault(key, result.benchmark)
        if result_observer is not None:
            result_observer(segment, cost_label, direction, horizon, result)
        return result

    variant_results: dict[tuple[str, int], ProductionLongOnlyResult] = {}
    variants: list[ScreenVariant] = []
    for direction in ("high", "low"):
        for horizon in policy.horizons:
            result = evaluate(
                direction=direction,
                horizon=horizon,
                buy_cost=policy.stress_buy_cost,
                sell_cost=policy.stress_sell_cost,
                decision_start=policy.discovery[0],
                decision_end=policy.discovery[1],
                segment="train",
                cost_label="stress",
            )
            variant_results[(direction, horizon)] = result
            variants.append(
                ScreenVariant(
                    direction=direction,
                    horizon=horizon,
                    stress_excess_sharpe=float(result.excess_metrics["sharpe"]),
                    coverage=result.strategy.average_coverage,
                    periods=len(result.strategy.net_returns),
                )
            )
    choice = choose_discovery_variant(
        variants,
        min_coverage=policy.minimum_coverage,
        min_periods=policy.minimum_periods,
    )
    discovery_stress = variant_results[(choice.direction, choice.horizon)]
    discovery_real = evaluate(
        direction=choice.direction,
        horizon=choice.horizon,
        buy_cost=policy.real_buy_cost,
        sell_cost=policy.real_sell_cost,
        decision_start=policy.discovery[0],
        decision_end=policy.discovery[1],
        segment="train",
        cost_label="real",
    )
    validation_stress = evaluate(
        direction=choice.direction,
        horizon=choice.horizon,
        buy_cost=policy.stress_buy_cost,
        sell_cost=policy.stress_sell_cost,
        decision_start=policy.validation[0],
        decision_end=policy.validation[1],
        segment="test",
        cost_label="stress",
    )
    validation_real = evaluate(
        direction=choice.direction,
        horizon=choice.horizon,
        buy_cost=policy.real_buy_cost,
        sell_cost=policy.real_sell_cost,
        decision_start=policy.validation[0],
        decision_end=policy.validation[1],
        segment="test",
        cost_label="real",
    )
    discovery_rank_ic = evaluate_production_rank_ic(
        values,
        execution_context,
        horizon=choice.horizon,
        signal_start=policy.discovery[0],
        signal_end=policy.discovery[1],
    )
    validation_rank_ic = evaluate_production_rank_ic(
        values,
        execution_context,
        horizon=choice.horizon,
        signal_start=policy.validation[0],
        signal_end=policy.validation[1],
    )
    return {
        "selected_direction": choice.direction,
        "selected_horizon": choice.horizon,
        "discovery_variants": [asdict(item) for item in variants],
        "discovery": {
            "direction": choice.direction,
            "horizon": choice.horizon,
            "rank_ic": discovery_rank_ic,
            "real": _summary(discovery_real),
            "stress": _summary(discovery_stress),
        },
        "validation": {
            "direction": choice.direction,
            "horizon": choice.horizon,
            "rank_ic": validation_rank_ic,
            "real": _summary(validation_real),
            "stress": _summary(validation_stress),
            "yearly_real": _yearly_summary(validation_real),
            "yearly_stress": _yearly_summary(validation_stress),
        },
    }


def evaluate_production_daily_factor_values(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionScreenPolicy,
    benchmark_cache: dict[tuple[Any, ...], ProductionPortfolioResult] | None = None,
) -> dict[str, Any]:
    """Evaluate a daily target portfolio; horizons are decay diagnostics only."""
    if policy.mode != "daily_factor" or policy.fixed_direction is None:
        raise ValueError("daily factor evaluation requires its frozen policy mode")
    report = evaluate_production_fixed_values(
        values,
        execution_context,
        policy=policy,
        direction=policy.fixed_direction,
        horizon=1,
        benchmark_cache=benchmark_cache,
    )
    diagnostic_horizons = policy.decay_horizons or (1, 5, 20, 60)
    report["evaluation_mode"] = "daily_factor"
    report["portfolio_construction"] = {
        "signal_refresh": "daily",
        "target_refresh": "daily",
        "account_sleeves": 1,
        "fixed_holding_period_days": None,
        "turnover_semantics": "net_target_change",
    }
    report["selected_horizon"] = None
    report["decay_horizons"] = list(diagnostic_horizons)
    for segment, window in (
        ("discovery", policy.discovery),
        ("validation", policy.validation),
    ):
        report[segment]["horizon"] = None
        report[segment].pop("rank_ic", None)
        report[segment]["rank_ic_decay"] = {
            str(horizon): evaluate_production_rank_ic(
                values,
                execution_context,
                horizon=horizon,
                signal_start=window[0],
                signal_end=window[1],
            )
            for horizon in diagnostic_horizons
        }
    return report


def screen_production_stress_values(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionScreenPolicy,
    benchmark_cache: dict[tuple[Any, ...], ProductionPortfolioResult] | None = None,
    result_observer: Callable[
        [str, str, str, int, ProductionLongOnlyResult], object
    ]
    | None = None,
) -> dict[str, Any]:
    """Select on train stress costs and freeze one stress-cost validation run."""

    if _discovery_average_coverage(values, execution_context, policy) < policy.minimum_coverage:
        raise ValueError("no discovery variant satisfies coverage and period gates")

    cache = {} if benchmark_cache is None else benchmark_cache

    def evaluate(
        *, direction: str, horizon: int, decision_start: str, decision_end: str,
        segment: str,
    ) -> ProductionLongOnlyResult:
        trade_start, fully_warmed = _warmup_start(
            execution_context.dates, score_start=decision_start, horizon=horizon
        )
        metric_window = (
            {"score_start": decision_start, "score_end": decision_end}
            if fully_warmed
            else {}
        )
        key = (
            horizon,
            policy.stress_buy_cost,
            policy.stress_sell_cost,
            trade_start,
            decision_end,
            decision_start if fully_warmed else None,
            decision_end if fully_warmed else None,
        )
        result = evaluate_production_staggered_long_only_context(
            values,
            execution_context,
            direction=direction,
            horizon=horizon,
            buy_cost=policy.stress_buy_cost,
            sell_cost=policy.stress_sell_cost,
            decision_start=trade_start,
            decision_end=decision_end,
            top_fraction=policy.top_fraction,
            record_events=False,
            benchmark_result=cache.get(key),
            **metric_window,
        )
        cache.setdefault(key, result.benchmark)
        if result_observer is not None:
            result_observer(segment, "stress", direction, horizon, result)
        return result

    variant_results: dict[tuple[str, int], ProductionLongOnlyResult] = {}
    variants: list[ScreenVariant] = []
    for direction in ("high", "low"):
        for horizon in policy.horizons:
            result = evaluate(
                direction=direction,
                horizon=horizon,
                decision_start=policy.discovery[0],
                decision_end=policy.discovery[1],
                segment="train",
            )
            variant_results[(direction, horizon)] = result
            variants.append(
                ScreenVariant(
                    direction=direction,
                    horizon=horizon,
                    stress_excess_sharpe=float(result.excess_metrics["sharpe"]),
                    coverage=result.strategy.average_coverage,
                    periods=len(result.strategy.net_returns),
                )
            )
    choice = choose_discovery_variant(
        variants,
        min_coverage=policy.minimum_coverage,
        min_periods=policy.minimum_periods,
    )
    discovery_stress = variant_results[(choice.direction, choice.horizon)]
    validation_stress = evaluate(
        direction=choice.direction,
        horizon=choice.horizon,
        decision_start=policy.validation[0],
        decision_end=policy.validation[1],
        segment="test",
    )
    return {
        "selected_direction": choice.direction,
        "selected_horizon": choice.horizon,
        "discovery_variants": [asdict(item) for item in variants],
        "discovery": {
            "direction": choice.direction,
            "horizon": choice.horizon,
            "stress": _summary(discovery_stress),
        },
        "validation": {
            "direction": choice.direction,
            "horizon": choice.horizon,
            "stress": _summary(validation_stress),
            "yearly_stress": _yearly_summary(validation_stress),
        },
    }


def evaluate_production_fixed_values(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionScreenPolicy,
    direction: str,
    horizon: int,
    benchmark_cache: dict[tuple[Any, ...], ProductionPortfolioResult] | None = None,
) -> dict[str, Any]:
    """Evaluate a preselected direction and horizon without parameter reselection."""
    if direction not in {"high", "low"} or horizon not in policy.horizons:
        raise ValueError("fixed production variant is outside the frozen policy")
    if _discovery_average_coverage(values, execution_context, policy) < policy.minimum_coverage:
        raise ValueError("no discovery variant satisfies coverage and period gates")
    cache = {} if benchmark_cache is None else benchmark_cache

    def evaluate(
        *, buy_cost: float, sell_cost: float, decision_start: str, decision_end: str
    ) -> ProductionLongOnlyResult:
        trade_start, fully_warmed = _warmup_start(
            execution_context.dates, score_start=decision_start, horizon=horizon
        )
        metric_window = (
            {"score_start": decision_start, "score_end": decision_end}
            if fully_warmed
            else {}
        )
        key = (
            horizon,
            buy_cost,
            sell_cost,
            trade_start,
            decision_end,
            decision_start if fully_warmed else None,
            decision_end if fully_warmed else None,
        )
        result = evaluate_production_staggered_long_only_context(
            values,
            execution_context,
            direction=direction,
            horizon=horizon,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            decision_start=trade_start,
            decision_end=decision_end,
            top_fraction=policy.top_fraction,
            record_events=False,
            benchmark_result=cache.get(key),
            **metric_window,
        )
        cache.setdefault(key, result.benchmark)
        return result

    discovery_stress = evaluate(
        buy_cost=policy.stress_buy_cost,
        sell_cost=policy.stress_sell_cost,
        decision_start=policy.discovery[0],
        decision_end=policy.discovery[1],
    )
    discovery_real = evaluate(
        buy_cost=policy.real_buy_cost,
        sell_cost=policy.real_sell_cost,
        decision_start=policy.discovery[0],
        decision_end=policy.discovery[1],
    )
    validation_stress = evaluate(
        buy_cost=policy.stress_buy_cost,
        sell_cost=policy.stress_sell_cost,
        decision_start=policy.validation[0],
        decision_end=policy.validation[1],
    )
    validation_real = evaluate(
        buy_cost=policy.real_buy_cost,
        sell_cost=policy.real_sell_cost,
        decision_start=policy.validation[0],
        decision_end=policy.validation[1],
    )
    discovery_rank_ic = evaluate_production_rank_ic(
        values,
        execution_context,
        horizon=horizon,
        signal_start=policy.discovery[0],
        signal_end=policy.discovery[1],
    )
    validation_rank_ic = evaluate_production_rank_ic(
        values,
        execution_context,
        horizon=horizon,
        signal_start=policy.validation[0],
        signal_end=policy.validation[1],
    )
    return {
        "selected_direction": direction,
        "selected_horizon": horizon,
        "discovery_variants": [],
        "discovery": {
            "direction": direction,
            "horizon": horizon,
            "rank_ic": discovery_rank_ic,
            "real": _summary(discovery_real),
            "stress": _summary(discovery_stress),
        },
        "validation": {
            "direction": direction,
            "horizon": horizon,
            "rank_ic": validation_rank_ic,
            "real": _summary(validation_real),
            "stress": _summary(validation_stress),
            "yearly_real": _yearly_summary(validation_real),
            "yearly_stress": _yearly_summary(validation_stress),
        },
    }


def _warmup_start(
    dates: pd.DatetimeIndex, *, score_start: str, horizon: int
) -> tuple[str, bool]:
    """Return the first signal date needed for a fully invested score start."""
    start_position = int(dates.searchsorted(pd.Timestamp(score_start), side="left"))
    if start_position >= len(dates):
        raise ValueError("score start follows all production dates")
    warmup_position = max(0, start_position - horizon)
    return (
        pd.Timestamp(dates[warmup_position]).date().isoformat(),
        start_position >= horizon,
    )


def _discovery_average_coverage(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    policy: ProductionScreenPolicy,
) -> float:
    dates = execution_context.dates
    positions = np.flatnonzero(dates <= pd.Timestamp(policy.discovery[1]))
    if not len(positions):
        return 0.0
    final_signal_position = int(positions[-1]) - 1
    if final_signal_position < 0:
        return 0.0
    signal_positions = np.flatnonzero(
        (dates >= pd.Timestamp(policy.discovery[0]))
        & (np.arange(len(dates)) <= final_signal_position)
    )
    if not len(signal_positions):
        return 0.0
    factor = values.reindex(
        index=dates, columns=execution_context.codes
    ).to_numpy(dtype=float)
    eligible = execution_context.signal_eligible[signal_positions]
    valid = np.isfinite(factor[signal_positions]) & eligible
    denominators = eligible.sum(axis=1)
    coverage = np.divide(
        valid.sum(axis=1),
        denominators,
        out=np.zeros(len(signal_positions), dtype=float),
        where=denominators > 0,
    )
    return float(coverage.mean())


def _summary(result: ProductionLongOnlyResult) -> dict[str, Any]:
    strategy = result.strategy
    benchmark = result.benchmark
    return {
        "periods": len(strategy.net_returns),
        "strategy_metrics": strategy.metrics,
        "benchmark_metrics": benchmark.metrics,
        "excess_metrics": result.excess_metrics,
        "average_turnover": strategy.average_turnover,
        "average_coverage": strategy.average_coverage,
        "average_selected_count": strategy.average_selected_count,
        "average_hhi": strategy.average_hhi,
        "blocked_exit_count": strategy.blocked_exit_count,
        "blocked_exit_order_count": strategy.blocked_exit_order_count,
        "blocked_entry_count": strategy.blocked_entry_count,
        "blocked_entry_order_count": strategy.blocked_entry_order_count,
        "scheduled_rebalance_count": strategy.scheduled_rebalance_count,
        "partial_rebalance_count": strategy.partial_rebalance_count,
        "forced_writeoff": strategy.forced_writeoff,
        "terminal_residual_value": strategy.terminal_residual_value,
        "total_cost": strategy.total_cost,
        "average_planned_turnover": strategy.average_planned_turnover,
        "average_target_tracking_error": strategy.average_target_tracking_error,
        "minimum_cash": min(strategy.minimum_cash_by_sleeve),
    }


def _yearly_summary(result: ProductionLongOnlyResult) -> dict[str, Any]:
    dates = pd.DatetimeIndex(result.strategy.return_dates)
    if tuple(dates) != result.benchmark.return_dates:
        raise RuntimeError("yearly production return dates are misaligned")
    strategy = np.asarray(result.strategy.net_returns, dtype=float)
    benchmark = np.asarray(result.benchmark.net_returns, dtype=float)
    excess = np.asarray(result.excess_returns, dtype=float)
    yearly: dict[str, Any] = {}
    for year in sorted(set(dates.year)):
        mask = dates.year == year
        yearly[str(year)] = {
            "strategy_metrics": _return_metrics(strategy[mask]),
            "benchmark_metrics": _return_metrics(benchmark[mask]),
            "excess_metrics": _return_metrics(excess[mask]),
        }
    return yearly


def _return_metrics(returns: np.ndarray) -> dict[str, Any]:
    bankrupt = bool((returns <= -1.0).any())
    safe = np.maximum(returns, np.nextafter(-1.0, 0.0))
    metrics = compute_periodic_metrics(safe, periods_per_year=252)
    metrics["bankrupt"] = bankrupt
    return metrics
