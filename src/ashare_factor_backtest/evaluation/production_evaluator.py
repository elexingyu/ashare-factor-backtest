"""Causal long-only production evaluator with frozen next-open orders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.metrics import compute_periodic_metrics
from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
    build_production_execution_context,
)


Direction = Literal["high", "low"]


@dataclass(frozen=True)
class ProductionOrderEvent:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    sleeve: int
    status: str
    frozen: tuple[str, ...]
    filled: tuple[str, ...]
    unfilled: tuple[str, ...]
    cash_slots: int
    retained: tuple[str, ...] = ()
    bought: tuple[str, ...] = ()
    sold: tuple[str, ...] = ()
    blocked_sells: tuple[str, ...] = ()
    blocked_buys: tuple[str, ...] = ()
    residual: tuple[str, ...] = ()
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    planned_turnover: float = 0.0
    actual_turnover: float = 0.0
    target_tracking_error: float = 0.0


@dataclass(frozen=True)
class ProductionPortfolioResult:
    return_dates: tuple[pd.Timestamp, ...]
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    metrics: dict[str, float]
    order_events: tuple[ProductionOrderEvent, ...]
    blocked_exit_count: int
    forced_writeoff: float
    total_cost: float
    average_turnover: float
    average_coverage: float
    average_selected_count: float
    average_hhi: float
    minimum_cash_by_sleeve: tuple[float, ...]
    scheduled_rebalance_count: int = 0
    partial_rebalance_count: int = 0
    blocked_exit_order_count: int = 0
    blocked_entry_count: int = 0
    blocked_entry_order_count: int = 0
    average_planned_turnover: float = 0.0
    average_target_tracking_error: float = 0.0
    terminal_residual_value: float = 0.0


@dataclass(frozen=True)
class _RebalanceOutcome:
    cash: float
    buy_fee: float
    sell_fee: float
    buy_notional: float
    sell_notional: float
    planned_notional: float
    tracking_error: float
    retained: np.ndarray
    bought: np.ndarray
    sold: np.ndarray
    blocked_sells: np.ndarray
    blocked_buys: np.ndarray
    residual: np.ndarray
    filled: np.ndarray
    unfilled: np.ndarray
    status: str


@dataclass(frozen=True)
class ProductionLongOnlyResult:
    strategy: ProductionPortfolioResult
    benchmark: ProductionPortfolioResult
    excess_returns: tuple[float, ...]
    excess_metrics: dict[str, float]


def evaluate_production_staggered_long_only(
    values: pd.DataFrame,
    execution_frame: pd.DataFrame,
    *,
    direction: Direction,
    horizon: int,
    buy_cost: float,
    sell_cost: float,
    decision_start: str,
    decision_end: str,
    score_start: str | None = None,
    score_end: str | None = None,
    top_fraction: float = 0.20,
    top_n: int | None = None,
    record_events: bool = True,
    quantile_index: int | None = None,
    quantile_count: int = 5,
) -> ProductionLongOnlyResult:
    dates = pd.DatetimeIndex(pd.to_datetime(values.index)).sort_values()
    codes = pd.Index(sorted(str(code) for code in values.columns))
    context = build_production_execution_context(
        execution_frame, dates=dates, codes=codes
    )
    return evaluate_production_staggered_long_only_context(
        values,
        context,
        direction=direction,
        horizon=horizon,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        decision_start=decision_start,
        decision_end=decision_end,
        score_start=score_start,
        score_end=score_end,
        top_fraction=top_fraction,
        top_n=top_n,
        record_events=record_events,
        quantile_index=quantile_index,
        quantile_count=quantile_count,
    )


def evaluate_production_staggered_long_only_context(
    values: pd.DataFrame,
    context: ProductionExecutionContext,
    *,
    direction: Direction,
    horizon: int,
    buy_cost: float,
    sell_cost: float,
    decision_start: str,
    decision_end: str,
    score_start: str | None = None,
    score_end: str | None = None,
    top_fraction: float = 0.20,
    top_n: int | None = None,
    record_events: bool = True,
    benchmark_result: ProductionPortfolioResult | None = None,
    quantile_index: int | None = None,
    quantile_count: int = 5,
) -> ProductionLongOnlyResult:
    if direction not in {"high", "low"}:
        raise ValueError("direction must be high or low")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= buy_cost < 1 or not 0 <= sell_cost < 1:
        raise ValueError("buy and sell costs must be in [0, 1)")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive")
    if quantile_index is not None:
        if quantile_count < 3 or not 0 <= quantile_index < quantile_count:
            raise ValueError("quantile selector is outside the frozen partition")
        if top_n is not None:
            raise ValueError("quantile selector cannot be combined with top_n")
    strict_score_window = score_start is not None or score_end is not None
    trade_start = pd.Timestamp(decision_start)
    trade_end = pd.Timestamp(decision_end)
    metric_start = pd.Timestamp(score_start) if score_start is not None else trade_start
    metric_end = pd.Timestamp(score_end) if score_end is not None else trade_end
    if trade_start > trade_end:
        raise ValueError("decision window is invalid")
    if not trade_start <= metric_start <= metric_end <= trade_end:
        raise ValueError("score window must be inside the decision window")

    dates = context.dates
    codes = context.codes
    factor = values.reindex(index=dates, columns=codes).astype(float)
    factor_array = factor.to_numpy(dtype=float)

    common = {
        "dates": dates,
        "codes": codes,
        "valuation": context.valuation_open,
        "buyable": context.buyable,
        "sellable": context.sellable,
        "eligible": context.signal_eligible,
        "horizon": horizon,
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "start": trade_start,
        "end": trade_end,
        "score_start": metric_start,
        "score_end": metric_end,
        "record_events": record_events,
        "quantile_index": quantile_index,
        "quantile_count": quantile_count,
    }
    strategy = _simulate(
        factor=factor_array,
        direction=direction,
        top_fraction=top_fraction,
        top_n=top_n,
        benchmark=False,
        **common,
    )
    expected_score_dates = tuple(
        pd.Timestamp(value)
        for value in dates[(dates >= metric_start) & (dates <= metric_end)]
    )
    if strict_score_window and strategy.return_dates != expected_score_dates:
        raise ValueError(
            "score window is not fully initialized; provide enough warmup history"
        )
    benchmark = benchmark_result
    if benchmark is None:
        benchmark = _simulate(
            factor=factor_array,
            direction=direction,
            top_fraction=1.0,
            top_n=None,
            benchmark=True,
            **common,
        )
    if strategy.return_dates != benchmark.return_dates:
        raise RuntimeError("strategy and benchmark production dates are misaligned")
    if any(1.0 + value <= 0.0 for value in benchmark.net_returns):
        raise ValueError("production benchmark reached zero wealth; excess return is undefined")
    excess = tuple(
        (1.0 + strategy_value) / (1.0 + benchmark_value) - 1.0
        for strategy_value, benchmark_value in zip(
            strategy.net_returns, benchmark.net_returns, strict=True
        )
    )
    return ProductionLongOnlyResult(
        strategy=strategy,
        benchmark=benchmark,
        excess_returns=excess,
        excess_metrics=_metrics(np.asarray(excess, dtype=float)),
    )


def _rebalance_sleeve(
    holdings: np.ndarray,
    cash: float,
    targets: np.ndarray,
    *,
    buyable: np.ndarray,
    sellable: np.ndarray,
    buy_cost: float,
    sell_cost: float,
) -> _RebalanceOutcome:
    """Trade executable target-weight deltas while preserving blocked residuals."""
    tolerance = 1e-15
    before = holdings.copy()
    sleeve_wealth = float(before.sum() + cash)
    desired = np.zeros_like(holdings)
    if len(targets) and sleeve_wealth > 0.0:
        desired[targets] = sleeve_wealth / len(targets)

    requested_sells = np.maximum(before - desired, 0.0)
    requested_sell_indexes = np.flatnonzero(requested_sells > tolerance)
    sold = requested_sell_indexes[sellable[requested_sell_indexes]]
    blocked_sells = requested_sell_indexes[~sellable[requested_sell_indexes]]
    sell_amounts = requested_sells[sold]
    sell_notional = float(sell_amounts.sum())
    sell_fee = sell_notional * sell_cost
    holdings[sold] -= sell_amounts
    cash += sell_notional - sell_fee

    deficits = np.maximum(desired - holdings, 0.0)
    requested_buy_indexes = np.flatnonzero(deficits > tolerance)
    executable_buys = requested_buy_indexes[buyable[requested_buy_indexes]]
    blocked_buys = requested_buy_indexes[~buyable[requested_buy_indexes]]
    requested_budgets = deficits[requested_buy_indexes] / (1.0 - buy_cost)
    executable_budgets = deficits[executable_buys] / (1.0 - buy_cost)
    executable_total = float(executable_budgets.sum())
    scale = min(1.0, cash / executable_total) if executable_total > 0.0 else 0.0
    spent = executable_budgets * scale
    bought = executable_buys[spent > tolerance]
    spent = spent[spent > tolerance]
    buy_notional = float(spent.sum())
    holdings[bought] += spent * (1.0 - buy_cost)
    cash -= buy_notional
    if -1e-12 < cash < 0.0:
        cash = 0.0
    buy_fee = buy_notional * buy_cost

    target_mask = np.zeros(len(holdings), dtype=bool)
    target_mask[targets] = True
    held = holdings > tolerance
    retained = np.flatnonzero(target_mask & (before > tolerance) & held)
    residual = np.flatnonzero(~target_mask & held)
    filled = np.flatnonzero(target_mask & held)
    unfilled = np.flatnonzero(target_mask & ~held)
    posttrade_wealth = float(holdings.sum() + cash)
    if posttrade_wealth > 0.0:
        actual_weights = holdings / posttrade_wealth
        target_weights = np.zeros_like(holdings)
        if len(targets):
            target_weights[targets] = 1.0 / len(targets)
        tracking_error = 0.5 * (
            float(np.abs(actual_weights - target_weights).sum())
            + abs(cash / posttrade_wealth)
        )
    else:
        tracking_error = 0.0
    planned_notional = float(requested_sells.sum() + requested_budgets.sum())
    actual_notional = sell_notional + buy_notional
    partial = bool(len(blocked_sells) or len(blocked_buys) or len(unfilled))
    if partial:
        status = "partial_fill"
    elif actual_notional <= tolerance and len(filled):
        status = "retained"
    elif len(filled):
        status = "filled"
    else:
        status = "cash"
    return _RebalanceOutcome(
        cash=float(cash),
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        buy_notional=buy_notional,
        sell_notional=sell_notional,
        planned_notional=planned_notional,
        tracking_error=tracking_error,
        retained=retained,
        bought=bought,
        sold=sold,
        blocked_sells=blocked_sells,
        blocked_buys=blocked_buys,
        residual=residual,
        filled=filled,
        unfilled=unfilled,
        status=status,
    )


def _simulate(
    *,
    factor: np.ndarray,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    valuation: np.ndarray,
    buyable: np.ndarray,
    sellable: np.ndarray,
    eligible: np.ndarray,
    direction: Direction,
    horizon: int,
    buy_cost: float,
    sell_cost: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    top_fraction: float,
    top_n: int | None,
    benchmark: bool,
    record_events: bool,
    quantile_index: int | None,
    quantile_count: int,
) -> ProductionPortfolioResult:
    end_positions = np.flatnonzero(dates <= end)
    if not len(end_positions):
        raise ValueError("production decision end precedes all factor dates")
    end_position = int(end_positions[-1])
    if end_position < 2:
        raise ValueError("production evaluation window is too short")
    columns = len(codes)
    holdings = np.zeros((horizon, columns), dtype=float)
    cash = np.full(horizon, 1.0 / horizon, dtype=float)
    minimum_cash = cash.copy()
    initialized = np.zeros(horizon, dtype=bool)
    previous_total = 1.0
    return_dates: list[pd.Timestamp] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    events: list[ProductionOrderEvent] = []
    turnovers: list[float] = []
    planned_turnovers: list[float] = []
    tracking_errors: list[float] = []
    coverages: list[float] = []
    selected_counts: list[int] = []
    hhis: list[float] = []
    blocked_exit_count = 0
    blocked_exit_order_count = 0
    blocked_entry_count = 0
    blocked_entry_order_count = 0
    scheduled_rebalance_count = 0
    partial_rebalance_count = 0
    forced_writeoff = 0.0
    terminal_residual_value = 0.0
    total_cost = 0.0

    first_signal_position = int(dates.searchsorted(start, side="left"))
    # Before the first scheduled signal every sleeve is cash, so replaying the
    # earlier price history cannot affect orders, returns, or execution state.
    first_position = max(1, first_signal_position + 1)
    for position in range(first_position, end_position + 1):
        # Compact production contexts may store prices as float32; all portfolio
        # arithmetic remains float64 to limit storage rounding to the input boundary.
        previous_prices = np.asarray(valuation[position - 1], dtype=np.float64)
        current_prices = np.asarray(valuation[position], dtype=np.float64)
        ratio = np.ones(columns, dtype=float)
        valid_ratio = (
            np.isfinite(previous_prices)
            & np.isfinite(current_prices)
            & (previous_prices > 0)
            & (current_prices > 0)
        )
        ratio[valid_ratio] = current_prices[valid_ratio] / previous_prices[valid_ratio]
        holdings *= ratio[None, :]
        pretrade_total = float(holdings.sum() + cash.sum())
        gross_return = pretrade_total / previous_total - 1.0

        signal_position = position - 1
        signal_date = pd.Timestamp(dates[signal_position])
        entry_date = pd.Timestamp(dates[position])
        # On the terminal valuation date the portfolio is liquidated at the open.
        # Opening a fresh sleeve immediately before that liquidation would only
        # manufacture a round-trip cost with zero holding time.
        scheduled = start <= signal_date <= end and position < end_position
        scored_entry = score_start <= entry_date <= score_end
        if scheduled:
            sleeve = signal_position % horizon
            targets = _targets(
                factor[signal_position],
                eligible[signal_position],
                codes,
                direction=direction,
                top_fraction=top_fraction,
                top_n=top_n,
                benchmark=benchmark,
                quantile_index=quantile_index,
                quantile_count=quantile_count,
            )
            frozen = (
                tuple(str(codes[index]) for index in targets)
                if record_events
                else ()
            )
            outcome = _rebalance_sleeve(
                holdings[sleeve],
                float(cash[sleeve]),
                targets,
                buyable=buyable[position],
                sellable=sellable[position],
                buy_cost=buy_cost,
                sell_cost=sell_cost,
            )
            cash[sleeve] = outcome.cash
            actual_notional = outcome.sell_notional + outcome.buy_notional
            if scored_entry:
                scheduled_rebalance_count += 1
                total_cost += outcome.sell_fee + outcome.buy_fee
                turnovers.append(
                    actual_notional / pretrade_total if pretrade_total > 0 else 0.0
                )
                planned_turnovers.append(
                    outcome.planned_notional / pretrade_total
                    if pretrade_total > 0
                    else 0.0
                )
                tracking_errors.append(outcome.tracking_error)
                if len(outcome.blocked_sells):
                    blocked_exit_count += 1
                    blocked_exit_order_count += len(outcome.blocked_sells)
                if len(outcome.blocked_buys):
                    blocked_entry_count += 1
                    blocked_entry_order_count += len(outcome.blocked_buys)
                if outcome.status == "partial_fill":
                    partial_rebalance_count += 1
            initialized[sleeve] = True
            if record_events:
                def names(indexes: np.ndarray) -> tuple[str, ...]:
                    return tuple(str(codes[index]) for index in indexes)

                events.append(
                    ProductionOrderEvent(
                        signal_date=signal_date,
                        entry_date=entry_date,
                        sleeve=sleeve,
                        status=outcome.status,
                        frozen=frozen,
                        filled=names(outcome.filled),
                        unfilled=names(outcome.unfilled),
                        cash_slots=len(outcome.unfilled),
                        retained=names(outcome.retained),
                        bought=names(outcome.bought),
                        sold=names(outcome.sold),
                        blocked_sells=names(outcome.blocked_sells),
                        blocked_buys=names(outcome.blocked_buys),
                        residual=names(outcome.residual),
                        buy_notional=outcome.buy_notional,
                        sell_notional=outcome.sell_notional,
                        planned_turnover=(
                            outcome.planned_notional / pretrade_total
                            if pretrade_total > 0
                            else 0.0
                        ),
                        actual_turnover=(
                            actual_notional / pretrade_total
                            if pretrade_total > 0
                            else 0.0
                        ),
                        target_tracking_error=outcome.tracking_error,
                    )
                )
            if scored_entry:
                valid_signal = (
                    np.isfinite(factor[signal_position]) & eligible[signal_position]
                )
                eligible_count = int(eligible[signal_position].sum())
                coverages.append(
                    float(valid_signal.sum() / eligible_count)
                    if eligible_count
                    else 0.0
                )
                selected_counts.append(len(targets))
                portfolio_values = holdings.sum(axis=0)
                total = float(portfolio_values.sum() + cash.sum())
                hhis.append(
                    float(np.square(portfolio_values / total).sum())
                    if total > 0
                    else 0.0
                )

        if position == end_position:
            for sleeve in range(horizon):
                held = np.flatnonzero(holdings[sleeve] > 0.0)
                if not len(held):
                    continue
                tradable = held[sellable[position, held]]
                blocked = held[~sellable[position, held]]
                sell_notional = float(holdings[sleeve, tradable].sum())
                fee = sell_notional * sell_cost
                cash[sleeve] += sell_notional - fee
                if scored_entry:
                    total_cost += fee
                terminal_residual_value += float(holdings[sleeve, blocked].sum())
                holdings[sleeve, tradable] = 0.0

        minimum_cash = np.minimum(minimum_cash, cash)
        if (cash < -1e-12).any():
            raise RuntimeError("production evaluator created negative sleeve cash")
        posttrade_total = float(holdings.sum() + cash.sum())
        net_return = posttrade_total / previous_total - 1.0
        if initialized.all() and scored_entry:
            return_dates.append(entry_date)
            gross_returns.append(gross_return)
            net_returns.append(net_return)
        previous_total = posttrade_total

    if not net_returns:
        raise ValueError("production evaluator produced no returns")
    return ProductionPortfolioResult(
        return_dates=tuple(return_dates),
        gross_returns=tuple(gross_returns),
        net_returns=tuple(net_returns),
        metrics=_metrics(np.asarray(net_returns, dtype=float)),
        order_events=tuple(events),
        blocked_exit_count=blocked_exit_count,
        forced_writeoff=forced_writeoff,
        total_cost=total_cost,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        average_coverage=float(np.mean(coverages)) if coverages else 0.0,
        average_selected_count=float(np.mean(selected_counts)) if selected_counts else 0.0,
        average_hhi=float(np.mean(hhis)) if hhis else 0.0,
        minimum_cash_by_sleeve=tuple(float(value) for value in minimum_cash),
        scheduled_rebalance_count=scheduled_rebalance_count,
        partial_rebalance_count=partial_rebalance_count,
        blocked_exit_order_count=blocked_exit_order_count,
        blocked_entry_count=blocked_entry_count,
        blocked_entry_order_count=blocked_entry_order_count,
        average_planned_turnover=(
            float(np.mean(planned_turnovers)) if planned_turnovers else 0.0
        ),
        average_target_tracking_error=(
            float(np.mean(tracking_errors)) if tracking_errors else 0.0
        ),
        terminal_residual_value=terminal_residual_value,
    )


def _targets(
    signal: np.ndarray,
    eligible: np.ndarray,
    codes: pd.Index,
    *,
    direction: Direction,
    top_fraction: float,
    top_n: int | None,
    benchmark: bool,
    quantile_index: int | None = None,
    quantile_count: int = 5,
) -> np.ndarray:
    if benchmark:
        return np.flatnonzero(eligible)
    valid = np.isfinite(signal) & eligible
    indexes = np.flatnonzero(valid)
    if not len(indexes):
        return indexes
    if quantile_index is not None:
        if len(indexes) == 1:
            return indexes if quantile_index == quantile_count // 2 else indexes[:0]
        _, inverse, counts = np.unique(
            signal[indexes], return_inverse=True, return_counts=True
        )
        starts = np.cumsum(np.r_[0, counts[:-1]])
        average_positions = starts + (counts - 1) / 2.0
        percentiles = average_positions[inverse] / (len(indexes) - 1)
        bands = np.minimum(
            np.floor(percentiles * quantile_count).astype(int), quantile_count - 1
        )
        return indexes[bands == quantile_index]
    oriented = signal[indexes] if direction == "high" else -signal[indexes]
    count = max(1, int(math.ceil(len(indexes) * top_fraction)))
    if top_n is not None:
        count = min(count, top_n)
    order = np.argsort(-oriented, kind="stable")[:count]
    return indexes[order]


def _metrics(returns: np.ndarray) -> dict[str, float]:
    bankrupt = bool((returns <= -1.0).any())
    metric_returns = np.maximum(returns, np.nextafter(-1.0, 0.0))
    metrics = compute_periodic_metrics(metric_returns, periods_per_year=252)
    metrics["bankrupt"] = bankrupt
    return metrics
