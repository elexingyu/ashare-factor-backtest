from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.metrics import compute_periodic_metrics


Direction = Literal["high", "low"]


@dataclass(frozen=True)
class StaggeredLongOnlyResult:
    return_dates: tuple[pd.Timestamp, ...]
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    excess_returns: tuple[float, ...]
    turnover_cost_loads: tuple[float, ...]
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    excess_metrics: dict[str, float]
    average_turnover: float
    average_coverage: float
    average_selected_count: float
    average_hhi: float
    ic_mean: float
    ic_ir: float
    yearly_excess: dict[str, float]


@dataclass(frozen=True)
class ScreenVariant:
    direction: Direction
    horizon: int
    stress_excess_sharpe: float
    coverage: float
    periods: int


def evaluate_staggered_long_only(
    values: pd.DataFrame,
    open_prices: pd.DataFrame,
    *,
    direction: Direction,
    horizon: int,
    roundtrip_cost: float,
    decision_start: str,
    decision_end: str,
    top_fraction: float = 0.20,
    top_n: int | None = None,
    compute_ic: bool = True,
    benchmark_all_tradable: bool = False,
) -> StaggeredLongOnlyResult:
    if direction not in {"high", "low"}:
        raise ValueError("direction must be high or low")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= roundtrip_cost < 1:
        raise ValueError("roundtrip_cost must be in [0, 1)")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    common_dates = values.index.intersection(open_prices.index).sort_values()
    common_codes = values.columns.intersection(open_prices.columns).sort_values()
    factor = values.loc[common_dates, common_codes].astype(float)
    prices = open_prices.loc[common_dates, common_codes].astype(float)
    start = pd.Timestamp(decision_start)
    end = pd.Timestamp(decision_end)
    factor_array = factor.to_numpy(dtype=float, na_value=np.nan)
    price_array = prices.to_numpy(dtype=float, na_value=np.nan)
    rows, columns = factor_array.shape
    weights = np.zeros((rows, columns), dtype=np.float64)
    benchmark_weights = np.zeros((rows, columns), dtype=np.float64)
    decision_flags = np.zeros(rows, dtype=np.int16)
    turnover_by_decision = np.full(rows, np.nan, dtype=float)
    turnovers: list[float] = []
    coverages: list[float] = []
    selected_counts: list[int] = []
    hhis: list[float] = []
    ics: list[float] = []

    for position, decision_date in enumerate(common_dates):
        if decision_date < start or decision_date > end:
            continue
        entry_position = position + 1
        exit_position = entry_position + horizon
        if exit_position >= rows:
            continue
        if pd.Timestamp(common_dates[exit_position]) > end:
            continue
        signal = factor_array[position]
        path = price_array[entry_position : exit_position + 1]
        tradable = np.isfinite(path).all(axis=0) & (path > 0).all(axis=0)
        valid = np.isfinite(signal) & tradable
        valid_indexes = np.flatnonzero(valid)
        if len(valid_indexes) == 0:
            continue
        benchmark_indexes = np.flatnonzero(tradable) if benchmark_all_tradable else valid_indexes
        if len(benchmark_indexes) == 0:
            continue
        oriented = signal[valid_indexes] if direction == "high" else -signal[valid_indexes]
        count = max(1, int(math.ceil(len(valid_indexes) * top_fraction)))
        if top_n is not None:
            count = min(count, top_n)
        order = np.argsort(-oriented, kind="stable")[:count]
        selected_indexes = valid_indexes[order]
        weights[position, selected_indexes] = 1.0 / count
        benchmark_weights[position, benchmark_indexes] = 1.0 / len(benchmark_indexes)
        decision_flags[position] = 1
        previous_position = position - horizon
        if previous_position >= 0 and decision_flags[previous_position]:
            turnover = float(
                0.5 * np.abs(weights[position] - weights[previous_position]).sum()
            )
        else:
            turnover = 1.0
        turnover_by_decision[position] = turnover
        turnovers.append(turnover)
        coverages.append(len(valid_indexes) / len(common_codes))
        selected_counts.append(len(selected_indexes))
        hhis.append(1.0 / len(selected_indexes))

        if compute_ic:
            forward = path[-1, valid_indexes] / path[0, valid_indexes] - 1.0
            signal_series = pd.Series(signal[valid_indexes])
            forward_series = pd.Series(forward)
            if signal_series.nunique() > 1 and forward_series.nunique() > 1:
                ic = signal_series.corr(forward_series, method="spearman")
                if pd.notna(ic):
                    ics.append(float(ic))

    cumulative_weights = np.vstack(
        (np.zeros((1, columns)), np.cumsum(weights, axis=0))
    )
    cumulative_benchmark = np.vstack(
        (np.zeros((1, columns)), np.cumsum(benchmark_weights, axis=0))
    )
    cumulative_decisions = np.concatenate(([0], np.cumsum(decision_flags)))
    interval_returns = price_array[1:] / price_array[:-1] - 1.0
    return_dates_list: list[pd.Timestamp] = []
    gross_list: list[float] = []
    net_list: list[float] = []
    benchmark_list: list[float] = []
    cost_loads: list[float] = []
    for interval_position in range(1, rows - 1):
        first_decision = max(0, interval_position - horizon)
        active_count = int(
            cumulative_decisions[interval_position]
            - cumulative_decisions[first_decision]
        )
        if active_count != horizon:
            continue
        active_weights = (
            cumulative_weights[interval_position]
            - cumulative_weights[first_decision]
        ) / horizon
        active_benchmark = (
            cumulative_benchmark[interval_position]
            - cumulative_benchmark[first_decision]
        ) / horizon
        asset_returns = interval_returns[interval_position]
        nonfinite = ~np.isfinite(asset_returns)
        if np.any(nonfinite & ((active_weights > 0) | (active_benchmark > 0))):
            raise ValueError("an active portfolio weight has a nonfinite open return")
        safe_returns = np.where(nonfinite, 0.0, asset_returns)
        gross_return = float(np.dot(active_weights, safe_returns))
        benchmark_return = float(np.dot(active_benchmark, safe_returns))
        new_decision = interval_position - 1
        cost_load = float(turnover_by_decision[new_decision] / horizon)
        if not np.isfinite(cost_load):
            continue
        net_return = (1.0 + gross_return) * (1.0 - roundtrip_cost * cost_load) - 1.0
        return_dates_list.append(pd.Timestamp(common_dates[interval_position + 1]))
        gross_list.append(gross_return)
        net_list.append(net_return)
        benchmark_list.append(benchmark_return)
        cost_loads.append(cost_load)

    if not net_list:
        raise ValueError("staggered screen produced no returns")
    return_dates = tuple(return_dates_list)
    gross = tuple(gross_list)
    net = tuple(net_list)
    benchmark = tuple(benchmark_list)
    excess = tuple(
        (1.0 + strategy) / (1.0 + baseline) - 1.0
        for strategy, baseline in zip(net, benchmark, strict=True)
    )
    ic_array = np.asarray(ics, dtype=float)
    ic_std = float(np.std(ic_array, ddof=1)) if len(ic_array) > 1 else 0.0
    return StaggeredLongOnlyResult(
        return_dates=return_dates,
        gross_returns=gross,
        net_returns=net,
        benchmark_returns=benchmark,
        excess_returns=excess,
        turnover_cost_loads=tuple(cost_loads),
        metrics=compute_periodic_metrics(np.asarray(net), periods_per_year=252),
        benchmark_metrics=compute_periodic_metrics(np.asarray(benchmark), periods_per_year=252),
        excess_metrics=compute_periodic_metrics(np.asarray(excess), periods_per_year=252),
        average_turnover=float(np.mean(turnovers)),
        average_coverage=float(np.mean(coverages)),
        average_selected_count=float(np.mean(selected_counts)),
        average_hhi=float(np.mean(hhis)),
        ic_mean=float(np.mean(ic_array)) if len(ic_array) else float("nan"),
        ic_ir=float(np.mean(ic_array) / ic_std) if ic_std > 0 else float("nan"),
        yearly_excess=_yearly_returns(return_dates, excess),
    )


def choose_discovery_variant(
    variants: list[ScreenVariant], *, min_coverage: float, min_periods: int
) -> ScreenVariant:
    eligible = [
        variant
        for variant in variants
        if variant.coverage >= min_coverage and variant.periods >= min_periods
    ]
    if not eligible:
        raise ValueError("no discovery variant satisfies coverage and period gates")
    return max(
        eligible,
        key=lambda item: (
            item.stress_excess_sharpe,
            -item.horizon,
            item.direction == "high",
        ),
    )


def reprice_staggered_result(
    result: StaggeredLongOnlyResult, *, roundtrip_cost: float
) -> StaggeredLongOnlyResult:
    if not 0 <= roundtrip_cost < 1:
        raise ValueError("roundtrip_cost must be in [0, 1)")
    net = tuple(
        (1.0 + gross) * (1.0 - roundtrip_cost * cost_load) - 1.0
        for gross, cost_load in zip(
            result.gross_returns, result.turnover_cost_loads, strict=True
        )
    )
    excess = tuple(
        (1.0 + strategy) / (1.0 + baseline) - 1.0
        for strategy, baseline in zip(net, result.benchmark_returns, strict=True)
    )
    return replace(
        result,
        net_returns=net,
        excess_returns=excess,
        metrics=compute_periodic_metrics(np.asarray(net), periods_per_year=252),
        excess_metrics=compute_periodic_metrics(np.asarray(excess), periods_per_year=252),
        yearly_excess=_yearly_returns(result.return_dates, excess),
    )


def cluster_rank_outputs(
    outputs: Mapping[str, pd.DataFrame], *, start: str, end: str, threshold: float
) -> dict[str, str]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    names = sorted(outputs)
    if not names:
        return {}
    common_dates = outputs[names[0]].index
    common_columns = outputs[names[0]].columns
    for name in names[1:]:
        common_dates = common_dates.intersection(outputs[name].index)
        common_columns = common_columns.intersection(outputs[name].columns)
    common_dates = common_dates[(common_dates >= pd.Timestamp(start)) & (common_dates <= pd.Timestamp(end))]
    common_columns = common_columns.sort_values()
    vectors: dict[str, np.ndarray] = {}
    for name in names:
        frame = outputs[name].loc[common_dates, common_columns]
        ranks = frame.rank(axis=1, method="average", pct=True)
        vectors[name] = ranks.to_numpy(dtype=np.float32, na_value=np.nan).ravel()
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            valid = np.isfinite(vectors[left]) & np.isfinite(vectors[right])
            if int(valid.sum()) < 3:
                continue
            correlation = np.corrcoef(vectors[left][valid], vectors[right][valid])[0, 1]
            if np.isfinite(correlation) and abs(float(correlation)) >= threshold:
                union(left, right)
    roots = sorted({find(name) for name in names})
    labels = {root: f"cluster_{index:04d}" for index, root in enumerate(roots)}
    return {name: labels[find(name)] for name in names}


def _one_way_turnover(previous: tuple[str, ...], current: tuple[str, ...]) -> float:
    if not previous:
        return 1.0
    previous_weight = 1.0 / len(previous)
    current_weight = 1.0 / len(current)
    overlap = set(previous) & set(current)
    retained = sum(min(previous_weight, current_weight) for _ in overlap)
    return float(1.0 - retained)


def _yearly_returns(
    dates: tuple[pd.Timestamp, ...], returns: tuple[float, ...]
) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for date, value in zip(dates, returns, strict=True):
        grouped.setdefault(date.year, []).append(value)
    return {
        str(year): float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)
        for year, values in sorted(grouped.items())
    }
