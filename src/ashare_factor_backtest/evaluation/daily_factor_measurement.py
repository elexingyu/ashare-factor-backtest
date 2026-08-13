"""Canonical daily factor measurement without account or admission semantics."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
)


MEASUREMENT_ENGINE_VERSION = "ashare-daily-factor-measurement.v2"
TRADING_DAYS_PER_YEAR = 242
TOP_FRACTION = 0.20
HAC_LAG = 20


def continuous_rank_weights(scores: np.ndarray, *, direction: float = 1.0) -> np.ndarray:
    """Return a zero-net, unit-gross rank book for one cross-section."""

    values = np.asarray(scores, dtype=float)
    weights = np.zeros(values.shape, dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return weights
    ranks = _rankdata(values[valid])
    centered = float(direction) * (ranks - ranks.mean())
    gross = float(np.abs(centered).sum())
    if gross > 0.0:
        weights[valid] = centered / gross
    return weights


def measure_daily_factor(
    values: pd.DataFrame,
    context: ProductionExecutionContext,
    *,
    direction: str,
    horizons: Sequence[int],
    rolling_windows: Sequence[int] = (252, 504),
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Measure one factor once; callers may describe or gate the same artifact."""

    sign = {"high": 1.0, "low": -1.0}.get(direction)
    if sign is None:
        raise ValueError("factor direction must be high or low")
    fixed_horizons = _positive_unique(horizons, "horizons")
    fixed_rolling = _positive_unique(rolling_windows, "rolling_windows")
    if max(fixed_horizons) + 1 >= len(context.dates):
        raise ValueError("measurement history is shorter than the maximum horizon")

    factor = values.reindex(index=context.dates, columns=context.codes).to_numpy(
        dtype=float, na_value=np.nan
    )
    opens = np.asarray(context.valuation_open, dtype=float)
    closes = np.asarray(context.valuation_close, dtype=float)
    eligible = np.asarray(context.signal_eligible, dtype=bool)
    trace_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    previous_weights: np.ndarray | None = None
    previous_members: set[int] | None = None
    weight_digest = hashlib.sha256()

    for signal_position in range(len(context.dates) - 2):
        entry_position = signal_position + 1
        exit_position = signal_position + 2
        scores = factor[signal_position]
        base_valid = eligible[signal_position] & np.isfinite(scores)
        directed_scores = np.where(base_valid, sign * scores, np.nan)
        weights = continuous_rank_weights(directed_scores)
        gross_exposure = float(np.abs(weights).sum())
        weight_digest.update(np.asarray(weights, dtype="<f8").tobytes())
        period_returns = _asset_return(opens[entry_position], opens[exit_position])
        intraday_returns = _asset_return(opens[entry_position], closes[entry_position])
        overnight_returns = _asset_return(closes[entry_position], opens[exit_position])
        factor_return = _weighted_return(weights, period_returns)
        intraday_return = _weighted_return(weights, intraday_returns)
        overnight_return = _weighted_return(weights, overnight_returns)
        rank_ic, rank_count = _rank_ic(directed_scores, period_returns)
        groups = _quantile_groups(directed_scores, 5)
        quintiles = [_group_mean(period_returns, group) for group in groups]
        top_members = set(int(value) for value in groups[-1]) if groups else set()
        bottom_members = set(int(value) for value in groups[0]) if groups else set()
        top_return = _group_mean(period_returns, np.asarray(sorted(top_members)))
        bottom_return = _group_mean(period_returns, np.asarray(sorted(bottom_members)))
        membership_turnover = (
            np.nan
            if previous_members is None
            else _jaccard_turnover(previous_members, top_members)
        )
        target_turnover = (
            np.nan
            if previous_weights is None
            else 0.5 * float(np.abs(weights - previous_weights).sum())
        )
        signal_date = pd.Timestamp(context.dates[signal_position])
        for member in sorted(top_members):
            membership_rows.append(
                {"signal_date": signal_date, "ts_code": str(context.codes[member])}
            )
        row: dict[str, Any] = {
            "signal_date": signal_date,
            "entry_date": pd.Timestamp(context.dates[entry_position]),
            "exit_date": pd.Timestamp(context.dates[exit_position]),
            "eligible_count": int(eligible[signal_position].sum()),
            "scored_count": int(base_valid.sum()),
            "coverage": _ratio(int(base_valid.sum()), int(eligible[signal_position].sum())),
            "gross_exposure": gross_exposure,
            "rank_ic": rank_ic,
            "rank_ic_count": rank_count,
            "factor_return": factor_return,
            "intraday_return": intraday_return,
            "overnight_return": overnight_return,
            "target_weight_turnover": target_turnover,
            "top20_membership_turnover": membership_turnover,
            "top20_return": top_return,
            "bottom20_return": bottom_return,
            "top_bottom_return": (
                top_return - bottom_return
                if np.isfinite(top_return) and np.isfinite(bottom_return)
                else np.nan
            ),
        }
        for position, result in enumerate(quintiles, start=1):
            row[f"q{position}_return"] = result
        trace_rows.append(row)
        previous_weights = weights
        previous_members = top_members

    trace = pd.DataFrame(trace_rows)
    memberships = pd.DataFrame(
        membership_rows, columns=["signal_date", "ts_code"]
    )
    horizon_curve = _horizon_curve(
        factor,
        context,
        sign=sign,
        horizons=fixed_horizons,
    )
    summary = _summarize_measurement(
        trace,
        direction=direction,
        horizons=horizon_curve,
        rolling_windows=fixed_rolling,
        weight_hash=weight_digest.hexdigest(),
    )
    return summary, trace, memberships


def summarize_daily_returns(values: np.ndarray) -> dict[str, Any]:
    returns = np.asarray(values, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return _empty_return_summary(len(returns))
    volatility = float(np.std(returns, ddof=1))
    curve = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(curve)
    mean = float(np.mean(returns))
    return {
        "observations": int(len(returns)),
        "mean_daily_return": mean,
        "annualized_mean_return": mean * TRADING_DAYS_PER_YEAR,
        "annualized_volatility": volatility * np.sqrt(TRADING_DAYS_PER_YEAR),
        "annualized_sharpe": (
            mean / volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
            if volatility > 0.0
            else None
        ),
        "maximum_drawdown": float(np.min(curve / peaks - 1.0)),
        "positive_day_ratio": float(np.mean(returns > 0.0)),
        "terminal_growth": float(curve[-1] - 1.0),
        "hac_t_stat": newey_west_mean_tstat(returns),
    }


def newey_west_mean_tstat(values: np.ndarray, *, max_lag: int = HAC_LAG) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 3:
        return None
    centered = array - array.mean()
    lag = min(int(max_lag), len(array) - 1)
    long_run_variance = float(np.dot(centered, centered) / len(array))
    for offset in range(1, lag + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / len(array)
        )
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    if long_run_variance <= 0.0:
        return None
    return float(array.mean() / np.sqrt(long_run_variance / len(array)))


def _summarize_measurement(
    trace: pd.DataFrame,
    *,
    direction: str,
    horizons: list[dict[str, Any]],
    rolling_windows: tuple[int, ...],
    weight_hash: str,
) -> dict[str, Any]:
    returns = trace["factor_return"].to_numpy(dtype=float)
    active = trace["gross_exposure"].to_numpy(dtype=float) > 0.0
    rank_ic = trace["rank_ic"].to_numpy(dtype=float)
    turnover = trace["target_weight_turnover"].to_numpy(dtype=float)
    membership_turnover = trace["top20_membership_turnover"].to_numpy(dtype=float)
    quintile_means = [float(trace[f"q{i}_return"].mean()) for i in range(1, 6)]
    finite_ic = rank_ic[np.isfinite(rank_ic)]
    yearly = []
    for year, frame in trace.groupby(trace["signal_date"].dt.year, sort=True):
        yearly.append(
            {
                "year": int(year),
                "factor_return": summarize_daily_returns(
                    frame["factor_return"].to_numpy(dtype=float)
                ),
                "rank_ic": _series_summary(frame["rank_ic"].to_numpy(dtype=float)),
                "coverage_mean": float(frame["coverage"].mean()),
                "target_weight_turnover_mean": _finite_mean(
                    frame["target_weight_turnover"].to_numpy(dtype=float)
                ),
            }
        )
    abs_returns = np.abs(returns[np.isfinite(returns)])
    extreme_share = (
        float(np.sort(abs_returns)[-min(10, len(abs_returns)) :].sum() / abs_returns.sum())
        if len(abs_returns) and abs_returns.sum() > 0.0
        else None
    )
    return {
        "schema_version": "ashare-factor-daily-measurement-summary.v1",
        "engine_version": MEASUREMENT_ENGINE_VERSION,
        "promotion_authority": False,
        "measurement_book": "continuous_cross_sectional_rank_zero_net_unit_gross",
        "return_clock": "signal_t_to_open_t_plus_1_to_open_t_plus_2",
        "direction": direction,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
        "factor_return": summarize_daily_returns(returns),
        "active_day_factor_return": summarize_daily_returns(returns[active]),
        "rank_ic": _series_summary(finite_ic),
        "target_weight_turnover": _distribution_summary(turnover),
        "top20_membership_turnover": _distribution_summary(membership_turnover),
        "fixed_top20": summarize_daily_returns(
            trace["top20_return"].to_numpy(dtype=float)
        ),
        "fixed_bottom20": summarize_daily_returns(
            trace["bottom20_return"].to_numpy(dtype=float)
        ),
        "fixed_top_bottom": summarize_daily_returns(
            trace["top_bottom_return"].to_numpy(dtype=float)
        ),
        "quintile_mean_daily_returns": quintile_means,
        "quintile_monotonicity": _correlation(
            np.arange(1.0, 6.0), np.asarray(quintile_means)
        ),
        "intraday": summarize_daily_returns(
            trace["intraday_return"].to_numpy(dtype=float)
        ),
        "overnight": summarize_daily_returns(
            trace["overnight_return"].to_numpy(dtype=float)
        ),
        "horizon_curve": horizons,
        "horizon_diagnostics": _horizon_diagnostics(horizons),
        "yearly": yearly,
        "rolling": {
            str(window): _rolling_summary(returns, window) for window in rolling_windows
        },
        "coverage": {
            "mean": float(trace["coverage"].mean()),
            "minimum": float(trace["coverage"].min()),
            "calendar_days": int(len(trace)),
            "finite_calendar_return_days": int(np.isfinite(returns).sum()),
            "active_book_days": int(active.sum()),
        },
        "extreme_abs_return_top10_share": extreme_share,
        "weight_sha256": weight_hash,
    }


def _horizon_curve(
    factor: np.ndarray,
    context: ProductionExecutionContext,
    *,
    sign: float,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    opens = np.asarray(context.valuation_open, dtype=float)
    eligible = np.asarray(context.signal_eligible, dtype=bool)
    rows: list[dict[str, Any]] = []
    previous_horizon = 0
    for horizon in horizons:
        payoffs: list[float] = []
        increments: list[float] = []
        ics: list[float] = []
        counts: list[int] = []
        for signal_position in range(len(context.dates) - horizon - 1):
            entry = signal_position + 1
            exit_position = entry + horizon
            previous_exit = entry + previous_horizon
            scores = factor[signal_position]
            directed = np.where(
                eligible[signal_position] & np.isfinite(scores), sign * scores, np.nan
            )
            weights = continuous_rank_weights(directed)
            forward = _asset_return(opens[entry], opens[exit_position])
            increment = _asset_return(opens[previous_exit], opens[exit_position])
            payoffs.append(_weighted_return(weights, forward))
            increments.append(_weighted_return(weights, increment))
            ic, count = _rank_ic(directed, forward)
            ics.append(ic)
            counts.append(count)
        payoff = np.asarray(payoffs, dtype=float)
        incremental = np.asarray(increments, dtype=float)
        ic_array = np.asarray(ics, dtype=float)
        rows.append(
            {
                "horizon": horizon,
                "payoff_mean": _finite_mean(payoff),
                "payoff_hac_t_stat": newey_west_mean_tstat(payoff),
                "rank_ic_mean": _finite_mean(ic_array),
                "rank_ic_hac_t_stat": newey_west_mean_tstat(ic_array),
                "positive_payoff_ratio": _finite_positive_ratio(payoff),
                "average_cross_section_count": float(np.mean(counts)) if counts else 0.0,
                "increment_from_previous_horizon": [previous_horizon, horizon],
                "incremental_payoff_mean": _finite_mean(incremental),
                "incremental_payoff_hac_t_stat": newey_west_mean_tstat(incremental),
            }
        )
        previous_horizon = horizon
    return rows


def _horizon_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if row["payoff_mean"] is not None]
    if not finite:
        return {
            "strongest_absolute_horizon": None,
            "first_reversal_horizon": None,
            "first_nonpositive_increment_horizon": None,
        }
    first_sign = np.sign(float(finite[0]["payoff_mean"]))
    reversal = next(
        (
            int(row["horizon"])
            for row in finite[1:]
            if np.sign(float(row["payoff_mean"])) not in {0.0, first_sign}
        ),
        None,
    )
    nonpositive = next(
        (
            int(row["horizon"])
            for row in finite
            if row["incremental_payoff_mean"] is not None
            and float(row["incremental_payoff_mean"]) <= 0.0
        ),
        None,
    )
    strongest = max(finite, key=lambda row: abs(float(row["payoff_mean"])))
    return {
        "strongest_absolute_horizon": int(strongest["horizon"]),
        "first_reversal_horizon": reversal,
        "first_nonpositive_increment_horizon": nonpositive,
    }


def _rolling_summary(values: np.ndarray, window: int) -> dict[str, Any]:
    series = pd.Series(np.asarray(values, dtype=float))
    minimum = max(2, window)
    mean = series.rolling(window, min_periods=minimum).mean() * TRADING_DAYS_PER_YEAR
    volatility = series.rolling(window, min_periods=minimum).std(ddof=1)
    sharpe = mean / (volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
    return {
        "window": window,
        "observations": int(sharpe.notna().sum()),
        "annualized_mean_return": _series_distribution(mean),
        "annualized_sharpe": _series_distribution(sharpe),
    }


def _series_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "observations": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else None,
        "positive_ratio": float(np.mean(array > 0.0)) if len(array) else None,
        "hac_t_stat": newey_west_mean_tstat(array),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "observations": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p90": float(np.quantile(array, 0.90)) if len(array) else None,
        "zero_ratio": float(np.mean(array == 0.0)) if len(array) else None,
    }


def _series_distribution(series: pd.Series) -> dict[str, Any]:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return {
        "minimum": float(np.min(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
        "latest": float(values[-1]) if len(values) else None,
    }


def _asset_return(entry: np.ndarray, exit_price: np.ndarray) -> np.ndarray:
    entry = np.asarray(entry, dtype=float)
    exit_price = np.asarray(exit_price, dtype=float)
    valid = np.isfinite(entry) & np.isfinite(exit_price) & (entry > 0.0) & (exit_price > 0.0)
    output = np.full(entry.shape, np.nan, dtype=float)
    output[valid] = exit_price[valid] / entry[valid] - 1.0
    return output


def _weighted_return(weights: np.ndarray, returns: np.ndarray) -> float:
    held = np.abs(weights) > 0.0
    if not held.any():
        return 0.0
    if not np.isfinite(returns[held]).all():
        return np.nan
    return float(np.dot(weights[held], returns[held]))


def _rank_ic(scores: np.ndarray, returns: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(scores) & np.isfinite(returns)
    count = int(valid.sum())
    if count < 2:
        return np.nan, count
    score_rank = _rankdata(scores[valid])
    return_rank = _rankdata(returns[valid])
    if np.std(score_rank) == 0.0 or np.std(return_rank) == 0.0:
        return np.nan, count
    return float(np.corrcoef(score_rank, return_rank)[0, 1]), count


def _quantile_groups(scores: np.ndarray, count: int) -> list[np.ndarray]:
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) < count:
        return []
    order = valid[np.argsort(scores[valid], kind="stable")]
    return [np.asarray(group, dtype=int) for group in np.array_split(order, count)]


def _group_mean(returns: np.ndarray, indexes: np.ndarray) -> float:
    if len(indexes) == 0:
        return np.nan
    values = returns[indexes]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _jaccard_turnover(previous: set[int], current: set[int]) -> float:
    union = previous | current
    return float(1.0 - len(previous & current) / len(union)) if union else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2 or np.std(right[valid]) == 0.0:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _finite_mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def _finite_positive_ratio(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array > 0.0)) if len(array) else None


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _positive_unique(values: Sequence[int], label: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError(f"{label} must contain positive integers")
    if len(parsed) != len(set(parsed)) or parsed != tuple(sorted(parsed)):
        raise ValueError(f"{label} must be unique and increasing")
    return parsed


def _empty_return_summary(observations: int) -> dict[str, Any]:
    return {
        "observations": int(observations),
        "mean_daily_return": None,
        "annualized_mean_return": None,
        "annualized_volatility": None,
        "annualized_sharpe": None,
        "maximum_drawdown": None,
        "positive_day_ratio": None,
        "terminal_growth": None,
        "hac_t_stat": None,
    }
