"""Causal cross-sectional Rank IC for the production execution clock."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
)


RANK_IC_SEMANTICS = "signal_t_to_open_t_plus_1_to_t_plus_1_plus_h"


def evaluate_production_rank_ic(
    values: pd.DataFrame,
    context: ProductionExecutionContext,
    *,
    horizon: int,
    signal_start: str,
    signal_end: str,
) -> dict[str, Any]:
    """Evaluate daily Spearman IC without using entry-day information in the signal."""
    if horizon <= 0:
        raise ValueError("rank IC horizon must be positive")
    start = pd.Timestamp(signal_start)
    end = pd.Timestamp(signal_end)
    if start > end:
        raise ValueError("rank IC signal window is invalid")

    factor = values.reindex(index=context.dates, columns=context.codes).to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    prices = np.asarray(context.valuation_open, dtype=np.float64)
    daily_ics: list[float] = []
    cross_section_counts: list[int] = []

    for signal_position, signal_date in enumerate(context.dates):
        entry_position = signal_position + 1
        exit_position = entry_position + horizon
        if signal_date < start or signal_date > end or exit_position >= len(context.dates):
            continue
        if pd.Timestamp(context.dates[exit_position]) > end:
            continue
        signal = factor[signal_position]
        entry = prices[entry_position]
        exit_price = prices[exit_position]
        valid = (
            context.signal_eligible[signal_position]
            & np.isfinite(signal)
            & np.isfinite(entry)
            & np.isfinite(exit_price)
            & (entry > 0.0)
            & (exit_price > 0.0)
        )
        indexes = np.flatnonzero(valid)
        if len(indexes) < 2:
            continue
        forward_return = exit_price[indexes] / entry[indexes] - 1.0
        signal_series = pd.Series(signal[indexes])
        return_series = pd.Series(forward_return)
        if signal_series.nunique() < 2 or return_series.nunique() < 2:
            continue
        signal_rank = signal_series.rank(method="average").to_numpy(dtype=float)
        return_rank = return_series.rank(method="average").to_numpy(dtype=float)
        ic = float(np.corrcoef(signal_rank, return_rank)[0, 1])
        if np.isfinite(ic):
            daily_ics.append(ic)
            cross_section_counts.append(len(indexes))

    array = np.asarray(daily_ics, dtype=float)
    standard_deviation = (
        float(array.std(ddof=1)) if len(array) > 1 else 0.0
    )
    mean = float(array.mean()) if len(array) else None
    return {
        "semantics": RANK_IC_SEMANTICS,
        "horizon": horizon,
        "signal_window": [start.date().isoformat(), end.date().isoformat()],
        "observation_count": len(daily_ics),
        "rank_ic_mean": mean,
        "rank_ic_std": standard_deviation if len(array) else None,
        "rank_ic_ir": (
            float(mean / standard_deviation)
            if mean is not None and standard_deviation > 0.0
            else None
        ),
        "positive_rate": float((array > 0.0).mean()) if len(array) else None,
        "average_cross_section_count": (
            float(np.mean(cross_section_counts)) if cross_section_counts else 0.0
        ),
    }
