"""Frequency-explicit return metrics used by the public factor evaluator."""

from __future__ import annotations

import numpy as np


def compute_periodic_metrics(
    returns: np.ndarray,
    periods_per_year: float,
) -> dict[str, float | int]:
    """Compute return metrics without inferring frequency from bar duration."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("returns must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and greater than -1")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite")

    nav = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    total_return = float(nav[-1] - 1.0)
    annual_return = float(nav[-1] ** (periods_per_year / len(values)) - 1.0)
    period_std = float(values.std())
    if period_std < np.finfo(float).eps:
        period_std = 0.0
    annual_volatility = float(period_std * np.sqrt(periods_per_year))
    sharpe = (
        float(values.mean() / period_std * np.sqrt(periods_per_year))
        if period_std > 0
        else 0.0
    )
    downside = values[values < 0]
    downside_std = float(downside.std()) if len(downside) else 0.0
    sortino = (
        float(values.mean() / downside_std * np.sqrt(periods_per_year))
        if downside_std > 0
        else 0.0
    )
    peak = np.maximum.accumulate(nav)
    max_drawdown = float(((peak - nav) / peak).max())
    calmar = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "positive_period_rate": float((values > 0).mean()),
        "period_count": int(len(values)),
        "periods_per_year": float(periods_per_year),
    }
