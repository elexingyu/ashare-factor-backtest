"""Backward-looking pairwise time-series operators."""

from __future__ import annotations

import pandas as pd


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    _aligned(x, y)
    x_rolling = x.rolling(window, min_periods=window)
    covariance = x_rolling.cov(y)
    x_std = x_rolling.std(ddof=1)
    y_std = y.rolling(window, min_periods=window).std(ddof=1)
    denominator = x_std * y_std
    return (covariance / denominator.where(
        (x_std.abs() > 1e-12) & (y_std.abs() > 1e-12)
    )).clip(-1.0, 1.0)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    _aligned(x, y)
    return x.rolling(window, min_periods=window).cov(y)


def ts_beta(x: pd.DataFrame, benchmark: pd.DataFrame, window: int) -> pd.DataFrame:
    _aligned(x, benchmark)
    rolling = benchmark.rolling(window, min_periods=window)
    covariance = x.rolling(window, min_periods=window).cov(benchmark)
    variance = rolling.var(ddof=1)
    return covariance / variance.where(variance.abs() > 1e-12)


def _aligned(x: pd.DataFrame, y: pd.DataFrame) -> None:
    if not x.index.equals(y.index) or not x.columns.equals(y.columns):
        raise ValueError("pairwise panels must be exactly aligned")
