"""Backward-looking pairwise time-series operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


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


def ts_cov_array(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    return _rolling_pairwise_stat_float64(
        np.ascontiguousarray(x.T),
        np.ascontiguousarray(y.T),
        window,
        0,
    ).T


def ts_corr_array(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    return _rolling_pairwise_stat_float64(
        np.ascontiguousarray(x.T),
        np.ascontiguousarray(y.T),
        window,
        1,
    ).T


def ts_beta_array(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    return _rolling_pairwise_stat_float64(
        np.ascontiguousarray(x.T),
        np.ascontiguousarray(y.T),
        window,
        2,
    ).T


@njit(cache=True, parallel=True)
def _rolling_pairwise_stat_float64(
    x: np.ndarray, y: np.ndarray, window: int, operation: int
) -> np.ndarray:
    column_count, row_count = x.shape
    output = np.full(x.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            x_total = 0.0
            y_total = 0.0
            complete = True
            for position in range(start, row + 1):
                x_value = x[column, position]
                y_value = y[column, position]
                if np.isnan(x_value) or np.isnan(y_value):
                    complete = False
                    break
                x_total += x_value
                y_total += y_value
            if not complete:
                continue
            x_mean = x_total / window
            y_mean = y_total / window
            covariance_numerator = 0.0
            x_squared_deviation = 0.0
            y_squared_deviation = 0.0
            for position in range(start, row + 1):
                x_difference = x[column, position] - x_mean
                y_difference = y[column, position] - y_mean
                covariance_numerator += x_difference * y_difference
                if operation == 1:
                    x_squared_deviation += x_difference * x_difference
                if operation in (1, 2):
                    y_squared_deviation += y_difference * y_difference
            covariance = covariance_numerator / (window - 1.0)
            if operation == 0:
                output[column, row] = covariance
            elif operation == 1:
                x_standard_deviation = np.sqrt(
                    x_squared_deviation / (window - 1.0)
                )
                y_standard_deviation = np.sqrt(
                    y_squared_deviation / (window - 1.0)
                )
                if (
                    x_standard_deviation > 1e-12
                    and y_standard_deviation > 1e-12
                ):
                    correlation = covariance / (
                        x_standard_deviation * y_standard_deviation
                    )
                    output[column, row] = min(1.0, max(-1.0, correlation))
            else:
                variance = y_squared_deviation / (window - 1.0)
                if abs(variance) > 1e-12:
                    output[column, row] = covariance / variance
    return output


def _aligned(x: pd.DataFrame, y: pd.DataFrame) -> None:
    if not x.index.equals(y.index) or not x.columns.equals(y.columns):
        raise ValueError("pairwise panels must be exactly aligned")
