"""Experimental causal time-series operators for the v9 catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import ndtri

from ashare_factor_backtest.expression.operators.time_series import ts_rank_array


def ts_median(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _frame_from_unary_array(x, ts_median_array(x.to_numpy(dtype=float), window))


def ts_median_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_robust_stat_float64(
        np.ascontiguousarray(values.T), window, 0
    ).T


def ts_mean_abs_dev(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _frame_from_unary_array(
        x, ts_mean_abs_dev_array(x.to_numpy(dtype=float), window)
    )


def ts_mean_abs_dev_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_robust_stat_float64(
        np.ascontiguousarray(values.T), window, 1
    ).T


@njit(cache=True, parallel=True)
def _rolling_robust_stat_float64(
    values: np.ndarray,
    window: int,
    operation: int,
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        sample = np.empty(window, dtype=np.float64)
        for row in range(window - 1, row_count):
            start = row - window + 1
            total = 0.0
            complete = True
            for offset in range(window):
                value = values[column, start + offset]
                if not np.isfinite(value):
                    complete = False
                    break
                sample[offset] = value
                total += value
            if not complete:
                continue
            if operation == 0:
                ordered = np.sort(sample)
                middle = window // 2
                if window % 2:
                    output[column, row] = ordered[middle]
                else:
                    output[column, row] = (
                        ordered[middle - 1] + ordered[middle]
                    ) / 2.0
                continue
            mean = total / window
            deviation = 0.0
            for offset in range(window):
                deviation += abs(sample[offset] - mean)
            output[column, row] = deviation / window
    return output


def ts_gaussianize(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _frame_from_unary_array(
        x, ts_gaussianize_array(x.to_numpy(dtype=float), window)
    )


def ts_gaussianize_array(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.full(np.asarray(values).shape, np.nan, dtype=float)
    ranks = ts_rank_array(np.asarray(values, dtype=float), window)
    lower = 0.5 / window
    probabilities = np.clip(ranks, lower, 1.0 - lower)
    return np.asarray(ndtri(probabilities), dtype=float)


def ts_regression_residual(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    values = ts_regression_residual_array(
        y.to_numpy(dtype=float),
        x.to_numpy(dtype=float),
        window,
    )
    return pd.DataFrame(values, index=y.index, columns=y.columns)


def ts_regression_residual_array(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    if y.shape != x.shape:
        raise ValueError("regression panels must have matching shapes")
    return _rolling_regression_residual_float64(
        np.ascontiguousarray(y.T),
        np.ascontiguousarray(x.T),
        window,
    ).T


@njit(cache=True, parallel=True)
def _rolling_regression_residual_float64(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
) -> np.ndarray:
    column_count, row_count = y.shape
    output = np.full(y.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            x_total = 0.0
            y_total = 0.0
            complete = True
            maximum_absolute_x = 1.0
            for position in range(start, row + 1):
                x_value = x[column, position]
                y_value = y[column, position]
                if not np.isfinite(x_value) or not np.isfinite(y_value):
                    complete = False
                    break
                x_total += x_value
                y_total += y_value
                maximum_absolute_x = max(maximum_absolute_x, abs(x_value))
            if not complete:
                continue
            x_mean = x_total / window
            y_mean = y_total / window
            x_ss = 0.0
            xy_ss = 0.0
            for position in range(start, row + 1):
                x_centered = x[column, position] - x_mean
                x_ss += x_centered * x_centered
                xy_ss += x_centered * (y[column, position] - y_mean)
            if x_ss <= maximum_absolute_x * maximum_absolute_x * 1e-24:
                continue
            beta = xy_ss / x_ss
            output[column, row] = (
                y[column, row] - y_mean - beta * (x[column, row] - x_mean)
            )
    return output


def ts_longest_signed_run(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _frame_from_unary_array(
        x, ts_longest_signed_run_array(x.to_numpy(dtype=float), window)
    )


def ts_longest_signed_run_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_path_stat_float64(
        np.ascontiguousarray(values.T), window, 0
    ).T


def ts_change_rate(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _frame_from_unary_array(
        x, ts_change_rate_array(x.to_numpy(dtype=float), window)
    )


def ts_change_rate_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_path_stat_float64(
        np.ascontiguousarray(values.T), window, 1
    ).T


@njit(cache=True, parallel=True)
def _rolling_path_stat_float64(
    values: np.ndarray,
    window: int,
    operation: int,
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    denominator = window - 1.0
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            complete = True
            changed_count = 0
            longest_run = 0
            longest_sign = 0
            current_run = 0
            current_sign = 0
            for position in range(start + 1, row + 1):
                left = values[column, position - 1]
                right = values[column, position]
                if not np.isfinite(left) or not np.isfinite(right):
                    complete = False
                    break
                tolerance = max(abs(left), abs(right), 1.0) * 1e-8
                difference = right - left
                sign = 0
                if difference > tolerance:
                    sign = 1
                elif difference < -tolerance:
                    sign = -1
                if sign:
                    changed_count += 1
                if sign == 0:
                    current_run = 0
                    current_sign = 0
                elif sign == current_sign:
                    current_run += 1
                else:
                    current_sign = sign
                    current_run = 1
                if current_run >= longest_run:
                    longest_run = current_run
                    longest_sign = current_sign
            if not complete:
                continue
            if operation == 1:
                output[column, row] = changed_count / denominator
            elif longest_run:
                output[column, row] = longest_sign * longest_run / denominator
    return output


def _frame_from_unary_array(
    template: pd.DataFrame,
    values: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns)
