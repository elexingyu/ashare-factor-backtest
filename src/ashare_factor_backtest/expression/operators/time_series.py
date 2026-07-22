"""Backward-looking single-panel time-series operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


def ts_delay(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.shift(window)


def ts_delta(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x - x.shift(window)


def ts_pct_change(x: pd.DataFrame, window: int) -> pd.DataFrame:
    delayed = x.shift(window)
    return x / delayed.where(delayed.abs() > 1e-12) - 1.0


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).sum()


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).std(ddof=1)


def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_min_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_max_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_min_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_extreme_float64(np.ascontiguousarray(values.T), window, True).T


def ts_max_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_extreme_float64(np.ascontiguousarray(values.T), window, False).T


@njit(cache=True, parallel=True)
def _rolling_extreme_float64(
    values: np.ndarray, window: int, find_minimum: bool
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        indexes = np.empty(row_count, dtype=np.int64)
        head = 0
        tail = 0
        missing_count = 0
        for row in range(row_count):
            expired = row - window
            if expired >= 0 and np.isnan(values[column, expired]):
                missing_count -= 1
            while head < tail and indexes[head] <= expired:
                head += 1

            value = values[column, row]
            if np.isnan(value):
                missing_count += 1
            else:
                if find_minimum:
                    while head < tail and values[column, indexes[tail - 1]] >= value:
                        tail -= 1
                else:
                    while head < tail and values[column, indexes[tail - 1]] <= value:
                        tail -= 1
                indexes[tail] = row
                tail += 1

            if row >= window - 1 and missing_count == 0:
                output[column, row] = values[column, indexes[head]]
    return output


def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_rank_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_rank_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_current_rank_float64(np.ascontiguousarray(values.T), window).T


@njit(cache=True, parallel=True)
def _rolling_current_rank_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    if window == 1:
        for column in prange(column_count):
            for row in range(row_count):
                if not np.isnan(values[column, row]):
                    output[column, row] = 0.0
        return output
    denominator = window - 1.0
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            current = values[column, row]
            if np.isnan(current):
                continue
            less_count = 0
            equal_count = 0
            complete = True
            for offset in range(window):
                value = values[column, row - offset]
                if np.isnan(value):
                    complete = False
                    break
                if value < current:
                    less_count += 1
                elif value == current:
                    equal_count += 1
            if complete:
                output[column, row] = (
                    less_count + (equal_count - 1.0) / 2.0
                ) / denominator
    return output


def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(
        lambda values: _recent_extreme(values, "min"), raw=True
    )


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(
        lambda values: _recent_extreme(values, "max"), raw=True
    )


def ts_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = ts_mean(x, window)
    std = ts_std(x, window)
    result = (x - mean) / std.where(std.abs() > 1e-12)
    return result.mask(std.eq(0) & x.notna(), 0.0)


def ts_ema(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.ewm(span=window, adjust=False, min_periods=window).mean()


def ts_decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1.0, window + 1.0)
    weights /= weights.sum()
    return x.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights)), raw=True
    )


def ts_scale(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_scale_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_scale_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_scale_float64(np.ascontiguousarray(values.T), window).T


@njit(cache=True, parallel=True)
def _rolling_scale_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        minimum_indexes = np.empty(row_count, dtype=np.int64)
        maximum_indexes = np.empty(row_count, dtype=np.int64)
        minimum_head = 0
        minimum_tail = 0
        maximum_head = 0
        maximum_tail = 0
        missing_count = 0
        for row in range(row_count):
            expired = row - window
            if expired >= 0 and np.isnan(values[column, expired]):
                missing_count -= 1
            while (
                minimum_head < minimum_tail and minimum_indexes[minimum_head] <= expired
            ):
                minimum_head += 1
            while (
                maximum_head < maximum_tail and maximum_indexes[maximum_head] <= expired
            ):
                maximum_head += 1

            value = values[column, row]
            if np.isnan(value):
                missing_count += 1
            else:
                while (
                    minimum_head < minimum_tail
                    and values[column, minimum_indexes[minimum_tail - 1]] >= value
                ):
                    minimum_tail -= 1
                minimum_indexes[minimum_tail] = row
                minimum_tail += 1
                while (
                    maximum_head < maximum_tail
                    and values[column, maximum_indexes[maximum_tail - 1]] <= value
                ):
                    maximum_tail -= 1
                maximum_indexes[maximum_tail] = row
                maximum_tail += 1

            if row < window - 1 or missing_count:
                continue
            minimum = values[column, minimum_indexes[minimum_head]]
            maximum = values[column, maximum_indexes[maximum_head]]
            span = maximum - minimum
            if abs(span) > 1e-12:
                output[column, row] = (value - minimum) / span
    return output


def ts_skew(x: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling = x.rolling(window, min_periods=window)
    result = rolling.skew()
    threshold = (
        x.abs().rolling(window, min_periods=window).max().clip(lower=1.0) * 1e-12
    )
    return result.mask(rolling.std(ddof=1).le(threshold))


def ts_kurt(x: pd.DataFrame, window: int) -> pd.DataFrame:
    rolling = x.rolling(window, min_periods=window)
    result = rolling.kurt()
    threshold = (
        x.abs().rolling(window, min_periods=window).max().clip(lower=1.0) * 1e-12
    )
    return result.mask(rolling.std(ddof=1).le(threshold))


def ts_slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
    time = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    time_sample_variance = window * (window + 1) / 12.0
    covariance = x.rolling(window, min_periods=window).cov(time)
    return covariance / time_sample_variance


def ts_r2(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_r2_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_r2_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_r2_float64(np.ascontiguousarray(values.T), window).T


@njit(cache=True, parallel=True)
def _rolling_r2_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    time_mean = (window - 1.0) / 2.0
    time_ss = window * (window * window - 1.0) / 12.0
    for column in prange(column_count):
        for end in range(window - 1, row_count):
            start = end - window + 1
            total = 0.0
            complete = True
            for offset in range(window):
                value = values[column, start + offset]
                if not np.isfinite(value):
                    complete = False
                    break
                total += value
            if not complete:
                continue
            mean = total / window
            covariance = 0.0
            value_ss = 0.0
            for offset in range(window):
                centered = values[column, start + offset] - mean
                covariance += (offset - time_mean) * centered
                value_ss += centered * centered
            if value_ss <= 0.0:
                continue
            r2 = covariance * covariance / (time_ss * value_ss)
            output[column, end] = min(1.0, max(0.0, r2))
    return output


def ts_product(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).apply(np.prod, raw=True)


def ts_path_efficiency(x: pd.DataFrame, window: int) -> pd.DataFrame:
    def efficiency(values: np.ndarray) -> float:
        changes = np.diff(values)
        path = float(np.abs(changes).sum())
        threshold = max(1.0, float(np.max(np.abs(values)))) * 1e-12
        if path <= threshold:
            return np.nan
        return float(abs(values[-1] - values[0]) / path)

    return x.rolling(window, min_periods=window).apply(efficiency, raw=True)


def ts_turning_rate(x: pd.DataFrame, window: int) -> pd.DataFrame:
    def turning_rate(values: np.ndarray) -> float:
        changes = np.diff(values)
        threshold = max(1.0, float(np.max(np.abs(values)))) * 1e-12
        signs = np.sign(changes[np.abs(changes) > threshold])
        if len(signs) < 2:
            return np.nan
        return float(np.count_nonzero(signs[1:] != signs[:-1]) / (len(signs) - 1))

    return x.rolling(window, min_periods=window).apply(turning_rate, raw=True)


def ts_signed_run_length(x: pd.DataFrame, window: int) -> pd.DataFrame:
    def signed_run(values: np.ndarray) -> float:
        changes = np.diff(values)
        threshold = max(1.0, float(np.max(np.abs(values)))) * 1e-12
        signs = np.where(np.abs(changes) > threshold, np.sign(changes), 0.0)
        current = signs[-1]
        if current == 0:
            return np.nan
        run = 0
        for sign in signs[::-1]:
            if sign != current:
                break
            run += 1
        return float(current * run / (window - 1))

    return x.rolling(window, min_periods=window).apply(signed_run, raw=True)


def _recent_extreme(values: np.ndarray, kind: str) -> float:
    reversed_values = values[::-1]
    extreme = np.min(reversed_values) if kind == "min" else np.max(reversed_values)
    tied = np.isclose(reversed_values, extreme, rtol=1e-12, atol=1e-12)
    return float(np.flatnonzero(tied)[0])
