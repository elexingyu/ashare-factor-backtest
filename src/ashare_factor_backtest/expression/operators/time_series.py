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


def ts_last_change(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _last_distinct_frame(x, window, operation=0)


def ts_last_pct_change(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _last_distinct_frame(x, window, operation=1)


def ts_days_since_change(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _last_distinct_frame(x, window, operation=2)


def _last_distinct_frame(
    x: pd.DataFrame,
    window: int,
    *,
    operation: int,
) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = _last_distinct_state_float64(
        np.ascontiguousarray(values.T),
        window,
        operation,
    ).T
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_last_change_array(values: np.ndarray, window: int) -> np.ndarray:
    return _last_distinct_state_float64(
        np.ascontiguousarray(values.T), window, 0
    ).T


def ts_last_pct_change_array(values: np.ndarray, window: int) -> np.ndarray:
    return _last_distinct_state_float64(
        np.ascontiguousarray(values.T), window, 1
    ).T


def ts_days_since_change_array(values: np.ndarray, window: int) -> np.ndarray:
    return _last_distinct_state_float64(
        np.ascontiguousarray(values.T), window, 2
    ).T


@njit(cache=True, parallel=True)
def _last_distinct_state_float64(
    values: np.ndarray,
    window: int,
    operation: int,
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    for column in prange(column_count):
        state_value = np.nan
        prior_state_value = np.nan
        prior_state_end = -1
        for row in range(row_count):
            current = values[column, row]
            if np.isnan(current):
                continue
            if np.isnan(state_value):
                state_value = current
                continue
            tolerance = max(abs(current), abs(state_value), 1.0) * 1e-8
            if abs(current - state_value) > tolerance:
                prior_state_value = state_value
                prior_state_end = row - 1
                state_value = current
            if row < window - 1 or prior_state_end < row - window + 1:
                continue
            if operation == 0:
                output[column, row] = current - prior_state_value
            elif operation == 1:
                if abs(prior_state_value) > 1e-12:
                    output[column, row] = current / prior_state_value - 1.0
            else:
                output[column, row] = row - prior_state_end - 1
    return output


def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).sum()


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).std(ddof=1)


def ts_sum_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_basic_stat_float64(
        np.ascontiguousarray(values.T), window, 0
    ).T


def ts_mean_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_basic_stat_float64(
        np.ascontiguousarray(values.T), window, 1
    ).T


def ts_std_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_basic_stat_float64(
        np.ascontiguousarray(values.T), window, 2
    ).T


def ts_zscore_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_basic_stat_float64(
        np.ascontiguousarray(values.T), window, 3
    ).T


@njit(cache=True, parallel=True)
def _rolling_basic_stat_float64(
    values: np.ndarray, window: int, operation: int
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            total = 0.0
            complete = True
            for position in range(start, row + 1):
                value = values[column, position]
                if np.isnan(value):
                    complete = False
                    break
                total += value
            if not complete:
                continue
            if operation == 0:
                output[column, row] = total
                continue
            mean = total / window
            if operation == 1:
                output[column, row] = mean
                continue
            if window < 2:
                continue
            squared_deviation = 0.0
            for position in range(start, row + 1):
                difference = values[column, position] - mean
                squared_deviation += difference * difference
            standard_deviation = np.sqrt(squared_deviation / (window - 1.0))
            if operation == 2:
                output[column, row] = standard_deviation
            elif standard_deviation == 0.0:
                output[column, row] = 0.0
            elif standard_deviation > 1e-12:
                output[column, row] = (
                    values[column, row] - mean
                ) / standard_deviation
    return output


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
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_argmin_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_argmax_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_argmin_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_recent_extreme_float64(
        np.ascontiguousarray(values.T), window, True
    ).T


def ts_argmax_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_recent_extreme_float64(
        np.ascontiguousarray(values.T), window, False
    ).T


@njit(cache=True, parallel=True)
def _rolling_recent_extreme_float64(
    values: np.ndarray, window: int, find_minimum: bool
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            extreme = values[column, start]
            if np.isnan(extreme):
                continue
            complete = True
            for position in range(start + 1, row + 1):
                value = values[column, position]
                if np.isnan(value):
                    complete = False
                    break
                if find_minimum:
                    if value < extreme:
                        extreme = value
                elif value > extreme:
                    extreme = value
            if not complete:
                continue
            tolerance = 1e-12 + 1e-12 * abs(extreme)
            for offset in range(window):
                value = values[column, row - offset]
                if value == extreme or abs(value - extreme) <= tolerance:
                    output[column, row] = float(offset)
                    break
    return output


def ts_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = ts_mean(x, window)
    std = ts_std(x, window)
    result = (x - mean) / std.where(std.abs() > 1e-12)
    return result.mask(std.eq(0) & x.notna(), 0.0)


def ts_skew_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_higher_moment_float64(
        np.ascontiguousarray(values.T), window, False
    ).T


def ts_kurt_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_higher_moment_float64(
        np.ascontiguousarray(values.T), window, True
    ).T


@njit(cache=True, parallel=True)
def _rolling_higher_moment_float64(
    values: np.ndarray, window: int, calculate_kurtosis: bool
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    minimum_window = 4 if calculate_kurtosis else 3
    if window < minimum_window or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            total = 0.0
            maximum_absolute = 1.0
            complete = True
            for position in range(start, row + 1):
                value = values[column, position]
                if np.isnan(value):
                    complete = False
                    break
                total += value
                maximum_absolute = max(maximum_absolute, abs(value))
            if not complete:
                continue
            mean = total / window
            second = 0.0
            third = 0.0
            fourth = 0.0
            for position in range(start, row + 1):
                difference = values[column, position] - mean
                difference_squared = difference * difference
                second += difference_squared
                if calculate_kurtosis:
                    fourth += difference_squared * difference_squared
                else:
                    third += difference_squared * difference
            standard_deviation = np.sqrt(second / (window - 1.0))
            if standard_deviation <= maximum_absolute * 1e-12:
                continue
            if calculate_kurtosis:
                standardized_fourth = fourth / (standard_deviation**4)
                output[column, row] = (
                    window
                    * (window + 1.0)
                    * standardized_fourth
                    / ((window - 1.0) * (window - 2.0) * (window - 3.0))
                    - 3.0
                    * (window - 1.0) ** 2
                    / ((window - 2.0) * (window - 3.0))
                )
            else:
                standardized_third = third / (standard_deviation**3)
                output[column, row] = (
                    window
                    * standardized_third
                    / ((window - 1.0) * (window - 2.0))
                )
    return output


def ts_ema(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.ewm(span=window, adjust=False, min_periods=window).mean()


def ts_decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_decay_linear_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_decay_linear_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_decay_linear_float64(
        np.ascontiguousarray(values.T), window
    ).T


@njit(cache=True, parallel=True)
def _rolling_decay_linear_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    denominator = window * (window + 1.0) / 2.0
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            weighted_sum = 0.0
            complete = True
            for offset in range(window):
                value = values[column, start + offset]
                if np.isnan(value):
                    complete = False
                    break
                weighted_sum += value * (offset + 1.0)
            if complete:
                output[column, row] = weighted_sum / denominator
    return output


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


def ts_slope_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_slope_float64(np.ascontiguousarray(values.T), window).T


@njit(cache=True, parallel=True)
def _rolling_slope_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    time_mean = (window - 1.0) / 2.0
    time_ss = window * (window * window - 1.0) / 12.0
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            total = 0.0
            weighted_total = 0.0
            complete = True
            for offset in range(window):
                value = values[column, start + offset]
                if np.isnan(value):
                    complete = False
                    break
                total += value
                weighted_total += offset * value
            if complete:
                output[column, row] = (
                    weighted_total - time_mean * total
                ) / time_ss
    return output
    return output


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
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = ts_product_array(values, window)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def ts_product_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_product_float64(np.ascontiguousarray(values.T), window).T


@njit(cache=True, parallel=True)
def _rolling_product_float64(values: np.ndarray, window: int) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            product = 1.0
            complete = True
            for position in range(start, row + 1):
                value = values[column, position]
                if np.isnan(value):
                    complete = False
                    break
                product *= value
            if complete:
                output[column, row] = product
    return output


def ts_path_efficiency(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = ts_path_efficiency_array(x.to_numpy(dtype=float), window)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def ts_path_efficiency_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_path_operator_float64(
        np.ascontiguousarray(values.T), window, 0
    ).T


def ts_turning_rate(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = ts_turning_rate_array(x.to_numpy(dtype=float), window)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def ts_turning_rate_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_path_operator_float64(
        np.ascontiguousarray(values.T), window, 1
    ).T


def ts_signed_run_length(x: pd.DataFrame, window: int) -> pd.DataFrame:
    values = ts_signed_run_length_array(x.to_numpy(dtype=float), window)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def ts_signed_run_length_array(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_path_operator_float64(
        np.ascontiguousarray(values.T), window, 2
    ).T


@njit(cache=True, parallel=True)
def _rolling_path_operator_float64(
    values: np.ndarray,
    window: int,
    operation: int,
) -> np.ndarray:
    column_count, row_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if window < 2 or window > row_count:
        return output
    for column in prange(column_count):
        for row in range(window - 1, row_count):
            start = row - window + 1
            maximum_absolute = 1.0
            complete = True
            for position in range(start, row + 1):
                value = values[column, position]
                if not np.isfinite(value):
                    complete = False
                    break
                maximum_absolute = max(maximum_absolute, abs(value))
            if not complete:
                continue
            threshold = maximum_absolute * 1e-12
            if operation == 0:
                path = 0.0
                for position in range(start + 1, row + 1):
                    path += abs(
                        values[column, position]
                        - values[column, position - 1]
                    )
                if path > threshold:
                    output[column, row] = abs(
                        values[column, row] - values[column, start]
                    ) / path
                continue
            if operation == 1:
                prior_sign = 0
                significant_count = 0
                turning_count = 0
                for position in range(start + 1, row + 1):
                    difference = (
                        values[column, position]
                        - values[column, position - 1]
                    )
                    if abs(difference) <= threshold:
                        continue
                    sign = 1 if difference > 0.0 else -1
                    if significant_count and sign != prior_sign:
                        turning_count += 1
                    prior_sign = sign
                    significant_count += 1
                if significant_count >= 2:
                    output[column, row] = turning_count / (
                        significant_count - 1.0
                    )
                continue
            last_difference = (
                values[column, row] - values[column, row - 1]
            )
            if abs(last_difference) <= threshold:
                continue
            current_sign = 1 if last_difference > 0.0 else -1
            run = 0
            for position in range(row, start, -1):
                difference = (
                    values[column, position]
                    - values[column, position - 1]
                )
                sign = 0
                if difference > threshold:
                    sign = 1
                elif difference < -threshold:
                    sign = -1
                if sign != current_sign:
                    break
                run += 1
            output[column, row] = current_sign * run / (window - 1.0)
    return output


def _recent_extreme(values: np.ndarray, kind: str) -> float:
    reversed_values = values[::-1]
    extreme = np.min(reversed_values) if kind == "min" else np.max(reversed_values)
    tied = np.isclose(reversed_values, extreme, rtol=1e-12, atol=1e-12)
    return float(np.flatnonzero(tied)[0])
