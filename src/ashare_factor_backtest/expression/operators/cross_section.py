"""Same-date cross-sectional operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    ranks = x.rank(axis=1, method="average")
    return _normalize_ranks(ranks, x)


def cs_rank_stable(x: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    ranked = cs_rank_stable_array(values)
    return pd.DataFrame(ranked, index=x.index, columns=x.columns)


def cs_rank_stable_array(values: np.ndarray) -> np.ndarray:
    return _stable_rank_rows(np.asarray(values, dtype=float))


@njit(cache=True, parallel=True)
def average_rank_rows(values: np.ndarray) -> np.ndarray:
    """Return SciPy-compatible one-based average ranks, omitting NaNs by row."""
    row_count, column_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for row in prange(row_count):
        valid_count = 0
        for column in range(column_count):
            if not np.isnan(values[row, column]):
                valid_count += 1
        if valid_count == 0:
            continue
        row_values = np.empty(valid_count, dtype=np.float64)
        columns = np.empty(valid_count, dtype=np.int64)
        position = 0
        for column in range(column_count):
            value = values[row, column]
            if np.isnan(value):
                continue
            row_values[position] = value
            columns[position] = column
            position += 1
        order = np.argsort(row_values)
        start = 0
        while start < valid_count:
            end = start + 1
            tied_value = row_values[order[start]]
            while end < valid_count and row_values[order[end]] == tied_value:
                end += 1
            average_one_based_rank = (start + 1.0 + end) / 2.0
            for tied_position in range(start, end):
                output[row, columns[order[tied_position]]] = average_one_based_rank
            start = end
    return output


@njit(cache=True, parallel=True)
def _stable_rank_rows(values: np.ndarray) -> np.ndarray:
    row_count, column_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for row in prange(row_count):
        scale = 0.0
        valid_count = 0
        for column in range(column_count):
            value = values[row, column]
            if np.isnan(value):
                continue
            valid_count += 1
            if np.isfinite(value) and abs(value) > scale:
                scale = abs(value)
        if scale <= 0.0:
            scale = 1.0
        if valid_count == 0:
            continue
        if valid_count == 1:
            for column in range(column_count):
                if not np.isnan(values[row, column]):
                    output[row, column] = 0.5
                    break
            continue

        normalized = np.empty(valid_count, dtype=np.float64)
        columns = np.empty(valid_count, dtype=np.int64)
        position = 0
        for column in range(column_count):
            value = values[row, column]
            if np.isnan(value):
                continue
            normalized[position] = (
                np.round(value / scale, 12) if np.isfinite(value) else value
            )
            columns[position] = column
            position += 1

        order = np.argsort(normalized)
        start = 0
        while start < valid_count:
            end = start + 1
            tied_value = normalized[order[start]]
            while end < valid_count and normalized[order[end]] == tied_value:
                end += 1
            average_one_based_rank = (start + 1.0 + end) / 2.0
            rank = (average_one_based_rank - 1.0) / (valid_count - 1.0)
            for tied_position in range(start, end):
                output[row, columns[order[tied_position]]] = rank
            start = end
    return output


def _normalize_ranks(ranks: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    counts = source.notna().sum(axis=1)
    result = ranks.sub(1.0).div(counts.sub(1.0).replace(0, np.nan), axis=0)
    singleton_rows = counts.eq(1)
    if singleton_rows.any():
        result.loc[singleton_rows] = (
            source.loc[singleton_rows].notna().astype(float).replace(0.0, np.nan) * 0.5
        )
    return result


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    values = np.asarray(x.to_numpy(dtype=float, na_value=np.nan), order="C")
    result = cs_zscore_array(values)
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def cs_zscore_array(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float, order="C", copy=True)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, values, 0.0).sum(axis=1)
    mean = np.divide(
        totals,
        counts,
        out=np.full(len(values), np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        counts,
        out=np.full(len(values), np.nan, dtype=float),
        where=counts > 0,
    )
    std = np.sqrt(variance)
    scale = np.maximum(1.0, np.where(finite, np.abs(values), 0.0).max(axis=1))
    near_zero = np.isnan(std) | (std <= scale * 1e-12)
    denominator = np.where(near_zero, np.nan, std)
    result = (values - mean[:, None]) / denominator[:, None]
    result[near_zero[:, None] & finite] = 0.0
    result[~finite] = np.nan
    return result


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    values, mean, _, finite = _row_moments(x)
    result = values - mean[:, None]
    result[~finite] = np.nan
    return pd.DataFrame(result, index=x.index, columns=x.columns)


def cs_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    y_values = y.to_numpy(dtype=float, na_value=np.nan)
    x_values = x.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(y_values) & np.isfinite(x_values)
    counts = finite.sum(axis=1)
    y_mean = np.divide(
        np.where(finite, y_values, 0.0).sum(axis=1),
        counts,
        out=np.full(len(y_values), np.nan),
        where=counts > 0,
    )
    x_mean = np.divide(
        np.where(finite, x_values, 0.0).sum(axis=1),
        counts,
        out=np.full(len(x_values), np.nan),
        where=counts > 0,
    )
    y_centered = np.where(finite, y_values - y_mean[:, None], 0.0)
    x_centered = np.where(finite, x_values - x_mean[:, None], 0.0)
    covariance = (x_centered * y_centered).sum(axis=1)
    variance = np.square(x_centered).sum(axis=1)
    scale = np.maximum(1.0, np.where(finite, np.abs(x_values), 0.0).max(axis=1))
    usable = (counts >= 3) & (variance > counts * np.square(scale * 1e-12))
    beta = np.divide(
        covariance,
        variance,
        out=np.full(len(variance), np.nan),
        where=usable,
    )
    result = y_centered - beta[:, None] * x_centered
    result[~finite | ~usable[:, None]] = np.nan
    return pd.DataFrame(result, index=y.index, columns=y.columns)


def cs_winsorize(x: pd.DataFrame, quantile: float) -> pd.DataFrame:
    if quantile not in {0.01, 0.05}:
        raise ValueError("winsorize quantile must be 0.01 or 0.05")
    lower = x.quantile(quantile, axis=1)
    upper = x.quantile(1.0 - quantile, axis=1)
    return x.clip(lower=lower, upper=upper, axis=0)


def _row_moments(
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.array(
        x.to_numpy(dtype=float, na_value=np.nan), dtype=float, order="C", copy=True
    )
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, values, 0.0).sum(axis=1)
    mean = np.divide(
        totals,
        counts,
        out=np.full(len(values), np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(finite, values - mean[:, None], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        counts,
        out=np.full(len(values), np.nan, dtype=float),
        where=counts > 0,
    )
    return values, mean, np.sqrt(variance), finite
