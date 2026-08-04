"""Same-date, within-group cross-sectional operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


def group_demean(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    _require_aligned(values, groups)
    result = group_demean_array(_values(values), _values(groups))
    return pd.DataFrame(result, index=values.index, columns=values.columns)


def group_zscore(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    _require_aligned(values, groups)
    result = group_zscore_array(_values(values), _values(groups))
    return pd.DataFrame(result, index=values.index, columns=values.columns)


def group_rank(values: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    _require_aligned(values, groups)
    result = group_rank_array(_values(values), _values(groups))
    return pd.DataFrame(result, index=values.index, columns=values.columns)


@njit(cache=True, parallel=True)
def group_demean_array(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    row_count, column_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for row in prange(row_count):
        ids, sums, _, counts, group_count = _group_moments(
            values[row], groups[row]
        )
        for column in range(column_count):
            value = values[row, column]
            group = groups[row, column]
            if not np.isfinite(value) or not np.isfinite(group):
                continue
            position = _find_group(ids, group_count, group)
            output[row, column] = value - sums[position] / counts[position]
    return output


@njit(cache=True, parallel=True)
def group_zscore_array(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    row_count, column_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for row in prange(row_count):
        ids, sums, squares, counts, group_count = _group_moments(
            values[row], groups[row]
        )
        for column in range(column_count):
            value = values[row, column]
            group = groups[row, column]
            if not np.isfinite(value) or not np.isfinite(group):
                continue
            position = _find_group(ids, group_count, group)
            mean = sums[position] / counts[position]
            variance = max(0.0, squares[position] / counts[position] - mean * mean)
            scale = max(1.0, abs(mean), abs(value))
            std = np.sqrt(variance)
            output[row, column] = (
                0.0 if std <= scale * 1e-12 else (value - mean) / std
            )
    return output


@njit(cache=True, parallel=True)
def group_rank_array(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    row_count, column_count = values.shape
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for row in prange(row_count):
        ids = np.empty(column_count, dtype=np.float64)
        group_count = 0
        for column in range(column_count):
            value = values[row, column]
            group = groups[row, column]
            if not np.isfinite(value) or not np.isfinite(group):
                continue
            if _find_group(ids, group_count, group) < 0:
                ids[group_count] = group
                group_count += 1
        for group_position in range(group_count):
            group = ids[group_position]
            member_count = 0
            scale = 0.0
            for column in range(column_count):
                value = values[row, column]
                if np.isfinite(value) and groups[row, column] == group:
                    member_count += 1
                    scale = max(scale, abs(value))
            if member_count == 0:
                continue
            if member_count == 1:
                for column in range(column_count):
                    if (
                        np.isfinite(values[row, column])
                        and groups[row, column] == group
                    ):
                        output[row, column] = 0.5
                        break
                continue
            scale = max(scale, 1.0)
            normalized = np.empty(member_count, dtype=np.float64)
            columns = np.empty(member_count, dtype=np.int64)
            position = 0
            for column in range(column_count):
                value = values[row, column]
                if np.isfinite(value) and groups[row, column] == group:
                    normalized[position] = np.round(value / scale, 12)
                    columns[position] = column
                    position += 1
            order = np.argsort(normalized)
            start = 0
            while start < member_count:
                end = start + 1
                tied_value = normalized[order[start]]
                while end < member_count and normalized[order[end]] == tied_value:
                    end += 1
                average_rank = (start + 1.0 + end) / 2.0
                rank = (average_rank - 1.0) / (member_count - 1.0)
                for tied_position in range(start, end):
                    output[row, columns[order[tied_position]]] = rank
                start = end
    return output


@njit(cache=True)
def _group_moments(
    values: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    size = len(values)
    ids = np.empty(size, dtype=np.float64)
    sums = np.zeros(size, dtype=np.float64)
    squares = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.int64)
    group_count = 0
    for column in range(size):
        value = values[column]
        group = groups[column]
        if not np.isfinite(value) or not np.isfinite(group):
            continue
        position = _find_group(ids, group_count, group)
        if position < 0:
            position = group_count
            ids[position] = group
            group_count += 1
        sums[position] += value
        squares[position] += value * value
        counts[position] += 1
    return ids, sums, squares, counts, group_count


@njit(cache=True)
def _find_group(ids: np.ndarray, count: int, target: float) -> int:
    for position in range(count):
        if ids[position] == target:
            return position
    return -1


def _values(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        frame.to_numpy(dtype=float, na_value=np.nan),
        dtype=np.float64,
        order="C",
    )


def _require_aligned(values: pd.DataFrame, groups: pd.DataFrame) -> None:
    if not isinstance(values, pd.DataFrame) or not isinstance(groups, pd.DataFrame):
        raise TypeError("group operators require DataFrame panels")
    if not values.index.equals(groups.index) or not values.columns.equals(
        groups.columns
    ):
        raise ValueError("group operator panels must be exactly aligned")
