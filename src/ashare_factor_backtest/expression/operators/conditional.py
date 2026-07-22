"""Nullable comparison and conditional operators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Number = pd.DataFrame | int | float


def gt(x: Number, y: Number) -> pd.DataFrame:
    return _compare(x, y, np.greater)


def ge(x: Number, y: Number) -> pd.DataFrame:
    return _compare(x, y, np.greater_equal)


def lt(x: Number, y: Number) -> pd.DataFrame:
    return _compare(x, y, np.less)


def le(x: Number, y: Number) -> pd.DataFrame:
    return _compare(x, y, np.less_equal)


def where(
    condition: pd.DataFrame,
    x: Number,
    y: Number,
) -> pd.DataFrame:
    if not isinstance(condition, pd.DataFrame):
        raise TypeError("where condition must be a DataFrame")
    template = condition
    x_values = _as_array_or_scalar(x, template)
    y_values = _as_array_or_scalar(y, template)
    condition_values = condition.to_numpy(dtype=bool, na_value=False)
    result = np.asarray(np.where(condition_values, x_values, y_values), dtype=float)
    result[condition.isna().to_numpy()] = np.nan
    return pd.DataFrame(result, index=template.index, columns=template.columns)


def trade_when(
    entry: pd.DataFrame,
    alpha: pd.DataFrame,
    exit_: pd.DataFrame,
) -> pd.DataFrame:
    """Causally update, clear, or carry one state per panel column."""
    if not all(isinstance(panel, pd.DataFrame) for panel in (entry, alpha, exit_)):
        raise TypeError("trade_when arguments must be DataFrames")
    if not (
        entry.index.equals(alpha.index)
        and entry.columns.equals(alpha.columns)
        and exit_.index.equals(alpha.index)
        and exit_.columns.equals(alpha.columns)
    ):
        raise ValueError("trade_when panels must be exactly aligned")
    enter = entry.astype("boolean").to_numpy(dtype=bool, na_value=False)
    leave = exit_.astype("boolean").to_numpy(dtype=bool, na_value=False)
    values = alpha.to_numpy(dtype=float, na_value=np.nan)
    state = np.full(values.shape[1], np.nan, dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    for row in range(values.shape[0]):
        update = enter[row]
        clear = leave[row] & ~update
        state[clear] = np.nan
        state[update] = values[row, update]
        output[row] = state
    return pd.DataFrame(output, index=alpha.index, columns=alpha.columns)


def _compare(
    x: Number,
    y: Number,
    operation: Callable[[object, object], object],
) -> pd.DataFrame:
    template = x if isinstance(x, pd.DataFrame) else y
    if not isinstance(template, pd.DataFrame):
        raise TypeError("comparison requires at least one panel")
    x_values = _as_array_or_scalar(x, template)
    y_values = _as_array_or_scalar(y, template)
    result = pd.DataFrame(
        operation(x_values, y_values),
        index=template.index,
        columns=template.columns,
        dtype="boolean",
    )
    missing = _missing_mask(x_values, template.shape) | _missing_mask(
        y_values, template.shape
    )
    return result.mask(missing, pd.NA)


def _as_array_or_scalar(
    value: Number, template: pd.DataFrame
) -> np.ndarray | int | float:
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(template.index) or not value.columns.equals(
            template.columns
        ):
            raise ValueError("conditional panels must be exactly aligned")
        return value.to_numpy(dtype=float, na_value=np.nan)
    return value


def _missing_mask(
    value: np.ndarray | int | float, shape: tuple[int, int]
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.isnan(value)
    return np.full(shape, pd.isna(value), dtype=bool)


def _as_panel(value: Number, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if not value.index.equals(template.index) or not value.columns.equals(
            template.columns
        ):
            raise ValueError("conditional panels must be exactly aligned")
        return value
    return pd.DataFrame(
        value, index=template.index, columns=template.columns, dtype=float
    )
