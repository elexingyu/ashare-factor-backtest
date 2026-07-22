"""Numerically safe element-wise arithmetic operators."""

from __future__ import annotations

import numpy as np
import pandas as pd


Number = pd.DataFrame | int | float


def add(x: Number, y: Number) -> Number:
    _aligned(x, y)
    return x + y


def sub(x: Number, y: Number) -> Number:
    _aligned(x, y)
    return x - y


def mul(x: Number, y: Number) -> Number:
    _aligned(x, y)
    return x * y


def div(x: Number, y: Number) -> Number:
    _aligned(x, y)
    if isinstance(y, pd.DataFrame):
        return x / y.where(y.abs() > 1e-12)
    if abs(y) <= 1e-12:
        return x * np.nan
    return x / y


def neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


def abs_value(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def signed_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.log1p(x.abs())


def signed_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.sqrt(x.abs())


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def signed_power(x: pd.DataFrame, exponent: int | float) -> pd.DataFrame:
    if exponent not in {0.5, 1.5, 2, 3}:
        raise ValueError("signed_power exponent is outside the frozen domain")
    return np.sign(x) * np.power(x.abs(), exponent)


def panel_min(x: Number, y: Number) -> pd.DataFrame:
    return _elementwise_extreme(np.minimum, x, y)


def panel_max(x: Number, y: Number) -> pd.DataFrame:
    return _elementwise_extreme(np.maximum, x, y)


def _aligned(x: Number, y: Number) -> None:
    if (
        isinstance(x, pd.DataFrame)
        and isinstance(y, pd.DataFrame)
        and (not x.index.equals(y.index) or not x.columns.equals(y.columns))
    ):
        raise ValueError("arithmetic panels must be exactly aligned")


def _elementwise_extreme(function, x: Number, y: Number) -> pd.DataFrame:
    _aligned(x, y)
    template = x if isinstance(x, pd.DataFrame) else y
    if not isinstance(template, pd.DataFrame):
        raise TypeError("elementwise extrema require at least one panel")
    result = function(x, y)
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(result, index=template.index, columns=template.columns)
