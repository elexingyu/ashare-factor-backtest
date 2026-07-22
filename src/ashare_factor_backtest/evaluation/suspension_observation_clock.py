"""No-return primitives for auditing suspension observation-clock contracts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_PRICE_FIELDS = frozenset({"open", "high", "low", "close"})
_ACTIVITY_FIELDS = frozenset({"volume", "amount"})


def classify_member_observations(
    members: pd.DataFrame,
    bars: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Classify each PIT member-day without treating unknown state as suspension."""
    keys = ["ts_code", "trade_date"]
    for label, frame, required in (
        ("members", members, set(keys)),
        ("bars", bars, set(keys)),
        ("states", states, {*keys, "is_suspended"}),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} contains duplicate member-day keys")
    work = members.loc[:, keys].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="raise")
    bar_keys = bars.loc[:, keys].copy()
    bar_keys["trade_date"] = pd.to_datetime(bar_keys["trade_date"], errors="raise")
    bar_keys["_bar_present"] = True
    state_values = states.loc[:, [*keys, "is_suspended"]].copy()
    state_values["trade_date"] = pd.to_datetime(
        state_values["trade_date"], errors="raise"
    )
    work = work.merge(bar_keys, on=keys, how="left", validate="one_to_one").merge(
        state_values, on=keys, how="left", validate="one_to_one"
    )
    bar_present = work["_bar_present"].eq(True)
    state_true = work["is_suspended"].eq(True)
    state_false = work["is_suspended"].eq(False)
    labels = np.full(len(work), "unknown_state_missing_bar", dtype=object)
    labels[bar_present & state_false] = "observed_trading"
    labels[bar_present & ~state_false & ~state_true] = "observed_state_unknown"
    labels[bar_present & state_true] = "confirmed_suspension_with_bar"
    labels[~bar_present & state_true] = "confirmed_suspension_no_bar"
    labels[~bar_present & state_false] = "confirmed_non_suspension_missing_bar"
    work["observation_state"] = labels
    return work.drop(columns="_bar_present")


def apply_confirmed_suspension_fill(
    fields: Mapping[str, pd.DataFrame],
    suspension: pd.DataFrame,
    *,
    initial_prices: Mapping[str, pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fill only missing values on rows explicitly confirmed as suspended."""
    if not fields:
        raise ValueError("suspension fill requires at least one field")
    reference = next(iter(fields.values()))
    _require_same_axes(reference, suspension)
    confirmed = suspension.eq(True).to_numpy(dtype=bool, copy=False)  # noqa: E712
    seeds = dict(initial_prices or {})
    unknown_seeds = set(seeds).difference(_PRICE_FIELDS)
    if unknown_seeds:
        raise ValueError(f"unsupported initial price fields: {sorted(unknown_seeds)}")
    result: dict[str, pd.DataFrame] = {}
    for name, panel in fields.items():
        _require_same_axes(reference, panel)
        if name in _PRICE_FIELDS:
            values = panel.to_numpy(dtype=float, copy=True)
            initial = seeds.get(name)
            if initial is not None:
                if not initial.index.is_unique:
                    raise ValueError("initial price state columns must be unique")
                initial_values = initial.reindex(panel.columns).to_numpy(dtype=float)
            else:
                initial_values = None
            result[name] = pd.DataFrame(
                _carry_prices_with_barriers(values, confirmed, initial_values),
                index=panel.index,
                columns=panel.columns,
            )
        elif name in _ACTIVITY_FIELDS:
            filled = panel.copy()
            filled_values = filled.to_numpy(dtype=float, copy=True)
            fillable = confirmed & ~np.isfinite(filled_values)
            filled_values[fillable] = 0.0
            result[name] = pd.DataFrame(
                filled_values, index=panel.index, columns=panel.columns
            )
        else:
            raise ValueError(f"unsupported suspension-fill field: {name}")
    return result


def fixed_calendar_rolling_mean(
    panel: pd.DataFrame, *, window: int, minimum_observations: int
) -> pd.DataFrame:
    _validate_rolling_contract(window, minimum_observations)
    return panel.rolling(window, min_periods=minimum_observations).mean()


def fixed_calendar_rolling_slope(
    panel: pd.DataFrame, *, window: int, minimum_observations: int
) -> pd.DataFrame:
    """Slope against original positions inside a fixed market-calendar window."""
    _validate_rolling_contract(window, minimum_observations)

    def slope(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        positions = np.flatnonzero(valid).astype(float)
        observed = values[valid]
        if len(observed) < max(2, minimum_observations):
            return np.nan
        centered_positions = positions - positions.mean()
        denominator = float(np.dot(centered_positions, centered_positions))
        if denominator <= 0:
            return np.nan
        return float(
            np.dot(centered_positions, observed - observed.mean()) / denominator
        )

    return panel.rolling(window, min_periods=minimum_observations).apply(
        slope, raw=True
    )


def compressed_rolling_mean(panel: pd.DataFrame, *, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    columns: dict[object, pd.Series] = {}
    for name in panel.columns:
        observed = panel[name].dropna()
        columns[name] = (
            observed.rolling(window, min_periods=window).mean().reindex(panel.index)
        )
    return pd.DataFrame(columns, index=panel.index)


def _carry_prices_with_barriers(
    values: np.ndarray,
    confirmed_suspension: np.ndarray,
    initial_values: np.ndarray | None = None,
) -> np.ndarray:
    if initial_values is not None:
        if initial_values.shape != (values.shape[1],):
            raise ValueError("initial price state shape mismatch")
        seeded_values = np.vstack((initial_values, values))
        seeded_suspension = np.vstack(
            (np.zeros((1, values.shape[1]), dtype=bool), confirmed_suspension)
        )
        return _carry_prices_with_barriers(seeded_values, seeded_suspension)[1:]
    rows, columns = values.shape
    row_ids = np.broadcast_to(np.arange(rows, dtype=np.int64)[:, None], (rows, columns))
    finite = np.isfinite(values)
    missing = ~finite
    barriers = missing & ~confirmed_suspension
    last_valid = np.maximum.accumulate(np.where(finite, row_ids, -1), axis=0)
    last_barrier = np.maximum.accumulate(np.where(barriers, row_ids, -1), axis=0)
    fillable = (
        missing & confirmed_suspension & (last_valid >= 0) & (last_valid > last_barrier)
    )
    safe_rows = np.maximum(last_valid, 0)
    source = values[safe_rows, np.broadcast_to(np.arange(columns), values.shape)]
    result = values.copy()
    result[fillable] = source[fillable]
    return result


def _validate_rolling_contract(window: int, minimum_observations: int) -> None:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    if not 1 <= minimum_observations <= window:
        raise ValueError("minimum observations must be within the rolling window")


def _require_same_axes(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        raise ValueError("suspension panels must have the same axes")
