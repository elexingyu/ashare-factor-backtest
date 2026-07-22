"""Compressed execution matrices shared by all production factor candidates."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from ashare_factor_backtest.evaluation.production_chunked_evaluator import FrameLoader
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


@dataclass(frozen=True)
class ProductionExecutionContext:
    dates: pd.DatetimeIndex
    codes: pd.Index
    valuation_open: np.ndarray
    buyable: np.ndarray
    sellable: np.ndarray
    signal_eligible: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.codes))
        for name in (
            "valuation_open",
            "buyable",
            "sellable",
            "signal_eligible",
        ):
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(f"production execution {name} shape mismatch")
        if self.dates.has_duplicates or not self.dates.is_monotonic_increasing:
            raise ValueError("production execution dates must be unique and sorted")
        if self.codes.has_duplicates or not self.codes.is_monotonic_increasing:
            raise ValueError("production execution codes must be unique and sorted")


def slice_production_execution_context(
    context: ProductionExecutionContext, *, start: str, end: str
) -> ProductionExecutionContext:
    mask = context.dates.to_series().between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
    positions = np.flatnonzero(mask)
    if len(positions) < 2:
        raise ValueError("production execution slice must contain at least two dates")
    if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
        raise ValueError("production execution slice must be contiguous")
    selected = slice(int(positions[0]), int(positions[-1]) + 1)
    return ProductionExecutionContext(
        dates=context.dates[selected],
        codes=context.codes,
        valuation_open=context.valuation_open[selected],
        buyable=context.buyable[selected],
        sellable=context.sellable[selected],
        signal_eligible=context.signal_eligible[selected],
    )


def build_production_execution_context(
    frame: pd.DataFrame,
    *,
    dates: Sequence[pd.Timestamp] | None = None,
    codes: Sequence[str] | None = None,
    price_storage_dtype: np.dtype | type = np.float64,
    eligibility_column: str = "signal_eligible",
) -> ProductionExecutionContext:
    panels = _execution_panels(frame, eligibility_column=eligibility_column)
    target_dates = (
        pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
        if dates is not None
        else panels["raw_open"].index.sort_values()
    )
    target_codes = (
        pd.Index(sorted(str(code) for code in codes))
        if codes is not None
        else panels["raw_open"].columns.sort_values()
    )
    aligned = {
        name: panel.reindex(index=target_dates, columns=target_codes)
        for name, panel in panels.items()
    }
    return _from_panels(
        target_dates,
        target_codes,
        aligned,
        price_storage_dtype=price_storage_dtype,
    )


def build_chunked_production_execution_context(
    *,
    chunks: Sequence[YearChunk],
    frame_loader: FrameLoader,
    price_storage_dtype: np.dtype | type = np.float64,
    eligibility_column: str = "signal_eligible",
) -> ProductionExecutionContext:
    if not chunks:
        raise ValueError("production execution chunks must not be empty")
    collected: dict[str, list[pd.DataFrame]] = {
        "raw_open": [],
        "buyable": [],
        "sellable": [],
        "eligible": [],
    }
    for position, chunk in enumerate(chunks, start=1):
        panels = _load_execution_panels(
            frame_loader,
            chunk.calculation_start,
            chunk.calculation_end,
            eligibility_column=eligibility_column,
        )
        mask = panels["raw_open"].index.to_series().between(
            chunk.calculation_start, chunk.calculation_end
        )
        for name, panel in panels.items():
            collected[name].append(panel.loc[mask].copy())
        print(
            f"[production-execution] chunk={position}/{len(chunks)} "
            f"year={chunk.year} dates={int(mask.sum())}",
            flush=True,
        )
        del panels
        gc.collect()
        pa.default_memory_pool().release_unused()

    combined: dict[str, pd.DataFrame] = {}
    for name, panels in collected.items():
        panel = pd.concat(panels, axis=0, join="outer").sort_index().sort_index(axis=1)
        if panel.index.duplicated().any():
            raise ValueError(f"production execution chunks duplicate dates in {name}")
        combined[name] = panel
    dates = combined["raw_open"].index
    codes = combined["raw_open"].columns
    return _from_panels(
        dates,
        codes,
        combined,
        price_storage_dtype=price_storage_dtype,
    )


def _load_execution_panels(
    frame_loader: FrameLoader,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    eligibility_column: str,
) -> dict[str, pd.DataFrame]:
    iterator = getattr(frame_loader, "iter_frames", None)
    if iterator is None:
        return _execution_panels(
            frame_loader(start, end), eligibility_column=eligibility_column
        )
    collected: dict[str, list[pd.DataFrame]] = {
        "raw_open": [],
        "buyable": [],
        "sellable": [],
        "eligible": [],
    }
    for frame in iterator(start, end):
        panels = _execution_panels(frame, eligibility_column=eligibility_column)
        for name, panel in panels.items():
            collected[name].append(panel)
        del frame, panels
        gc.collect()
    combined = {
        name: pd.concat(parts, axis=1, join="outer").sort_index().sort_index(axis=1)
        for name, parts in collected.items()
    }
    for name, panel in combined.items():
        if panel.columns.duplicated().any():
            raise ValueError(f"production execution batches duplicate columns in {name}")
    return combined


def _execution_panels(
    frame: pd.DataFrame, *, eligibility_column: str = "signal_eligible"
) -> dict[str, pd.DataFrame]:
    required = {
        "ts_code",
        "trade_date",
        "hfq_open",
        eligibility_column,
        "is_suspended",
        "hfq_up_limit",
        "hfq_down_limit",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"production execution frame is missing: {', '.join(missing)}")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("production execution frame contains duplicate keys")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    if work["trade_date"].isna().any():
        raise ValueError("production execution frame contains invalid dates")
    work["ts_code"] = work["ts_code"].astype(str)
    raw_open = _pivot(work, "hfq_open", numeric=True)
    up_limit = _pivot(work, "hfq_up_limit", numeric=True)
    down_limit = _pivot(work, "hfq_down_limit", numeric=True)
    suspended = _pivot(work, "is_suspended", numeric=False)
    eligible = _pivot(work, eligibility_column, numeric=False).eq(True)
    known_open = raw_open.notna() & raw_open.gt(0)
    not_suspended = suspended.eq(False)
    return {
        "raw_open": raw_open,
        "buyable": known_open & not_suspended & up_limit.notna() & raw_open.lt(up_limit),
        "sellable": known_open
        & not_suspended
        & down_limit.notna()
        & raw_open.gt(down_limit),
        "eligible": eligible,
    }


def _from_panels(
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    panels: dict[str, pd.DataFrame],
    *,
    price_storage_dtype: np.dtype | type = np.float64,
) -> ProductionExecutionContext:
    dtype = np.dtype(price_storage_dtype)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("production execution price storage must be float32 or float64")
    raw_open = panels["raw_open"].astype(float)
    valuation = raw_open.ffill().to_numpy(dtype=dtype)
    return ProductionExecutionContext(
        dates=pd.DatetimeIndex(dates),
        codes=pd.Index(codes),
        valuation_open=valuation,
        buyable=panels["buyable"].astype("boolean").to_numpy(dtype=bool, na_value=False),
        sellable=panels["sellable"].astype("boolean").to_numpy(dtype=bool, na_value=False),
        signal_eligible=panels["eligible"]
        .astype("boolean")
        .to_numpy(dtype=bool, na_value=False),
    )


def _pivot(frame: pd.DataFrame, column: str, *, numeric: bool) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce") if numeric else frame[column]
    panel = frame.assign(_value=values).pivot(
        index="trade_date", columns="ts_code", values="_value"
    )
    return panel.sort_index().sort_index(axis=1)
