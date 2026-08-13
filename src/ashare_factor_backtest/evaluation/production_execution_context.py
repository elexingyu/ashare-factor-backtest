"""Compressed execution matrices shared by all production factor candidates."""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from ashare_factor_backtest.evaluation.production_chunked_evaluator import FrameLoader
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


_EXECUTION_PANEL_NAMES = (
    "execution_open",
    "valuation_open",
    "valuation_close",
    "buyable",
    "sellable",
    "eligible",
)


@dataclass(frozen=True)
class ProductionExecutionContext:
    dates: pd.DatetimeIndex
    codes: pd.Index
    valuation_open: np.ndarray
    buyable: np.ndarray
    sellable: np.ndarray
    signal_eligible: np.ndarray
    execution_open: np.ndarray | None = None
    valuation_close: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.execution_open is None:
            object.__setattr__(self, "execution_open", self.valuation_open.copy())
        if self.valuation_close is None:
            object.__setattr__(self, "valuation_close", self.valuation_open.copy())
        shape = (len(self.dates), len(self.codes))
        for name in (
            "valuation_open",
            "execution_open",
            "valuation_close",
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


class ChunkedProductionExecutionAccumulator:
    """Collect compact execution panels while factor frames stream past."""

    def __init__(
        self,
        chunks: Sequence[YearChunk],
        *,
        price_storage_dtype: np.dtype | type = np.float64,
        eligibility_column: str = "signal_eligible",
    ) -> None:
        self._chunks = tuple(chunks)
        self._expected = {_chunk_key(chunk): chunk for chunk in self._chunks}
        if not self._chunks:
            raise ValueError("production execution capture chunks must not be empty")
        if len(self._expected) != len(self._chunks):
            raise ValueError("production execution capture chunks must be unique")
        self._price_storage_dtype = price_storage_dtype
        self._eligibility_column = eligibility_column
        self._active: set[tuple[object, ...]] = set()
        self._completed: set[tuple[object, ...]] = set()
        self._collected: dict[str, list[pd.DataFrame]] = {
            name: [] for name in _EXECUTION_PANEL_NAMES
        }

    def capture_frames(
        self,
        chunk: YearChunk,
        frames: Iterable[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        key = _chunk_key(chunk)
        if key not in self._expected:
            raise ValueError("production execution chunk is outside capture contract")
        if key in self._active or key in self._completed:
            raise ValueError("production execution chunk already captured")
        self._active.add(key)
        parts: dict[str, list[pd.DataFrame]] = {
            name: [] for name in _EXECUTION_PANEL_NAMES
        }
        try:
            for frame in frames:
                panels = _execution_panels(
                    frame,
                    eligibility_column=self._eligibility_column,
                )
                for name, panel in panels.items():
                    parts[name].append(panel)
                yield frame
            finalized = _combine_symbol_batch_panels(parts)
            selected = _select_chunk_panels(chunk, finalized)
            for name in _EXECUTION_PANEL_NAMES:
                self._collected[name].append(selected[name])
            self._completed.add(key)
        finally:
            self._active.discard(key)

    def build(self) -> ProductionExecutionContext:
        if self._active or self._completed != set(self._expected):
            raise ValueError("production execution capture is not complete")
        combined: dict[str, pd.DataFrame] = {}
        for name, panels in self._collected.items():
            panel = (
                pd.concat(panels, axis=0, join="outer")
                .sort_index()
                .sort_index(axis=1)
            )
            if panel.index.duplicated().any():
                raise ValueError(f"production execution chunks duplicate dates in {name}")
            combined[name] = panel
        return _from_panels(
            combined["valuation_open"].index,
            combined["valuation_open"].columns,
            combined,
            price_storage_dtype=self._price_storage_dtype,
        )


class ExecutionCapturingFrameLoader:
    """Proxy a frame loader and capture execution data from the same batches."""

    def __init__(
        self,
        frame_loader: FrameLoader,
        chunks: Sequence[YearChunk],
        *,
        price_storage_dtype: np.dtype | type = np.float64,
        eligibility_column: str = "signal_eligible",
    ) -> None:
        self._frame_loader = frame_loader
        self._chunks = tuple(chunks)
        self._chunks_by_load_range = {
            _load_range_key(chunk.load_start, chunk.load_end): chunk
            for chunk in self._chunks
        }
        if len(self._chunks_by_load_range) != len(self._chunks):
            raise ValueError("production execution capture load ranges must be unique")
        self._accumulator = ChunkedProductionExecutionAccumulator(
            self._chunks,
            price_storage_dtype=price_storage_dtype,
            eligibility_column=eligibility_column,
        )
        self.additional_field_specs = dict(
            getattr(frame_loader, "additional_field_specs", {})
        )
        self.additional_dataset_versions = dict(
            getattr(frame_loader, "additional_dataset_versions", {})
        )

    def iter_frames(
        self,
        load_start: pd.Timestamp,
        load_end: pd.Timestamp,
    ) -> Iterator[pd.DataFrame]:
        key = _load_range_key(load_start, load_end)
        chunk = self._chunks_by_load_range.get(key)
        if chunk is None:
            raise ValueError("production execution capture range is outside contract")
        iterator = getattr(self._frame_loader, "iter_frames", None)
        frames = (
            iterator(load_start, load_end)
            if iterator is not None
            else (self._frame_loader(load_start, load_end),)
        )
        yield from self._accumulator.capture_frames(chunk, frames)

    def __call__(
        self,
        load_start: pd.Timestamp,
        load_end: pd.Timestamp,
    ) -> pd.DataFrame:
        frames = list(self.iter_frames(load_start, load_end))
        if not frames:
            raise ValueError("production execution capture loader returned no rows")
        result = pd.concat(frames, ignore_index=True)
        if result.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("production execution capture contains duplicate keys")
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def execution_context(self) -> ProductionExecutionContext:
        return self._accumulator.build()


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
        execution_open=context.execution_open[selected],
        valuation_close=context.valuation_close[selected],
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
        else panels["valuation_open"].index.sort_values()
    )
    target_codes = (
        pd.Index(sorted(str(code) for code in codes))
        if codes is not None
        else panels["valuation_open"].columns.sort_values()
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
        "execution_open": [],
        "valuation_open": [],
        "valuation_close": [],
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
        mask = panels["valuation_open"].index.to_series().between(
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
    dates = combined["valuation_open"].index
    codes = combined["valuation_open"].columns
    return _from_panels(
        dates,
        codes,
        combined,
        price_storage_dtype=price_storage_dtype,
    )


def _chunk_key(chunk: YearChunk) -> tuple[object, ...]:
    return (
        int(chunk.year),
        pd.Timestamp(chunk.calculation_start),
        pd.Timestamp(chunk.calculation_end),
        pd.Timestamp(chunk.load_start),
        pd.Timestamp(chunk.load_end),
        int(chunk.max_lookback),
        int(chunk.forward_tail),
        str(chunk.partition_axis),
    )


def _load_range_key(
    load_start: pd.Timestamp,
    load_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(load_start), pd.Timestamp(load_end)


def _combine_symbol_batch_panels(
    parts: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if any(not parts.get(name) for name in _EXECUTION_PANEL_NAMES):
        raise ValueError("production execution capture frames are empty")
    combined = {
        name: (
            pd.concat(parts[name], axis=1, join="outer")
            .sort_index()
            .sort_index(axis=1)
        )
        for name in _EXECUTION_PANEL_NAMES
    }
    for name, panel in combined.items():
        if panel.columns.duplicated().any():
            raise ValueError(f"production execution batches duplicate columns in {name}")
    return combined


def _select_chunk_panels(
    chunk: YearChunk,
    panels: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    mask = panels["valuation_open"].index.to_series().between(
        chunk.calculation_start,
        chunk.calculation_end,
    )
    if not mask.any():
        raise ValueError(f"production execution chunk {chunk.year} has no dates")
    selected = {
        name: panel.loc[mask].copy()
        for name, panel in panels.items()
    }
    reference = selected["valuation_open"]
    for name, panel in selected.items():
        if not panel.index.equals(reference.index) or not panel.columns.equals(
            reference.columns
        ):
            raise ValueError(f"production execution capture {name} axes mismatch")
    return selected


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
        "execution_open": [],
        "valuation_open": [],
        "valuation_close": [],
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
    valuation_open = _pivot(work, "hfq_open", numeric=True)
    valuation_close = (
        _pivot(work, "hfq_close", numeric=True)
        if "hfq_close" in work.columns
        else valuation_open.copy()
    )
    raw_contract = {"raw_open", "up_limit", "down_limit"}
    if raw_contract.issubset(work.columns):
        execution_open = _pivot(work, "raw_open", numeric=True)
        up_limit = _pivot(work, "up_limit", numeric=True)
        down_limit = _pivot(work, "down_limit", numeric=True)
    else:
        execution_open = valuation_open
        up_limit = _pivot(work, "hfq_up_limit", numeric=True)
        down_limit = _pivot(work, "hfq_down_limit", numeric=True)
    suspended = _pivot(work, "is_suspended", numeric=False)
    eligible = _pivot(work, eligibility_column, numeric=False).eq(True)
    known_open = execution_open.notna() & execution_open.gt(0)
    not_suspended = suspended.eq(False)
    return {
        "execution_open": execution_open,
        "valuation_open": valuation_open,
        "valuation_close": valuation_close,
        "buyable": known_open
        & not_suspended
        & up_limit.notna()
        & execution_open.lt(up_limit),
        "sellable": known_open
        & not_suspended
        & down_limit.notna()
        & execution_open.gt(down_limit),
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
    execution_open = panels["execution_open"].astype(float).to_numpy(dtype=dtype)
    valuation = panels["valuation_open"].astype(float).ffill().to_numpy(dtype=dtype)
    valuation_close = (
        panels["valuation_close"].astype(float).ffill().to_numpy(dtype=dtype)
    )
    return ProductionExecutionContext(
        dates=pd.DatetimeIndex(dates),
        codes=pd.Index(codes),
        valuation_open=valuation,
        buyable=panels["buyable"].astype("boolean").to_numpy(dtype=bool, na_value=False),
        sellable=panels["sellable"].astype("boolean").to_numpy(dtype=bool, na_value=False),
        signal_eligible=panels["eligible"]
        .astype("boolean")
        .to_numpy(dtype=bool, na_value=False),
        execution_open=execution_open,
        valuation_close=valuation_close,
    )


def _pivot(frame: pd.DataFrame, column: str, *, numeric: bool) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce") if numeric else frame[column]
    panel = frame.assign(_value=values).pivot(
        index="trade_date", columns="ts_code", values="_value"
    )
    return panel.sort_index().sort_index(axis=1)
