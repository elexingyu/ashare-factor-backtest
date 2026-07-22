"""No-return readiness checks for the production A-share research universe."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq


def stable_readiness_hash(report: dict[str, Any]) -> str:
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class YearChunk:
    year: int
    calculation_start: pd.Timestamp
    calculation_end: pd.Timestamp
    load_start: pd.Timestamp
    load_end: pd.Timestamp
    max_lookback: int
    forward_tail: int
    partition_axis: str = "time"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("calculation_start", "calculation_end", "load_start", "load_end"):
            payload[key] = payload[key].strftime("%Y-%m-%d")
        return payload


def build_year_chunks(
    trading_dates: Sequence[pd.Timestamp], *, max_lookback: int, forward_tail: int
) -> tuple[YearChunk, ...]:
    if max_lookback < 0 or forward_tail < 0:
        raise ValueError("chunk overlap sizes must be nonnegative")
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("trading dates must not be empty")
    chunks: list[YearChunk] = []
    for year in sorted(set(dates.year)):
        positions = [index for index, date in enumerate(dates) if date.year == year]
        first, last = positions[0], positions[-1]
        load_first = max(0, first - max_lookback)
        load_last = min(len(dates) - 1, last + forward_tail)
        chunks.append(
            YearChunk(
                year=year,
                calculation_start=pd.Timestamp(dates[first]),
                calculation_end=pd.Timestamp(dates[last]),
                load_start=pd.Timestamp(dates[load_first]),
                load_end=pd.Timestamp(dates[load_last]),
                max_lookback=max_lookback,
                forward_tail=forward_tail,
            )
        )
    return tuple(chunks)


def build_bounded_time_chunks(
    trading_dates: Sequence[pd.Timestamp],
    *,
    max_lookback: int,
    forward_tail: int,
    max_calculation_dates: int,
) -> tuple[YearChunk, ...]:
    if max_calculation_dates <= 0:
        raise ValueError("max calculation dates must be positive")
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("trading dates must not be empty")
    chunks: list[YearChunk] = []
    for year in sorted(set(dates.year)):
        positions = [index for index, date in enumerate(dates) if date.year == year]
        for offset in range(0, len(positions), max_calculation_dates):
            segment = positions[offset : offset + max_calculation_dates]
            first, last = segment[0], segment[-1]
            chunks.append(
                YearChunk(
                    year=year,
                    calculation_start=pd.Timestamp(dates[first]),
                    calculation_end=pd.Timestamp(dates[last]),
                    load_start=pd.Timestamp(dates[max(0, first - max_lookback)]),
                    load_end=pd.Timestamp(
                        dates[min(len(dates) - 1, last + forward_tail)]
                    ),
                    max_lookback=max_lookback,
                    forward_tail=forward_tail,
                )
            )
    return tuple(chunks)


def audit_production_universe_assets(
    *,
    daily_dir: Path,
    adj_dir: Path,
    stock_master: Path,
    st_intervals: Path,
    trade_calendar: Path,
    market_states: Path,
    csi1000_weights: Path,
    analysis_start: str,
    analysis_end: str,
) -> dict[str, Any]:
    start = pd.Timestamp(analysis_start)
    end = pd.Timestamp(analysis_end)
    if start > end:
        raise ValueError("analysis start must not exceed analysis end")

    assets: dict[str, Any] = {}
    primary_blockers: list[str] = []
    daily_files = _symbol_files(daily_dir, ".csv")
    adj_files = _symbol_files(adj_dir, ".csv")
    common_files = sorted(daily_files & adj_files)
    assets["daily"] = {"file_count": len(daily_files), "path": str(daily_dir)}
    assets["adj_factor"] = {"file_count": len(adj_files), "path": str(adj_dir)}
    assets["daily_adj_common"] = {"file_count": len(common_files)}
    if not common_files:
        primary_blockers.append("daily_adj_common_universe_empty")

    master = _read_optional_table(stock_master)
    assets["stock_master"] = _table_asset(master, stock_master)
    required_master = {"ts_code", "market", "list_date", "delist_date"}
    if master is None:
        primary_blockers.append("stock_master_missing")
        shsz_count = 0
        mainboard_count = 0
    else:
        missing = required_master.difference(master.columns)
        primary_blockers.extend(f"stock_master_missing_{name}" for name in sorted(missing))
        codes = master["ts_code"].astype(str) if "ts_code" in master else pd.Series(dtype=str)
        shsz = codes.str.endswith((".SH", ".SZ"))
        shsz_count = int(shsz.sum())
        mainboard_count = int(
            (shsz & master.get("market", pd.Series(index=master.index, dtype=str)).eq("主板")).sum()
        )
        if shsz_count and len(common_files) / shsz_count < 0.95:
            primary_blockers.append("daily_adj_coverage_below_95pct_of_shsz_master")
    assets["stock_master"].update(
        {"shsz_count": shsz_count, "mainboard_count": mainboard_count}
    )

    st = _read_optional_table(st_intervals)
    assets["st_intervals"] = _table_asset(st, st_intervals)
    primary_blockers.extend(
        _schema_blockers(st, {"ts_code", "start_date", "end_date"}, "st_intervals")
    )

    required_first_date = start
    required_last_date = end
    calendar = _read_optional_table(trade_calendar)
    assets["trade_calendar"] = _table_asset(calendar, trade_calendar)
    primary_blockers.extend(
        _schema_blockers(calendar, {"exchange", "cal_date", "is_open"}, "trade_calendar")
    )
    if calendar is not None and "exchange" in calendar:
        exchanges = set(calendar["exchange"].dropna().astype(str))
        assets["trade_calendar"]["exchanges"] = sorted(exchanges)
        for exchange in ("SSE", "SZSE"):
            if exchange not in exchanges:
                primary_blockers.append(f"trade_calendar_missing_{exchange}")
    if calendar is not None and "cal_date" in calendar:
        calendar_dates = _dates(calendar["cal_date"])
        assets["trade_calendar"].update(_coverage(calendar_dates))
        open_mask = pd.to_numeric(calendar.get("is_open"), errors="coerce").eq(1)
        open_dates = _dates(calendar.loc[open_mask, "cal_date"])
        in_analysis = open_dates[(open_dates >= start) & (open_dates <= end)]
        if in_analysis.empty:
            primary_blockers.append("trade_calendar_does_not_cover_analysis")
        else:
            required_first_date = pd.Timestamp(in_analysis.min())
            required_last_date = pd.Timestamp(in_analysis.max())
            if required_first_date > start + pd.Timedelta(days=7):
                primary_blockers.append("trade_calendar_starts_late")
            if required_last_date < end - pd.Timedelta(days=7):
                primary_blockers.append("trade_calendar_ends_early")

    states = None
    state_dates: pd.DatetimeIndex | None = None
    if market_states.is_dir():
        state_asset, state_dates, state_blockers = _partitioned_parquet_asset(
            market_states,
            required={"ts_code", "trade_date", "is_suspended", "up_limit", "down_limit"},
            date_column="trade_date",
            label="market_states",
        )
        assets["market_states"] = state_asset
        primary_blockers.extend(state_blockers)
    else:
        states = _read_optional_table(market_states)
        assets["market_states"] = _table_asset(states, market_states)
    if not assets["market_states"]["exists"]:
        primary_blockers.append("market_states_missing")
    elif states is not None:
        primary_blockers.extend(
            _schema_blockers(
                states,
                {"ts_code", "trade_date", "is_suspended", "up_limit", "down_limit"},
                "market_states",
            )
        )
        if "trade_date" in states:
            state_dates = _dates(states["trade_date"])
            assets["market_states"].update(_coverage(state_dates))
    if state_dates is not None and (
        state_dates.min() > required_first_date
        or state_dates.max() < required_last_date
    ):
        primary_blockers.append("market_states_do_not_cover_analysis")

    csi = _read_optional_table(csi1000_weights)
    assets["csi1000_weights"] = _table_asset(csi, csi1000_weights)
    csi_blockers = _schema_blockers(
        csi, {"index_code", "con_code", "weight"}, "csi1000"
    )
    csi_date_column: str | None = None
    if csi is not None:
        if {"snapshot_date", "effective_date"}.issubset(csi.columns):
            csi_date_column = "snapshot_date"
            effective_dates = _dates(csi["effective_date"])
            assets["csi1000_weights"]["date_semantics"] = "effective_date"
            assets["csi1000_weights"]["effective_coverage"] = _coverage(
                effective_dates
            )
            if effective_dates.empty or effective_dates.min() > required_first_date:
                csi_blockers.append(
                    "csi1000_effective_history_starts_after_analysis"
                )
        elif "trade_date" in csi:
            csi_date_column = "trade_date"
            assets["csi1000_weights"]["date_semantics"] = "legacy_trade_date"
        else:
            csi_blockers.append("csi1000_missing_membership_dates")
    if csi is not None and csi_date_column is not None:
        csi_dates = _dates(csi[csi_date_column])
        assets["csi1000_weights"].update(
            {**_coverage(csi_dates), "snapshot_count": int(csi_dates.nunique())}
        )
        first_month_end = start + pd.offsets.MonthEnd(0)
        last_month_start = end - pd.offsets.MonthBegin(1)
        if csi_dates.min() > first_month_end:
            csi_blockers.append("csi1000_history_starts_after_analysis")
        if csi_dates.max() < last_month_start:
            csi_blockers.append("csi1000_history_ends_before_analysis")

    primary_blockers = sorted(set(primary_blockers))
    csi_blockers = sorted(set([*primary_blockers, *csi_blockers]))
    views = {
        "ALL_SHSZ_PIT": _view(primary_blockers, shsz_count),
        "MAINBOARD_PIT": _view(primary_blockers, mainboard_count),
        "CSI1000_PIT": _view(csi_blockers, 1000),
    }
    return {
        "schema": "astock_production_universe_readiness_v1",
        "analysis": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        "return_data_read": False,
        "primary_view": "ALL_SHSZ_PIT",
        "robustness_views": ["MAINBOARD_PIT", "CSI1000_PIT"],
        "production_ready": views["ALL_SHSZ_PIT"]["status"] == "ready",
        "assets": assets,
        "views": views,
    }


def _symbol_files(path: Path, suffix: str) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.stem for item in path.glob(f"*{suffix}") if item.is_file()}


def _read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported readiness table: {path}")


def _table_asset(frame: pd.DataFrame | None, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": frame is not None,
        "row_count": len(frame) if frame is not None else 0,
        "columns": sorted(frame.columns.astype(str).tolist()) if frame is not None else [],
    }


def _partitioned_parquet_asset(
    path: Path,
    *,
    required: set[str],
    date_column: str,
    label: str,
) -> tuple[dict[str, Any], pd.DatetimeIndex | None, list[str]]:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return (
            {
                "path": str(path),
                "exists": False,
                "row_count": 0,
                "columns": [],
                "partition_count": 0,
            },
            None,
            [f"{label}_missing"],
        )
    row_count = 0
    common_columns: set[str] | None = None
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    blockers: list[str] = []
    for file in files:
        parquet = pq.ParquetFile(file)
        columns = set(parquet.schema_arrow.names)
        common_columns = columns if common_columns is None else common_columns & columns
        row_count += parquet.metadata.num_rows
        if date_column not in columns:
            continue
        bounds = _parquet_date_bounds(parquet, file, date_column)
        if bounds is None:
            continue
        file_minimum, file_maximum = bounds
        minimum = file_minimum if minimum is None else min(minimum, file_minimum)
        maximum = file_maximum if maximum is None else max(maximum, file_maximum)
    common_columns = common_columns or set()
    blockers.extend(
        f"{label}_missing_{name}" for name in sorted(required.difference(common_columns))
    )
    dates = (
        pd.DatetimeIndex([minimum, maximum])
        if minimum is not None and maximum is not None
        else None
    )
    asset = {
        "path": str(path),
        "exists": True,
        "row_count": row_count,
        "columns": sorted(common_columns),
        "partition_count": len(files),
    }
    if dates is not None:
        asset.update(_coverage(dates))
    else:
        blockers.append(f"{label}_date_coverage_unknown")
    return asset, dates, blockers


def _parquet_date_bounds(
    parquet: pq.ParquetFile, path: Path, column: str
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    column_index = parquet.schema_arrow.names.index(column)
    raw_bounds: list[Any] = []
    for row_group in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(row_group).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            values = pd.read_parquet(path, columns=[column])[column]
            dates = _dates(values)
            return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
        raw_bounds.extend([statistics.min, statistics.max])
    normalized = [
        value.decode("utf-8") if isinstance(value, bytes) else value for value in raw_bounds
    ]
    dates = _dates(pd.Series(normalized))
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _schema_blockers(
    frame: pd.DataFrame | None, required: set[str], label: str
) -> list[str]:
    if frame is None:
        return [f"{label}_missing"]
    return [f"{label}_missing_{name}" for name in sorted(required.difference(frame.columns))]


def _dates(values: pd.Series) -> pd.DatetimeIndex:
    normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            normalized.loc[missing], format="%Y-%m-%d", errors="coerce"
        )
    parsed = parsed.dropna()
    if parsed.empty:
        raise ValueError("readiness date column contains no valid dates")
    return pd.DatetimeIndex(parsed)


def _coverage(dates: pd.DatetimeIndex) -> dict[str, str]:
    return {
        "coverage_start": dates.min().strftime("%Y-%m-%d"),
        "coverage_end": dates.max().strftime("%Y-%m-%d"),
    }


def _view(blockers: list[str], nominal_count: int) -> dict[str, Any]:
    return {
        "status": "ready" if not blockers else "data_blocked",
        "blockers": blockers,
        "nominal_symbol_count": nominal_count,
    }
