"""Point-in-time index membership normalization and lookup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def normalize_index_weight(
    frame: pd.DataFrame,
    *,
    index_code: str,
    trading_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"index weight response is missing: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"index weight response is empty: {index_code}")
    work = frame.loc[:, sorted(required)].copy()
    work["index_code"] = work["index_code"].astype(str)
    if set(work["index_code"]) != {index_code}:
        raise ValueError("index weight response contains a foreign index")
    work["con_code"] = work["con_code"].astype(str)
    work["snapshot_date"] = pd.to_datetime(
        work.pop("trade_date"), format="%Y%m%d", errors="coerce"
    )
    if work["snapshot_date"].isna().any():
        raise ValueError("index weight response contains an invalid snapshot date")
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).unique().sort_values()
    if calendar.empty:
        raise ValueError("index membership requires a nonempty trading calendar")
    snapshots = pd.DatetimeIndex(work["snapshot_date"].unique()).sort_values()
    effective: dict[pd.Timestamp, pd.Timestamp] = {}
    for snapshot in snapshots:
        position = int(calendar.searchsorted(snapshot, side="right"))
        if position >= len(calendar):
            raise ValueError(f"index snapshot has no following trading day: {snapshot.date()}")
        effective[pd.Timestamp(snapshot)] = pd.Timestamp(calendar[position])
    work["effective_date"] = work["snapshot_date"].map(effective)
    work["source"] = "tushare_index_weight"
    return work[
        [
            "index_code",
            "snapshot_date",
            "effective_date",
            "con_code",
            "weight",
            "source",
        ]
    ].sort_values(["snapshot_date", "con_code"], ignore_index=True)


def validate_index_snapshots(
    rows: pd.DataFrame,
    *,
    expected_sizes: Mapping[str, int],
    size_tolerance_fraction: float = 0.05,
) -> dict[str, Any]:
    required = {
        "index_code",
        "snapshot_date",
        "effective_date",
        "con_code",
        "weight",
        "source",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"index membership rows are missing: {', '.join(missing)}")
    if rows.empty or not 0.0 <= size_tolerance_fraction < 1.0:
        raise ValueError("index membership audit input is invalid")
    duplicate = rows.duplicated(["index_code", "snapshot_date", "con_code"])
    if duplicate.any():
        raise ValueError("index membership contains duplicate snapshot members")
    unknown = sorted(set(rows["index_code"]).difference(expected_sizes))
    if unknown:
        raise ValueError(f"index membership contains unknown indices: {unknown}")
    counts = rows.groupby(["index_code", "snapshot_date"])["con_code"].nunique()
    for (index_code, snapshot), count in counts.items():
        expected = int(expected_sizes[str(index_code)])
        lower = expected * (1.0 - size_tolerance_fraction)
        upper = expected * (1.0 + size_tolerance_fraction)
        if not lower <= int(count) <= upper:
            raise ValueError(
                "index snapshot member count outside tolerance: "
                f"{index_code} {pd.Timestamp(snapshot).date()} count={count} expected={expected}"
            )
    if (pd.to_datetime(rows["effective_date"]) <= pd.to_datetime(rows["snapshot_date"])).any():
        raise ValueError("index membership effective date must follow snapshot date")
    by_index = {}
    for index_code, group in rows.groupby("index_code"):
        member_counts = group.groupby("snapshot_date")["con_code"].nunique()
        by_index[str(index_code)] = {
            "snapshot_count": int(member_counts.size),
            "coverage_start": pd.Timestamp(group["snapshot_date"].min()).date().isoformat(),
            "coverage_end": pd.Timestamp(group["snapshot_date"].max()).date().isoformat(),
            "minimum_members": int(member_counts.min()),
            "maximum_members": int(member_counts.max()),
            "unique_members": int(group["con_code"].nunique()),
        }
    return {
        "row_count": len(rows),
        "index_count": len(by_index),
        "indices": by_index,
    }


def build_membership_mask(
    rows: pd.DataFrame,
    *,
    index_code: str,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    codes: Sequence[str] | pd.Index,
) -> pd.DataFrame:
    target_dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    target_codes = pd.Index(map(str, codes))
    if target_dates.has_duplicates or not target_dates.is_monotonic_increasing:
        raise ValueError("membership mask dates must be unique and sorted")
    if target_codes.has_duplicates:
        raise ValueError("membership mask codes must be unique")
    selected = rows.loc[rows["index_code"].astype(str).eq(index_code)].copy()
    if selected.empty:
        raise ValueError(f"index membership has no rows for {index_code}")
    selected["effective_date"] = pd.to_datetime(selected["effective_date"])
    effective_dates = pd.DatetimeIndex(selected["effective_date"].unique()).sort_values()
    code_positions = {code: position for position, code in enumerate(target_codes)}
    mask = np.zeros((len(target_dates), len(target_codes)), dtype=bool)
    for position, effective_date in enumerate(effective_dates):
        end = effective_dates[position + 1] if position + 1 < len(effective_dates) else None
        date_mask = target_dates >= effective_date
        if end is not None:
            date_mask &= target_dates < end
        members = selected.loc[
            selected["effective_date"].eq(effective_date), "con_code"
        ].astype(str)
        member_positions = [code_positions[code] for code in members if code in code_positions]
        if member_positions:
            mask[np.ix_(np.flatnonzero(date_mask), member_positions)] = True
    return pd.DataFrame(mask, index=target_dates, columns=target_codes)
