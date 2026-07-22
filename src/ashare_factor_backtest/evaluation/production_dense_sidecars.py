"""Normalize dense post-close A-share fields for production factor evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


DAILY_BASIC_SOURCE_FIELDS = (
    "turnover_rate_f",
    "volume_ratio",
    "pb",
    "circ_mv",
)
DAILY_BASIC_FIELDS = ("turnover_free", "volume_ratio", "pb", "circ_mv")
MONEYFLOW_AMOUNT_FIELDS = tuple(
    f"{side}_{size}_amount"
    for size in ("sm", "md", "lg", "elg")
    for side in ("buy", "sell")
)
MONEYFLOW_FIELDS = (
    "small_flow_imbalance",
    "medium_flow_imbalance",
    "large_flow_imbalance",
    "extra_large_flow_imbalance",
    "total_flow_imbalance",
    "large_flow_share",
)
PRODUCTION_EXTERNAL_FIELD_SOURCES = {
    **{name: "daily_basic" for name in DAILY_BASIC_FIELDS},
    **{name: "moneyflow" for name in MONEYFLOW_FIELDS},
}
PRODUCTION_EXTERNAL_FIELD_UNITS = {
    "turnover_free": "ratio",
    "volume_ratio": "ratio",
    "pb": "ratio",
    "circ_mv": "currency_10k",
    **{name: "ratio" for name in MONEYFLOW_FIELDS},
}


def normalize_daily_basic_sidecar(frame: pd.DataFrame) -> pd.DataFrame:
    work = _source_frame(
        frame,
        required={"ts_code", "trade_date", *DAILY_BASIC_SOURCE_FIELDS},
        label="daily_basic",
    )
    for column in DAILY_BASIC_SOURCE_FIELDS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["turnover_free"] = work["turnover_rate_f"] / 100.0
    return work.loc[
        :, ["ts_code", "trade_date", *DAILY_BASIC_FIELDS]
    ].reset_index(drop=True)


def normalize_moneyflow_sidecar(frame: pd.DataFrame) -> pd.DataFrame:
    work = _source_frame(
        frame,
        required={"ts_code", "trade_date", *MONEYFLOW_AMOUNT_FIELDS},
        label="moneyflow",
    )
    work.loc[:, MONEYFLOW_AMOUNT_FIELDS] = work.loc[
        :, MONEYFLOW_AMOUNT_FIELDS
    ].apply(pd.to_numeric, errors="coerce")
    amounts = work.loc[:, MONEYFLOW_AMOUNT_FIELDS].to_numpy(
        dtype=float, na_value=np.nan
    )
    finite = amounts[np.isfinite(amounts)]
    if len(finite) and np.any(finite < 0):
        raise ValueError("moneyflow amounts must be nonnegative")

    buy = {
        size: work[f"buy_{size}_amount"]
        for size in ("sm", "md", "lg", "elg")
    }
    sell = {
        size: work[f"sell_{size}_amount"]
        for size in ("sm", "md", "lg", "elg")
    }
    labels = {
        "sm": "small",
        "md": "medium",
        "lg": "large",
        "elg": "extra_large",
    }
    for size, label in labels.items():
        work[f"{label}_flow_imbalance"] = _protected_ratio(
            buy[size] - sell[size], buy[size] + sell[size]
        )
    total_buy = sum(buy.values())
    total_sell = sum(sell.values())
    total_gross = total_buy + total_sell
    work["total_flow_imbalance"] = _protected_ratio(
        total_buy - total_sell, total_gross
    )
    large_gross = buy["lg"] + sell["lg"] + buy["elg"] + sell["elg"]
    work["large_flow_share"] = _protected_ratio(large_gross, total_gross)
    return work.loc[:, ["ts_code", "trade_date", *MONEYFLOW_FIELDS]].reset_index(
        drop=True
    )


def _source_frame(
    frame: pd.DataFrame, *, required: set[str], label: str
) -> pd.DataFrame:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    work = frame.loc[:, ["ts_code", "trade_date", *sorted(required - {"ts_code", "trade_date"})]].copy()
    work["ts_code"] = work["ts_code"].astype("string")
    work["trade_date"] = pd.to_datetime(
        work["trade_date"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    )
    if work["ts_code"].isna().any() or work["trade_date"].isna().any():
        raise ValueError(f"{label} contains invalid business keys")
    if work.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError(f"{label} contains duplicate business keys")
    return work.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(
        drop=True
    )


def _protected_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)
