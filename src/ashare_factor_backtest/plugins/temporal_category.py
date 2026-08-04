"""Point-in-time materialization for interval-valued categorical data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalCategoryMaterialization:
    rows: pd.DataFrame
    codebook: dict[str, int]
    audit: dict[str, Any]


def materialize_temporal_category(
    intervals: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    category_column: str,
    output_field: str,
    entity_column: str = "ts_code",
    observation_date_column: str = "trade_date",
    start_column: str = "in_date",
    end_column: str = "out_date",
) -> TemporalCategoryMaterialization:
    interval_columns = {
        entity_column,
        category_column,
        start_column,
        end_column,
    }
    observation_columns = {entity_column, observation_date_column}
    missing_intervals = sorted(interval_columns.difference(intervals.columns))
    missing_observations = sorted(observation_columns.difference(observations.columns))
    if missing_intervals:
        raise ValueError(
            f"temporal category intervals are missing: {', '.join(missing_intervals)}"
        )
    if missing_observations:
        raise ValueError(
            "temporal category observations are missing: "
            + ", ".join(missing_observations)
        )
    if observations.empty:
        raise ValueError("temporal category observations must not be empty")

    observed = observations.loc[:, [observation_date_column, entity_column]].copy()
    observed[observation_date_column] = pd.to_datetime(
        observed[observation_date_column], errors="coerce"
    )
    observed[entity_column] = observed[entity_column].astype("string").str.strip()
    if (
        observed[observation_date_column].isna().any()
        or observed[entity_column].isna().any()
        or observed[entity_column].eq("").any()
    ):
        raise ValueError("temporal category observations contain invalid keys")
    if observed.duplicated([observation_date_column, entity_column]).any():
        raise ValueError("temporal category observations contain duplicate keys")
    observed = observed.sort_values(
        [observation_date_column, entity_column], ignore_index=True
    )

    work = intervals.loc[:, sorted(interval_columns)].copy()
    work[entity_column] = work[entity_column].astype("string").str.strip()
    work[category_column] = work[category_column].astype("string").str.strip()
    work[start_column] = _parse_dates(work[start_column])
    work[end_column] = _parse_dates(work[end_column])
    invalid_keys = (
        work[entity_column].isna()
        | work[entity_column].eq("")
        | work[category_column].isna()
        | work[category_column].eq("")
        | work[start_column].isna()
    )
    if invalid_keys.any():
        raise ValueError("temporal category intervals contain invalid keys")
    exact_before = len(work)
    work = work.drop_duplicates(
        [entity_column, category_column, start_column, end_column]
    ).reset_index(drop=True)
    exact_removed = exact_before - len(work)
    effective_end = work[end_column].fillna(pd.Timestamp.max.normalize())
    if work[start_column].gt(effective_end).any():
        raise ValueError("temporal category interval starts after it ends")
    work["_effective_end"] = effective_end

    labels = sorted(work[category_column].astype(str).unique())
    codebook = {label: _stable_category_code(label) for label in labels}
    if len(set(codebook.values())) != len(codebook):
        raise RuntimeError("temporal category code collision")

    output = np.full(len(observed), np.nan, dtype=np.float64)
    hit_count = np.zeros(len(observed), dtype=np.int16)
    positions_by_entity = observed.groupby(entity_column, sort=False).indices
    for entity, group in work.groupby(entity_column, sort=False):
        positions = positions_by_entity.get(entity)
        if positions is None:
            continue
        positions = np.asarray(positions, dtype=np.int64)
        dates = observed.iloc[positions][observation_date_column].to_numpy(
            dtype="datetime64[ns]"
        )
        selected_columns = [
            category_column,
            start_column,
            "_effective_end",
        ]
        for category, start_value, end_value in group[
            selected_columns
        ].itertuples(index=False, name=None):
            start = np.datetime64(start_value, "ns")
            end = np.datetime64(end_value, "ns")
            active = (dates >= start) & (dates <= end)
            if not active.any():
                continue
            selected = positions[active]
            hit_count[selected] += 1
            output[selected] = float(codebook[str(category)])
    output[hit_count != 1] = np.nan

    rows = observed.copy()
    rows[output_field] = output
    yearly: dict[str, dict[str, float | int]] = {}
    for year, positions in rows.groupby(
        rows[observation_date_column].dt.year, sort=True
    ).indices.items():
        selected_hits = hit_count[np.asarray(positions, dtype=np.int64)]
        count = len(selected_hits)
        yearly[str(int(year))] = {
            "row_count": count,
            "unique_rows": int((selected_hits == 1).sum()),
            "missing_rows": int((selected_hits == 0).sum()),
            "ambiguous_rows": int((selected_hits > 1).sum()),
            "unique_coverage": float((selected_hits == 1).mean()) if count else 0.0,
        }
    audit = {
        "row_count": len(rows),
        "interval_count": len(work),
        "exact_intervals_removed": exact_removed,
        "unique_rows": int((hit_count == 1).sum()),
        "missing_rows": int((hit_count == 0).sum()),
        "ambiguous_rows": int((hit_count > 1).sum()),
        "unique_coverage": float((hit_count == 1).mean()),
        "category_count": len(codebook),
        "yearly": yearly,
    }
    return TemporalCategoryMaterialization(rows, codebook, audit)


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    compact = text.str.replace("-", "", regex=False)
    return pd.to_datetime(compact, format="%Y%m%d", errors="coerce")


def _stable_category_code(label: str) -> int:
    code = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:12], 16)
    return code or 1
