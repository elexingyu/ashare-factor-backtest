"""Build causal yearly factor inputs for the dynamic production universe."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from math import ceil

import numpy as np
import pandas as pd

from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.catalog import FieldCatalog
from ashare_factor_backtest.expression.evaluator import EvaluationContext
from ashare_factor_backtest.expression.model import FieldSpec, ValueType
from ashare_factor_backtest.plugins.index_membership import build_membership_mask
from ashare_factor_backtest.evaluation.production_dense_sidecars import (
    PRODUCTION_EXTERNAL_FIELD_SOURCES,
    PRODUCTION_EXTERNAL_FIELD_UNITS,
)
from ashare_factor_backtest.evaluation.suspension_observation_clock import (
    apply_confirmed_suspension_fill,
)


INDEX_MEMBERSHIP_VIEWS = {
    "sse50_pit": "000016.SH",
    "csi300_pit": "000300.SH",
    "csi500_pit": "000905.SH",
    "csi1000_pit": "000852.SH",
}
DYNAMIC_UNIVERSE_VIEWS = {"dynamic_small_liquid"}

PRODUCTION_FACTOR_EVALUATION_SEMANTICS = (
    "confirmed_suspension_carry_mark_flow_pit_cross_section_mask_"
    "dual_price_partial_net_r4_v8"
)


def attach_index_membership_views(
    frame: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    views: Mapping[str, str] = INDEX_MEMBERSHIP_VIEWS,
) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "signal_eligible"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"production frame is missing: {', '.join(missing)}")
    if not views:
        raise ValueError("index membership views must not be empty")
    result = frame.copy()
    dates = (
        pd.DatetimeIndex(pd.to_datetime(result["trade_date"])).unique().sort_values()
    )
    codes = pd.Index(result["ts_code"].astype(str).unique()).sort_values()
    row_dates = dates.get_indexer(pd.to_datetime(result["trade_date"]))
    row_codes = codes.get_indexer(result["ts_code"].astype(str))
    if (row_dates < 0).any() or (row_codes < 0).any():
        raise ValueError("production frame contains invalid index membership keys")
    eligible = result["signal_eligible"].eq(True).to_numpy()
    for view, index_code in views.items():
        if not view or view in required:
            raise ValueError(f"invalid index membership view: {view}")
        mask = build_membership_mask(
            memberships,
            index_code=str(index_code),
            dates=dates,
            codes=codes,
        ).to_numpy(copy=False)
        result[str(view)] = eligible & mask[row_dates, row_codes]
    return result


def attach_dynamic_small_cap_view(
    frame: pd.DataFrame,
    *,
    view: str = "dynamic_small_liquid",
    target_count: int = 1_000,
    liquidity_tail_fraction: float = 0.10,
) -> pd.DataFrame:
    required = {
        "ts_code",
        "trade_date",
        "signal_eligible",
        "trailing_20d_median_amount_cny",
        "log_total_mv",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"production frame is missing: {', '.join(missing)}")
    if view not in DYNAMIC_UNIVERSE_VIEWS:
        raise ValueError(f"unsupported dynamic universe view: {view}")
    if target_count <= 0 or not 0.0 <= liquidity_tail_fraction < 1.0:
        raise ValueError("dynamic universe selection contract is invalid")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("production frame contains duplicate date/ts_code keys")

    result = frame.copy()
    result[view] = False
    market_cap = pd.to_numeric(result["log_total_mv"], errors="coerce")
    liquidity = pd.to_numeric(
        result["trailing_20d_median_amount_cny"], errors="coerce"
    )
    eligible = (
        result["signal_eligible"].eq(True)
        & np.isfinite(market_cap)
        & np.isfinite(liquidity)
        & liquidity.gt(0)
    )
    for _, indices in result.loc[eligible].groupby("trade_date", sort=True).groups.items():
        candidates = result.loc[indices, ["ts_code"]].assign(
            _market_cap=market_cap.loc[indices],
            _liquidity=liquidity.loc[indices],
        )
        candidates = candidates.sort_values(
            ["_liquidity", "ts_code"], kind="mergesort"
        )
        drop_count = min(
            len(candidates), ceil(len(candidates) * liquidity_tail_fraction)
        )
        selected = candidates.iloc[drop_count:].sort_values(
            ["_market_cap", "ts_code"], kind="mergesort"
        ).head(target_count)
        result.loc[selected.index, view] = True
    return result


def build_production_evaluation_context(
    frame: pd.DataFrame,
    *,
    dataset_version: str,
    view: str,
    required_fields: set[str] | None = None,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
    additional_dataset_versions: Mapping[str, str] | None = None,
    initial_price_values: Mapping[str, pd.Series] | None = None,
) -> tuple[FieldCatalog, EvaluationContext]:
    specs = dict(additional_field_specs or {})
    panels, universe_mask = _production_field_panels(
        frame,
        view=view,
        required_fields=required_fields,
        additional_field_specs=specs,
        initial_price_values=initial_price_values,
    )
    return _evaluation_context_from_panels(
        panels,
        universe_mask=universe_mask,
        dataset_version=dataset_version,
        view=view,
        additional_field_specs=specs,
        additional_dataset_versions=additional_dataset_versions,
    )


def build_production_evaluation_context_from_batches(
    frames: Iterable[pd.DataFrame],
    *,
    dataset_version: str,
    view: str,
    required_fields: set[str] | None = None,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
    additional_dataset_versions: Mapping[str, str] | None = None,
    initial_price_values: Mapping[str, pd.Series] | None = None,
) -> tuple[FieldCatalog, EvaluationContext]:
    specs = dict(additional_field_specs or {})
    collected: dict[str, list[pd.DataFrame]] = {}
    collected_masks: list[pd.DataFrame] = []
    batch_count = 0
    for frame in frames:
        panels, universe_mask = _production_field_panels(
            frame,
            view=view,
            required_fields=required_fields,
            additional_field_specs=specs,
            initial_price_values=initial_price_values,
        )
        for name, panel in panels.items():
            collected.setdefault(name, []).append(panel)
        collected_masks.append(universe_mask)
        batch_count += 1
    if not batch_count:
        raise ValueError("production evaluation batches are empty")
    combined = {
        name: pd.concat(parts, axis=1, join="outer").sort_index().sort_index(axis=1)
        for name, parts in collected.items()
    }
    for name, panel in combined.items():
        if panel.columns.duplicated().any():
            raise ValueError(f"production evaluation batches duplicate {name} columns")
    combined_mask = (
        pd.concat(collected_masks, axis=1, join="outer").sort_index().sort_index(axis=1)
    )
    if combined_mask.columns.duplicated().any():
        raise ValueError(
            "production evaluation batches duplicate universe-mask columns"
        )
    combined_mask = combined_mask.astype("boolean").fillna(False).astype(bool)
    return _evaluation_context_from_panels(
        combined,
        universe_mask=combined_mask,
        dataset_version=dataset_version,
        view=view,
        additional_field_specs=specs,
        additional_dataset_versions=additional_dataset_versions,
    )


def _production_field_panels(
    frame: pd.DataFrame,
    *,
    view: str,
    required_fields: set[str] | None = None,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
    initial_price_values: Mapping[str, pd.Series] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    views = {
        "signal_eligible",
        "mainboard",
        "liquid_20m",
        *INDEX_MEMBERSHIP_VIEWS,
        *DYNAMIC_UNIVERSE_VIEWS,
    }
    if view not in views:
        raise ValueError(f"unsupported production universe view: {view}")
    required = {
        "ts_code",
        "trade_date",
        "hfq_open",
        "hfq_high",
        "hfq_low",
        "hfq_close",
        "volume_shares",
        "amount_cny",
        view,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"production frame is missing: {', '.join(missing)}")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("production frame contains duplicate date/ts_code keys")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    if work["trade_date"].isna().any():
        raise ValueError("production frame contains invalid trade dates")
    membership = work[view].eq(True)
    universe_mask = work.assign(_membership=membership).pivot(
        index="trade_date", columns="ts_code", values="_membership"
    )
    universe_mask = (
        universe_mask.sort_index()
        .sort_index(axis=1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    sources = {
        "open": "hfq_open",
        "high": "hfq_high",
        "low": "hfq_low",
        "close": "hfq_close",
        "volume": "volume_shares",
        "amount": "amount_cny",
        **{name: name for name in PRODUCTION_EXTERNAL_FIELD_SOURCES},
        **{name: name for name in (additional_field_specs or {})},
    }
    selected_fields = (
        {name for name, source in sources.items() if source in work.columns}
        if required_fields is None
        else set(required_fields)
    )
    unknown = selected_fields.difference(sources)
    if unknown or not selected_fields:
        raise ValueError(f"unsupported production fields: {sorted(unknown)}")
    sources = {
        name: source for name, source in sources.items() if name in selected_fields
    }
    panels: dict[str, pd.DataFrame] = {}
    for name, source in sources.items():
        values = pd.to_numeric(work[source], errors="coerce")
        panel = work.assign(_value=values).pivot(
            index="trade_date", columns="ts_code", values="_value"
        )
        panels[name] = panel.sort_index().sort_index(axis=1)
    fillable = {
        name: panel
        for name, panel in panels.items()
        if name in {"open", "high", "low", "close", "volume", "amount"}
    }
    if fillable:
        if "is_suspended" in work:
            suspension = (
                work.pivot(index="trade_date", columns="ts_code", values="is_suspended")
                .sort_index()
                .sort_index(axis=1)
            )
        elif any(panel.isna().to_numpy().any() for panel in fillable.values()):
            raise ValueError(
                "production fields with missing values require is_suspended state"
            )
        else:
            reference = next(iter(fillable.values()))
            suspension = pd.DataFrame(
                False, index=reference.index, columns=reference.columns
            )
        seeds = {
            name: values
            for name, values in (initial_price_values or {}).items()
            if name in fillable
        }
        panels.update(
            apply_confirmed_suspension_fill(
                fillable,
                suspension,
                initial_prices=seeds,
            )
        )
    return panels, universe_mask


def price_carry_state_before(
    context: EvaluationContext, *, before: pd.Timestamp
) -> dict[str, pd.Series]:
    """Capture the compact price state needed by the next time chunk."""
    cutoff = pd.Timestamp(before)
    result: dict[str, pd.Series] = {}
    for name in ("open", "high", "low", "close"):
        panel = context.fields.get(name)
        if panel is None:
            continue
        prior = panel.loc[panel.index < cutoff]
        result[name] = (
            pd.Series(float("nan"), index=panel.columns, dtype=float)
            if prior.empty
            else prior.iloc[-1].copy()
        )
    return result


def production_field_catalog_version(
    *,
    dataset_version: str,
    view: str,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
) -> str:
    specs = dict(additional_field_specs or {})
    version = f"production_daily_fields_{dataset_version}_{view}"
    if not specs:
        return version
    payload = [
        {
            "available_at": spec.available_at,
            "dataset_version": spec.dataset_version,
            "name": name,
            "price_basis": spec.price_basis.value if spec.price_basis else None,
            "unit_lineage": spec.unit_lineage,
        }
        for name, spec in sorted(specs.items())
    ]
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{version}_plugins_{suffix}"


def build_production_field_catalog(
    *,
    field_names: set[str],
    date_range: tuple[str, str],
    dataset_version: str,
    view: str,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
) -> FieldCatalog:
    if not field_names or len(date_range) != 2:
        raise ValueError("production field catalog requires fields and a date range")
    specs = dict(additional_field_specs or {})
    catalog = FieldCatalog(
        production_field_catalog_version(
            dataset_version=dataset_version,
            view=view,
            additional_field_specs=specs,
        )
    )
    units = {
        "open": "price",
        "high": "price",
        "low": "price",
        "close": "price",
        "volume": "volume_shares",
        "amount": "currency",
        **PRODUCTION_EXTERNAL_FIELD_UNITS,
    }
    unsupported = field_names.difference(units).difference(specs)
    if unsupported:
        raise ValueError(f"unsupported production fields: {sorted(unsupported)}")
    start, end = date_range
    for name in sorted(field_names):
        if name in specs:
            catalog.register(specs[name])
            continue
        catalog.register(
            FieldSpec(
                name=name,
                value_type=ValueType.PANEL_FLOAT,
                available_at=(
                    "same_day_post_close"
                    if name in PRODUCTION_EXTERNAL_FIELD_SOURCES
                    else "same_day_15:00"
                ),
                price_basis=PriceBasis.HFQ_PIT if units[name] == "price" else None,
                unit_lineage=units[name],
                dataset_version=dataset_version,
                min_date=start,
                max_date=end,
                coverage_note=(
                    f"complete causal history with dynamic PIT cross-section mask: {view}"
                ),
            )
        )
    return catalog


def _evaluation_context_from_panels(
    panels: dict[str, pd.DataFrame],
    *,
    universe_mask: pd.DataFrame,
    dataset_version: str,
    view: str,
    additional_field_specs: Mapping[str, FieldSpec] | None = None,
    additional_dataset_versions: Mapping[str, str] | None = None,
) -> tuple[FieldCatalog, EvaluationContext]:
    reference = next(iter(panels.values()))
    start = reference.index.min().date().isoformat()
    end = reference.index.max().date().isoformat()
    catalog = build_production_field_catalog(
        field_names=set(panels),
        date_range=(start, end),
        dataset_version=dataset_version,
        view=view,
        additional_field_specs=additional_field_specs,
    )
    if not universe_mask.index.equals(
        reference.index
    ) or not universe_mask.columns.equals(reference.columns):
        raise ValueError("production universe mask must align with field panels")
    universe_size = universe_mask.sum(axis=1).astype(int)
    versions = {"production_daily": dataset_version}
    for name, identity in (additional_dataset_versions or {}).items():
        if name in versions and versions[name] != identity:
            raise ValueError(f"production dataset identity collision: {name}")
        versions[name] = identity
    context = EvaluationContext(
        fields=panels,
        dataset_versions=versions,
        universe_policy=f"dynamic_pit:{view}",
        date_range=(start, end),
        universe_size=universe_size,
        universe_mask=universe_mask,
        evaluation_price_basis=PriceBasis.HFQ_PIT,
    )
    return catalog, context


def build_production_year_frame(
    *,
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    stock_master: pd.DataFrame,
    st_intervals: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    market_states: pd.DataFrame,
    min_listing_trading_days: int = 120,
    liquidity_lookback: int = 20,
    liquidity_floor_cny: float = 20_000_000.0,
) -> pd.DataFrame:
    if min_listing_trading_days < 0 or liquidity_lookback <= 0:
        raise ValueError(
            "listing age must be nonnegative and liquidity lookback positive"
        )
    if not np.isfinite(liquidity_floor_cny) or liquidity_floor_cny <= 0:
        raise ValueError("liquidity floor must be positive and finite")
    bars = _frame(
        daily,
        {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"},
        "daily",
    )
    factors = _frame(adj, {"ts_code", "trade_date", "adj_factor"}, "adj")
    master = _frame(
        stock_master,
        {"ts_code", "exchange", "market", "list_date", "delist_date"},
        "stock_master",
    )
    states = _frame(
        market_states,
        {"ts_code", "trade_date", "is_suspended", "up_limit", "down_limit"},
        "market_states",
    )
    calendar = _frame(
        trade_calendar, {"exchange", "cal_date", "is_open"}, "trade_calendar"
    )
    intervals = _frame(
        st_intervals, {"ts_code", "start_date", "end_date"}, "st_intervals"
    )

    for frame, column in (
        (bars, "trade_date"),
        (factors, "trade_date"),
        (states, "trade_date"),
    ):
        frame[column] = _dates(frame[column], allow_missing=False)
    master["list_date"] = _dates(master["list_date"], allow_missing=False)
    master["delist_date"] = _dates(master["delist_date"], allow_missing=True)
    calendar["cal_date"] = _dates(calendar["cal_date"], allow_missing=False)
    intervals["start_date"] = _dates(intervals["start_date"], allow_missing=False)
    intervals["end_date"] = _dates(intervals["end_date"], allow_missing=True)

    _require_unique(bars, ["ts_code", "trade_date"], "daily")
    _require_unique(factors, ["ts_code", "trade_date"], "adj")
    _require_unique(states, ["ts_code", "trade_date"], "market_states")
    _require_unique(master, ["ts_code"], "stock_master")
    _require_unique(calendar, ["exchange", "cal_date"], "trade_calendar")

    numeric_bars = ["open", "high", "low", "close", "vol", "amount"]
    bars[numeric_bars] = bars[numeric_bars].apply(pd.to_numeric, errors="coerce")
    factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
    work = (
        bars.merge(
            factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one"
        )
        .merge(
            states.loc[
                :, ["ts_code", "trade_date", "is_suspended", "up_limit", "down_limit"]
            ],
            on=["ts_code", "trade_date"],
            how="outer",
            validate="one_to_one",
        )
        .merge(master, on="ts_code", how="left", validate="many_to_one")
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    work = work[work["exchange"].isin(["SSE", "SZSE"])].copy()

    shsz = work["exchange"].isin(["SSE", "SZSE"])
    listed = work["list_date"].notna() & work["list_date"].le(work["trade_date"])
    not_delisted = work["delist_date"].isna() | work["trade_date"].le(
        work["delist_date"]
    )
    listing_age = _listing_age(work, calendar)
    st_active = _active_st_mask(work, intervals)
    known_suspension = work["is_suspended"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    not_suspended = known_suspension & work["is_suspended"].eq(False)

    numeric = work.loc[:, [*numeric_bars, "adj_factor"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = pd.Series(np.isfinite(numeric).all(axis=1), index=work.index)
    positive_prices = (
        numeric[["open", "high", "low", "close", "adj_factor"]].gt(0).all(axis=1)
    )
    valid_activity = numeric["vol"].gt(0) & numeric["amount"].gt(0)
    signal_eligible = (
        shsz
        & listed
        & not_delisted
        & listing_age.ge(min_listing_trading_days)
        & ~st_active
        & not_suspended
        & finite
        & positive_prices
        & valid_activity
    )

    work["volume_shares"] = numeric["vol"] * 100.0
    work["amount_cny"] = numeric["amount"] * 1_000.0
    for name in ("open", "high", "low", "close"):
        work[f"raw_{name}"] = numeric[name]
        work[f"hfq_{name}"] = numeric[name] * numeric["adj_factor"]
    work["adj_factor"] = numeric["adj_factor"]
    work["hfq_up_limit"] = (
        pd.to_numeric(work["up_limit"], errors="coerce") * numeric["adj_factor"]
    )
    work["hfq_down_limit"] = (
        pd.to_numeric(work["down_limit"], errors="coerce") * numeric["adj_factor"]
    )
    trailing_amount = work.groupby("ts_code", sort=False)["amount_cny"].transform(
        lambda values: (
            values.shift(1)
            .rolling(liquidity_lookback, min_periods=liquidity_lookback)
            .median()
        )
    )
    work["signal_eligible"] = signal_eligible.astype(bool)
    work["mainboard"] = (signal_eligible & work["market"].eq("主板")).astype(bool)
    work["liquid_20m"] = (
        signal_eligible & trailing_amount.ge(liquidity_floor_cny)
    ).astype(bool)
    work["listing_trading_days"] = listing_age
    work["trailing_20d_median_amount_cny"] = trailing_amount

    columns = [
        "ts_code",
        "trade_date",
        "exchange",
        "market",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "adj_factor",
        "hfq_open",
        "hfq_high",
        "hfq_low",
        "hfq_close",
        "volume_shares",
        "amount_cny",
        "signal_eligible",
        "mainboard",
        "liquid_20m",
        "listing_trading_days",
        "trailing_20d_median_amount_cny",
        "is_suspended",
        "up_limit",
        "down_limit",
        "hfq_up_limit",
        "hfq_down_limit",
    ]
    return (
        work.loc[:, columns]
        .sort_values(["trade_date", "ts_code"])
        .reset_index(drop=True)
    )


def _listing_age(work: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    result = pd.Series(0, index=work.index, dtype="int64")
    open_calendar = calendar[pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)]
    for exchange, indices in work.groupby(
        "exchange", dropna=False, sort=False
    ).groups.items():
        if exchange not in {"SSE", "SZSE"}:
            continue
        dates = np.sort(
            open_calendar.loc[open_calendar["exchange"].eq(exchange), "cal_date"]
            .dropna()
            .to_numpy(dtype="datetime64[ns]")
        )
        if not len(dates):
            continue
        rows = work.loc[indices]
        listed = rows["list_date"].to_numpy(dtype="datetime64[ns]")
        traded = rows["trade_date"].to_numpy(dtype="datetime64[ns]")
        starts = np.searchsorted(dates, listed, side="left")
        ends = np.searchsorted(dates, traded, side="right")
        result.loc[indices] = np.maximum(0, ends - starts)
    return result


def _active_st_mask(work: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
    result = pd.Series(False, index=work.index)
    if intervals.empty:
        return result
    grouped = {code: rows for code, rows in intervals.groupby("ts_code", sort=False)}
    for code, indices in work.groupby("ts_code", sort=False).groups.items():
        code_intervals = grouped.get(code)
        if code_intervals is None:
            continue
        dates = work.loc[indices, "trade_date"]
        active = pd.Series(False, index=indices)
        for row in code_intervals.itertuples(index=False):
            within = dates.ge(row.start_date) & (
                pd.isna(row.end_date) | dates.le(row.end_date)
            )
            active |= within
        result.loc[indices] = active
    return result


def _frame(source: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")
    return source.copy()


def _require_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys: {', '.join(keys)}")


def _dates(values: pd.Series, *, allow_missing: bool) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
        invalid = values.notna() & parsed.isna() if allow_missing else parsed.isna()
        if invalid.any():
            raise ValueError("date column contains invalid values")
        return parsed
    normalized = values.astype("string").str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
    fallback = parsed.isna() & normalized.notna()
    if fallback.any():
        parsed.loc[fallback] = pd.to_datetime(
            normalized.loc[fallback], format="%Y-%m-%d", errors="coerce"
        )
    invalid = values.notna() & parsed.isna() if allow_missing else parsed.isna()
    if invalid.any():
        raise ValueError("date column contains invalid values")
    return parsed
