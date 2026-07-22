"""Load bounded annual production partitions into the causal dynamic-universe frame."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Sequence
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ashare_factor_backtest.expression.model import FieldSpec, ValueType
from ashare_factor_backtest.evaluation.production_context import (
    attach_index_membership_views,
    build_production_year_frame,
)
from ashare_factor_backtest.evaluation.production_dense_sidecars import (
    PRODUCTION_EXTERNAL_FIELD_SOURCES,
)
from ashare_factor_backtest.plugins.production_bridge import ProductionPluginFieldSource


class ProductionFrameLoader:
    def __init__(
        self,
        *,
        bars_dir: Path,
        market_states_dir: Path,
        stock_master_path: Path,
        st_intervals_path: Path,
        trade_calendar_path: Path,
        symbols: Sequence[str] | None = None,
        dense_sidecars_dir: Path | None = None,
        external_fields: set[str] | None = None,
        plugin_manifest_paths: Sequence[Path] | None = None,
        plugin_fields: set[str] | None = None,
        plugin_field_source: ProductionPluginFieldSource | None = None,
        index_membership_path: Path | None = None,
        index_membership_views: Mapping[str, str] | None = None,
    ) -> None:
        self.bars_dir = bars_dir
        self.market_states_dir = market_states_dir
        selected = tuple(sorted(str(code) for code in symbols)) if symbols else None
        if selected is not None and (not selected or len(set(selected)) != len(selected)):
            raise ValueError("production loader symbols must be unique")
        self.symbols = selected
        self.dense_sidecars_dir = dense_sidecars_dir
        self.external_fields = set(external_fields or ())
        unknown = self.external_fields.difference(PRODUCTION_EXTERNAL_FIELD_SOURCES)
        if unknown:
            raise ValueError(f"unsupported production external fields: {sorted(unknown)}")
        if self.external_fields and self.dense_sidecars_dir is None:
            raise ValueError("production external fields require dense sidecars")
        if plugin_manifest_paths and plugin_field_source is not None:
            raise ValueError("provide plugin manifests or a plugin field source, not both")
        self.plugin_field_source = plugin_field_source or (
            ProductionPluginFieldSource(plugin_manifest_paths)
            if plugin_manifest_paths
            else None
        )
        self.plugin_fields = set(plugin_fields or ())
        if self.plugin_fields and self.plugin_field_source is None:
            raise ValueError("production plugin fields require plugin manifests")
        if self.plugin_field_source is not None:
            unknown_plugin = self.plugin_fields.difference(
                self.plugin_field_source.available_fields
            )
            if unknown_plugin:
                raise ValueError(
                    f"unsupported production plugin fields: {sorted(unknown_plugin)}"
                )
        overlap = self.external_fields.intersection(self.plugin_fields)
        if overlap:
            raise ValueError(f"production fields have multiple sources: {sorted(overlap)}")
        self.additional_field_specs: dict[str, FieldSpec] = {}
        self.additional_dataset_versions: dict[str, str] = {}
        if self.plugin_field_source is not None:
            contracts = self.plugin_field_source.field_contracts
            for name in sorted(self.plugin_fields):
                field, manifest = contracts[name]
                self.additional_field_specs[name] = FieldSpec(
                    name=name,
                    value_type=ValueType.PANEL_FLOAT,
                    available_at=(
                        f"day+{field.available_at.day_offset}_"
                        f"{field.available_at.time.isoformat(timespec='minutes')}"
                    ),
                    price_basis=field.price_basis,
                    unit_lineage=field.unit_lineage,
                    dataset_version=manifest.identity,
                    min_date=manifest.coverage_start.isoformat(),
                    max_date=manifest.coverage_end.isoformat(),
                    coverage_note=f"validated production plugin: {manifest.dataset}",
                )
                self.additional_dataset_versions[manifest.dataset] = manifest.identity
        self.index_membership_views = dict(index_membership_views or {})
        if bool(index_membership_path) != bool(self.index_membership_views):
            raise ValueError(
                "production index membership path and views must be provided together"
            )
        self.stock_master = pd.read_parquet(stock_master_path)
        self.st_intervals = pd.read_parquet(st_intervals_path)
        self.trade_calendar = pd.read_csv(trade_calendar_path)
        if self.symbols is not None:
            wanted = set(self.symbols)
            self.stock_master = self.stock_master[
                self.stock_master["ts_code"].astype(str).isin(wanted)
            ].copy()
            self.st_intervals = self.st_intervals[
                self.st_intervals["ts_code"].astype(str).isin(wanted)
            ].copy()
        if self.stock_master.empty:
            raise ValueError("production loader stock master is empty")
        self.index_memberships: pd.DataFrame | None = None
        if index_membership_path is not None:
            available = set(
                pd.read_parquet(index_membership_path, columns=["index_code"])[
                    "index_code"
                ].astype(str)
            )
            missing_indices = sorted(
                set(self.index_membership_views.values()).difference(available)
            )
            if missing_indices:
                raise ValueError(
                    f"index membership source is missing: {', '.join(missing_indices)}"
                )
            filters: list[tuple[str, str, object]] = [
                ("index_code", "in", sorted(set(self.index_membership_views.values())))
            ]
            if self.symbols is not None:
                filters.append(("con_code", "in", list(self.symbols)))
            self.index_memberships = pd.read_parquet(
                index_membership_path,
                columns=[
                    "index_code",
                    "snapshot_date",
                    "effective_date",
                    "con_code",
                    "weight",
                    "source",
                ],
                filters=filters,
            )

    def __call__(self, load_start: pd.Timestamp, load_end: pd.Timestamp) -> pd.DataFrame:
        start = pd.Timestamp(load_start).normalize()
        end = pd.Timestamp(load_end).normalize()
        if start > end:
            raise ValueError("production load start must not exceed end")
        years = tuple(range(start.year, end.year + 1))
        bars = self._read_years(
            self.bars_dir,
            years=years,
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
                "adj_factor",
            ],
            start=start,
            end=end,
            label="production bars",
        )
        states = self._read_years(
            self.market_states_dir,
            years=years,
            columns=["ts_code", "trade_date", "is_suspended", "up_limit", "down_limit"],
            start=start,
            end=end,
            label="market states",
        )
        if bars.empty:
            raise ValueError("production bar partitions contain no requested rows")
        if bars.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("production bars contain duplicate keys")
        if states.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("market states contain duplicate keys")
        missing_states = bars.loc[:, ["ts_code", "trade_date"]].merge(
            states.loc[:, ["ts_code", "trade_date"]],
            on=["ts_code", "trade_date"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        if missing_states["_merge"].ne("both").any():
            raise ValueError("production bars have missing market-state keys")
        master_codes = set(self.stock_master["ts_code"].astype(str))
        missing_master = set(bars["ts_code"].astype(str)).difference(master_codes)
        if missing_master:
            raise ValueError(
                f"production bars have symbols absent from stock master: {sorted(missing_master)[:3]}"
            )
        daily = bars.drop(columns="adj_factor")
        adj = bars.loc[:, ["ts_code", "trade_date", "adj_factor"]]
        result = build_production_year_frame(
            daily=daily,
            adj=adj,
            stock_master=self.stock_master,
            st_intervals=self.st_intervals,
            trade_calendar=self.trade_calendar,
            market_states=states,
        )
        if self.external_fields:
            assert self.dense_sidecars_dir is not None
            for source in sorted(set(
                PRODUCTION_EXTERNAL_FIELD_SOURCES[name]
                for name in self.external_fields
            )):
                fields = sorted(
                    name for name in self.external_fields
                    if PRODUCTION_EXTERNAL_FIELD_SOURCES[name] == source
                )
                sidecar = self._read_years(
                    self.dense_sidecars_dir / source,
                    years=years,
                    columns=["ts_code", "trade_date", *fields],
                    start=start,
                    end=end,
                    label=f"production {source} sidecar",
                )
                if sidecar.duplicated(["ts_code", "trade_date"]).any():
                    raise ValueError(f"production {source} sidecar contains duplicate keys")
                result = result.merge(
                    sidecar,
                    on=["ts_code", "trade_date"],
                    how="left",
                    validate="one_to_one",
                )
        if self.plugin_fields:
            assert self.plugin_field_source is not None
            plugin = self.plugin_field_source.read(
                fields=self.plugin_fields,
                start=start,
                end=end,
                symbols=self.symbols,
            )
            result = result.merge(
                plugin,
                on=["ts_code", "trade_date"],
                how="left",
                validate="one_to_one",
            )
        if self.index_memberships is not None:
            populated = {
                view: index_code
                for view, index_code in self.index_membership_views.items()
                if self.index_memberships["index_code"].astype(str).eq(index_code).any()
            }
            for view in set(self.index_membership_views).difference(populated):
                result[view] = False
            if populated:
                result = attach_index_membership_views(
                    result,
                    self.index_memberships,
                    views=populated,
                )
        if "forward_return" in result.columns:
            raise RuntimeError("production frame loader returned forward_return")
        return result

    def _read_years(
        self,
        root: Path,
        *,
        years: tuple[int, ...],
        columns: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        label: str,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for year in years:
            part = root / f"year={year}" / "part.parquet"
            if not part.is_file():
                raise FileNotFoundError(f"{label} partition is missing: {part}")
            if self.symbols is not None:
                selected: list[pd.DataFrame] = []
                wanted = set(self.symbols)
                for batch in pq.ParquetFile(part).iter_batches(
                    batch_size=65_536, columns=columns
                ):
                    frame = batch.to_pandas()
                    frame = frame[frame["ts_code"].astype(str).isin(wanted)].copy()
                    if frame.empty:
                        continue
                    frame["trade_date"] = pd.to_datetime(
                        frame["trade_date"], errors="coerce"
                    )
                    selected.append(
                        frame[frame["trade_date"].between(start, end)].copy()
                    )
                if selected:
                    frames.append(pd.concat(selected, ignore_index=True))
                continue
            date_type = pq.read_schema(part).field("trade_date").type
            if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
                lower: object = start.strftime("%Y%m%d")
                upper: object = end.strftime("%Y%m%d")
            elif pa.types.is_date(date_type):
                lower = start.date()
                upper = end.date()
            else:
                lower = start
                upper = end
            filters: list[tuple[str, str, object]] = [
                ("trade_date", ">=", lower),
                ("trade_date", "<=", upper),
            ]
            frames.append(pd.read_parquet(part, columns=columns, filters=filters))
        if not frames:
            return pd.DataFrame(columns=columns)
        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
        if result["trade_date"].isna().any():
            raise ValueError(f"{label} contains invalid trade dates")
        return result[result["trade_date"].between(start, end)].copy()


class BatchedProductionFrameLoader:
    """Build a full cross-section from bounded symbol batches."""

    def __init__(
        self,
        *,
        bars_dir: Path,
        market_states_dir: Path,
        stock_master_path: Path,
        st_intervals_path: Path,
        trade_calendar_path: Path,
        batch_size: int = 256,
        dense_sidecars_dir: Path | None = None,
        external_fields: set[str] | None = None,
        plugin_manifest_paths: Sequence[Path] | None = None,
        plugin_fields: set[str] | None = None,
        plugin_field_source: ProductionPluginFieldSource | None = None,
        index_membership_path: Path | None = None,
        index_membership_views: Mapping[str, str] | None = None,
        symbols: Sequence[str] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("production symbol batch size must be positive")
        master = pd.read_parquet(stock_master_path, columns=["ts_code"])
        available = set(master["ts_code"].dropna().astype(str).unique())
        if symbols is None:
            selected_symbols = tuple(sorted(available))
        else:
            requested = tuple(sorted(str(code) for code in symbols))
            if not requested or len(set(requested)) != len(requested):
                raise ValueError("production batched loader symbols must be unique")
            unknown = sorted(set(requested).difference(available))
            if unknown:
                raise ValueError(
                    f"production batched loader symbols are absent from master: {unknown[:3]}"
                )
            selected_symbols = requested
        if not selected_symbols:
            raise ValueError("production batched loader stock master is empty")
        if plugin_manifest_paths and plugin_field_source is not None:
            raise ValueError(
                "provide production plugin manifests or field source, not both"
            )
        plugin_source = plugin_field_source or (
            ProductionPluginFieldSource(plugin_manifest_paths)
            if plugin_manifest_paths
            else None
        )
        self.plugin_field_source = plugin_source
        self.loaders = tuple(
            ProductionFrameLoader(
                bars_dir=bars_dir,
                market_states_dir=market_states_dir,
                stock_master_path=stock_master_path,
                st_intervals_path=st_intervals_path,
                trade_calendar_path=trade_calendar_path,
                symbols=selected_symbols[offset : offset + batch_size],
                dense_sidecars_dir=dense_sidecars_dir,
                external_fields=external_fields,
                plugin_fields=plugin_fields,
                plugin_field_source=plugin_source,
                index_membership_path=index_membership_path,
                index_membership_views=index_membership_views,
            )
            for offset in range(0, len(selected_symbols), batch_size)
        )
        self.symbol_count = len(selected_symbols)
        self.batch_size = batch_size
        self.additional_field_specs = dict(self.loaders[0].additional_field_specs)
        self.additional_dataset_versions = dict(
            self.loaders[0].additional_dataset_versions
        )

    def __call__(self, load_start: pd.Timestamp, load_end: pd.Timestamp) -> pd.DataFrame:
        frames = list(self.iter_frames(load_start, load_end))
        if not frames:
            raise ValueError("production batched loader returned no rows")
        result = pd.concat(frames, ignore_index=True)
        if result.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("production symbol batches contain duplicate keys")
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def iter_frames(
        self, load_start: pd.Timestamp, load_end: pd.Timestamp
    ):
        cumulative_rows = 0
        for position, loader in enumerate(self.loaders, start=1):
            try:
                frame = loader(load_start, load_end)
            except ValueError as error:
                if str(error) != "production bar partitions contain no requested rows":
                    raise
                frame = pd.DataFrame()
            if not frame.empty:
                cumulative_rows += len(frame)
                yield frame
            if position % 5 == 0 or position == len(self.loaders):
                print(
                    f"[production-symbol-batch] batch={position}/{len(self.loaders)} "
                    f"rows={cumulative_rows}",
                    flush=True,
                )
            gc.collect()
            pa.default_memory_pool().release_unused()
