"""Bridge validated daily plugins into the production expression frame."""

from __future__ import annotations

from collections import defaultdict
from datetime import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ashare_factor_backtest.plugins.parquet_adapter import ParquetDataAdapter, ScanRequest
from ashare_factor_backtest.plugins.data_manifest import DatasetManifest, FieldManifest


class ProductionPluginFieldSource:
    """Resolve and scan immutable plugin fields under next-open timing semantics."""

    def __init__(
        self,
        manifest_paths: Sequence[Path],
        *,
        validation_cache_root: Path | None = None,
    ) -> None:
        paths = tuple(Path(path) for path in manifest_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("production plugin manifests must be non-empty and unique")
        self._adapters = tuple(
            ParquetDataAdapter(path, validation_cache_root=validation_cache_root)
            for path in paths
        )
        owners: dict[str, ParquetDataAdapter] = {}
        contracts: dict[str, tuple[FieldManifest, DatasetManifest]] = {}
        datasets: dict[str, str] = {}
        for adapter in self._adapters:
            manifest = adapter.manifest
            if (
                manifest.entity_keys != ("ts_code",)
                or manifest.time_key != "trade_date"
            ):
                raise ValueError(
                    "production plugins require entity key ts_code and time key trade_date"
                )
            for field in manifest.fields:
                if field.name in owners:
                    raise ValueError(f"duplicate production plugin field: {field.name}")
                owners[field.name] = adapter
                contracts[field.name] = (field, manifest)
            existing = datasets.get(manifest.dataset)
            if existing is not None and existing != manifest.identity:
                raise ValueError(
                    f"production plugin dataset has multiple identities: {manifest.dataset}"
                )
            datasets[manifest.dataset] = manifest.identity
        self._owners = owners
        self._contracts = contracts
        self._dataset_versions = datasets
        self._manifests_by_path = {
            adapter.manifest_path.resolve(): adapter.manifest
            for adapter in self._adapters
        }

    @property
    def available_fields(self) -> frozenset[str]:
        return frozenset(self._owners)

    @property
    def field_contracts(self) -> dict[str, tuple[FieldManifest, DatasetManifest]]:
        return dict(self._contracts)

    @property
    def dataset_versions(self) -> dict[str, str]:
        return dict(self._dataset_versions)

    @property
    def manifests_by_path(self) -> dict[Path, DatasetManifest]:
        return dict(self._manifests_by_path)

    def read(
        self,
        *,
        fields: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        selected = tuple(sorted(set(fields)))
        if not selected:
            raise ValueError("production plugin fields must not be empty")
        unknown = sorted(set(selected).difference(self._owners))
        if unknown:
            raise ValueError(f"unknown production plugin fields: {unknown}")
        grouped: dict[ParquetDataAdapter, list[str]] = defaultdict(list)
        for name in selected:
            adapter = self._owners[name]
            field = next(item for item in adapter.manifest.fields if item.name == name)
            if field.available_at.day_offset != 0 or field.available_at.time > time(
                15, 0
            ):
                raise ValueError(
                    f"production plugin field {name} is unavailable by observation-day close"
                )
            grouped[adapter].append(name)

        result: pd.DataFrame | None = None
        entity_filters = (
            (("ts_code", tuple(str(code) for code in symbols)),) if symbols else ()
        )
        for adapter, names in grouped.items():
            table = adapter.scan(
                ScanRequest(
                    fields=tuple(sorted(names)),
                    start=pd.Timestamp(start).date(),
                    end=pd.Timestamp(end).date(),
                    entity_filters=entity_filters,
                )
            )
            frame = table.to_pandas()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
            if frame["trade_date"].isna().any():
                raise ValueError("production plugin contains invalid trade dates")
            if frame.duplicated(["ts_code", "trade_date"]).any():
                raise ValueError("production plugin contains duplicate business keys")
            result = (
                frame
                if result is None
                else result.merge(
                    frame,
                    on=["ts_code", "trade_date"],
                    how="outer",
                    validate="one_to_one",
                )
            )
        assert result is not None
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
