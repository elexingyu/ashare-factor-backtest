"""Minimal column-pruned Parquet port for validated data plugins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import os
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.dataset as ds
import yaml

from ashare_factor_backtest.data.manifest import content_fingerprint, read_manifest
from ashare_factor_backtest.plugins.data_manifest import load_dataset_manifest
from ashare_factor_backtest.plugins.validator import (
    validate_plugin,
    validate_plugin_cached,
)


_CANONICAL_FIELDS = {
    "open": ("price", "raw", "bar_open", "09:30"),
    "high": ("price", "raw", "bar_close", "15:00"),
    "low": ("price", "raw", "bar_close", "15:00"),
    "close": ("price", "raw", "bar_close", "15:00"),
    "pre_close": ("price", "raw", "bar_open", "09:30"),
    "volume_shares": ("shares", None, "bar_close", "15:00"),
    "amount_cny": ("currency_cny", None, "bar_close", "15:00"),
    "adj_factor": ("adjustment_factor", None, "bar_close", "15:00"),
}
_MANIFEST_DTYPES = {
    "bool": "bool",
    "double": "float64",
    "float": "float32",
    "int32": "int32",
    "int64": "int64",
    "large_string": "string",
    "string": "string",
}


@dataclass(frozen=True)
class ScanRequest:
    fields: tuple[str, ...]
    start: date | None = None
    end: date | None = None
    entity_filters: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("fields must be a non-empty unique tuple")
        if (self.start is not None and not isinstance(self.start, date)) or (
            self.end is not None and not isinstance(self.end, date)
        ):
            raise TypeError("scan start and end must be dates")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("scan start must not exceed end")
        keys = [key for key, _ in self.entity_filters]
        if len(set(keys)) != len(keys):
            raise ValueError("entity filter keys must be unique")
        for key, values in self.entity_filters:
            if not key or not values or len(set(values)) != len(values):
                raise ValueError("entity filters must contain non-empty unique values")


class ParquetDataAdapter:
    def __init__(
        self,
        manifest_path: Path,
        *,
        validation_cache_root: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        result = (
            validate_plugin(self.manifest_path)
            if validation_cache_root is None
            else validate_plugin_cached(
                self.manifest_path, Path(validation_cache_root)
            )[0]
        )
        if not result.valid:
            labels = ", ".join(issue.code for issue in result.issues)
            raise ValueError(f"data plugin validation failed: {labels}")
        self.manifest = load_dataset_manifest(self.manifest_path)
        self._dataset = ds.dataset(
            [str(self.manifest_path.parent / item) for item in self.manifest.paths],
            format="parquet",
        )

    def scan(self, request: ScanRequest) -> pa.Table:
        manifest = self.manifest
        declared = {field.name for field in manifest.fields}
        unknown = [field for field in request.fields if field not in declared]
        if unknown:
            raise ValueError(f"unknown plugin fields: {', '.join(unknown)}")
        unknown_keys = [
            key for key, _ in request.entity_filters if key not in manifest.entity_keys
        ]
        if unknown_keys:
            raise ValueError(f"unknown plugin entity keys: {', '.join(unknown_keys)}")
        columns = [*manifest.entity_keys, manifest.time_key, *request.fields]
        predicate = None
        time_type = self._dataset.schema.field(manifest.time_key).type
        if request.start is not None:
            predicate = ds.field(manifest.time_key) >= _date_scalar(
                request.start, time_type
            )
        if request.end is not None:
            condition = ds.field(manifest.time_key) <= _date_scalar(
                request.end, time_type
            )
            predicate = condition if predicate is None else predicate & condition
        for key, values in request.entity_filters:
            condition = ds.field(key).isin(values)
            predicate = condition if predicate is None else predicate & condition
        return self._dataset.to_table(columns=columns, filter=predicate)


def _date_scalar(value: date, arrow_type: pa.DataType) -> pa.Scalar:
    if pa.types.is_date(arrow_type):
        return pa.scalar(value, type=arrow_type)
    if pa.types.is_timestamp(arrow_type):
        return pa.scalar(datetime.combine(value, time.min), type=arrow_type)
    raise ValueError("plugin time key must use a date or timestamp parquet type")


def export_canonical_daily_plugin(
    data_path: Path,
    canonical_manifest_path: Path,
    plugin_manifest_path: Path,
) -> Path:
    """Expose a PASS canonical daily dataset through the public plugin contract."""
    canonical = read_manifest(canonical_manifest_path)
    if canonical.version.dataset != "canonical_daily":
        raise ValueError("canonical manifest dataset must be canonical_daily")

    source = Path(data_path).resolve(strict=True)
    target = Path(plugin_manifest_path)
    target_parent = target.parent.resolve(strict=True)
    paths = _canonical_parquet_paths(source)
    try:
        relative_paths = tuple(
            path.relative_to(target_parent).as_posix() for path in paths
        )
    except ValueError as error:
        raise ValueError(
            "plugin manifest must be stored at or above its parquet files"
        ) from error

    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    fields = []
    for name, (unit, basis, observed, available_time) in _CANONICAL_FIELDS.items():
        if name not in dataset.schema.names:
            continue
        arrow_type = str(dataset.schema.field(name).type)
        try:
            dtype = _MANIFEST_DTYPES[arrow_type]
        except KeyError as error:
            raise ValueError(
                f"unsupported canonical dtype for {name}: {arrow_type}"
            ) from error
        fields.append(
            {
                "available_at": {"day_offset": 0, "time": available_time},
                "dtype": dtype,
                "name": name,
                "observed_at": observed,
                "price_basis": basis,
                "unit_lineage": unit,
            }
        )
    if not fields:
        raise ValueError("canonical daily dataset contains no supported factor fields")

    payload = {
        "content_hash": content_fingerprint(paths),
        "coverage_end": canonical.coverage_end.isoformat(),
        "coverage_start": canonical.coverage_start.isoformat(),
        "dataset": "canonical_daily",
        "entity_keys": ["ts_code"],
        "fields": fields,
        "format": "parquet",
        "frequency": "1d",
        "immutable": True,
        "paths": list(relative_paths),
        "row_count": canonical.row_count,
        "schema_version": "1",
        "time_key": "trade_date",
        "timezone": "Asia/Shanghai",
        "version": canonical.version.version,
    }
    _write_yaml_atomic(target, payload)
    result = validate_plugin(target)
    if not result.valid:
        labels = ", ".join(issue.code for issue in result.issues)
        target.unlink(missing_ok=True)
        raise ValueError(f"exported canonical plugin failed validation: {labels}")
    return target


def _canonical_parquet_paths(source: Path) -> tuple[Path, ...]:
    if source.is_file() and source.suffix == ".parquet":
        return (source,)
    if not source.is_dir():
        raise ValueError("canonical data path must be a parquet file or directory")
    paths = tuple(sorted(source.glob("year=*/part-*.parquet")))
    if not paths:
        raise ValueError("canonical directory contains no year partitions")
    return paths


def _write_yaml_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
