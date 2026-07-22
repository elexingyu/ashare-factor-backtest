"""Typed, immutable contracts for external factor-factory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from ashare_factor_backtest.contracts import PriceBasis


_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DTYPES = frozenset({"bool", "float32", "float64", "int32", "int64", "string"})
_OBSERVATION_POINTS = frozenset({"bar_open", "bar_close", "event_time"})


@dataclass(frozen=True)
class Availability:
    day_offset: int
    time: time

    def __post_init__(self) -> None:
        if isinstance(self.day_offset, bool) or not isinstance(self.day_offset, int):
            raise TypeError("available_at.day_offset must be an integer")
        if self.day_offset < 0:
            raise ValueError("available_at.day_offset must be non-negative")
        if not isinstance(self.time, time) or self.time.tzinfo is not None:
            raise ValueError("available_at.time must be a timezone-naive local time")

    def sort_key(self) -> tuple[int, int, int, int]:
        return self.day_offset, self.time.hour, self.time.minute, self.time.second


@dataclass(frozen=True)
class FieldManifest:
    name: str
    dtype: str
    unit_lineage: str
    price_basis: PriceBasis | None
    observed_at: str
    available_at: Availability

    def __post_init__(self) -> None:
        _require_name("field name", self.name)
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported field dtype: {self.dtype}")
        _require_text("unit_lineage", self.unit_lineage)
        if self.observed_at not in _OBSERVATION_POINTS:
            raise ValueError(f"unsupported observed_at: {self.observed_at}")
        if not isinstance(self.available_at, Availability):
            raise TypeError("available_at must be an Availability")
        if self.unit_lineage == "price" and not isinstance(self.price_basis, PriceBasis):
            raise ValueError("price fields must declare a supported price_basis")
        if self.price_basis is not None and not isinstance(self.price_basis, PriceBasis):
            raise ValueError("unsupported price_basis")


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset: str
    version: str
    immutable: bool
    format: str
    paths: tuple[str, ...]
    frequency: str
    timezone: str
    entity_keys: tuple[str, ...]
    time_key: str
    coverage_start: date
    coverage_end: date
    row_count: int
    content_hash: str
    fields: tuple[FieldManifest, ...]
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("unsupported schema_version")
        _require_name("dataset", self.dataset)
        if not isinstance(self.version, str) or _VERSION.fullmatch(self.version) is None:
            raise ValueError("version must be immutable and normalized")
        if self.immutable is not True:
            raise ValueError("immutable must be true")
        if self.format != "parquet":
            raise ValueError("only parquet format is supported")
        if not self.paths:
            raise ValueError("paths must not be empty")
        for value in self.paths:
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
                raise ValueError("paths must be relative parquet paths without traversal")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("paths must be unique")
        if self.frequency != "1d":
            raise ValueError("prototype supports frequency 1d only")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
            raise ValueError("timezone must be a valid IANA timezone") from error
        if not self.entity_keys:
            raise ValueError("entity_keys must not be empty")
        for key in self.entity_keys:
            _require_name("entity key", key)
        _require_name("time_key", self.time_key)
        if self.time_key in self.entity_keys:
            raise ValueError("time_key must differ from entity_keys")
        if not isinstance(self.coverage_start, date) or isinstance(self.coverage_start, datetime):
            raise ValueError("coverage_start must be an ISO date")
        if not isinstance(self.coverage_end, date) or isinstance(self.coverage_end, datetime):
            raise ValueError("coverage_end must be an ISO date")
        if self.coverage_start > self.coverage_end:
            raise ValueError("coverage_start must not be after coverage_end")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        if not self.fields:
            raise ValueError("fields must not be empty")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        if self.provenance is not None:
            if not isinstance(self.provenance, Mapping) or not self.provenance:
                raise ValueError("provenance must be a non-empty string mapping")
            for key, value in self.provenance.items():
                _require_name("provenance key", key)
                _require_text("provenance value", value)

    def canonical_bytes(self) -> bytes:
        payload = {
            "content_hash": self.content_hash,
            "coverage_end": self.coverage_end.isoformat(),
            "coverage_start": self.coverage_start.isoformat(),
            "dataset": self.dataset,
            "entity_keys": list(self.entity_keys),
            "fields": [
                {
                    "available_at": {
                        "day_offset": field.available_at.day_offset,
                        "time": field.available_at.time.isoformat(timespec="minutes"),
                    },
                    "dtype": field.dtype,
                    "name": field.name,
                    "observed_at": field.observed_at,
                    "price_basis": field.price_basis.value if field.price_basis else None,
                    "unit_lineage": field.unit_lineage,
                }
                for field in self.fields
            ],
            "format": self.format,
            "frequency": self.frequency,
            "immutable": self.immutable,
            "paths": list(self.paths),
            "schema_version": self.schema_version,
            "row_count": self.row_count,
            "time_key": self.time_key,
            "timezone": self.timezone,
            "version": self.version,
        }
        if self.provenance is not None:
            payload["provenance"] = dict(sorted(self.provenance.items()))
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_dataset_manifest(path: Path) -> DatasetManifest:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read dataset manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("dataset manifest must contain an object")
    try:
        fields = tuple(_field_from_payload(item) for item in payload["fields"])
        return DatasetManifest(
            schema_version=payload["schema_version"],
            dataset=payload["dataset"],
            version=payload["version"],
            immutable=payload["immutable"],
            format=payload["format"],
            paths=tuple(payload["paths"]),
            frequency=payload["frequency"],
            timezone=payload["timezone"],
            entity_keys=tuple(payload["entity_keys"]),
            time_key=payload["time_key"],
            coverage_start=_parse_date("coverage_start", payload["coverage_start"]),
            coverage_end=_parse_date("coverage_end", payload["coverage_end"]),
            row_count=payload["row_count"],
            content_hash=payload["content_hash"],
            fields=fields,
            provenance=_provenance_from_payload(payload.get("provenance")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid dataset manifest: {error}") from error


def _field_from_payload(payload: Any) -> FieldManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("fields must contain objects")
    available = payload.get("available_at")
    if not isinstance(available, Mapping):
        raise ValueError("available_at must be an object")
    price_basis_value = payload.get("price_basis")
    try:
        basis = PriceBasis(price_basis_value) if price_basis_value is not None else None
    except ValueError as error:
        raise ValueError(f"unsupported price_basis: {price_basis_value}") from error
    try:
        parsed_time = time.fromisoformat(available["time"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("available_at.time must be normalized HH:MM") from error
    return FieldManifest(
        name=payload["name"],
        dtype=payload["dtype"],
        unit_lineage=payload["unit_lineage"],
        price_basis=basis,
        observed_at=payload["observed_at"],
        available_at=Availability(available["day_offset"], parsed_time),
    )


def _provenance_from_payload(payload: Any) -> Mapping[str, str] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("provenance must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def _require_name(label: str, value: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must use lowercase snake_case")


def _require_text(label: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized non-empty string")


def _parse_date(label: str, value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date") from error
