"""Static and streamed validation for external data plugins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

import pyarrow as pa
import pyarrow.dataset as ds

from ashare_factor_backtest.data.manifest import content_fingerprint
from ashare_factor_backtest.plugins.data_manifest import (
    DatasetManifest,
    load_dataset_manifest,
)


_OBSERVED_MINIMUM = {
    "bar_open": (0, 9, 30, 0),
    "bar_close": (0, 15, 0, 0),
    "event_time": (0, 0, 0, 0),
}
_FIELD_OBSERVATION_POLICY = {
    "open": "bar_open",
    "high": "bar_close",
    "low": "bar_close",
    "close": "bar_close",
    "volume": "bar_close",
    "amount": "bar_close",
    "vwap": "bar_close",
    "adj_factor": "bar_close",
}
_ARROW_DTYPES = {
    "bool": {"bool"},
    "float32": {"float"},
    "float64": {"double"},
    "int32": {"int32"},
    "int64": {"int64"},
    "string": {"string", "large_string"},
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    manifest_identity: str | None
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class _ScanStats:
    duplicate_keys: bool
    row_count: int
    coverage_start: date | None
    coverage_end: date | None


def validate_plugin(path: Path) -> ValidationResult:
    manifest_path = Path(path)
    try:
        manifest = load_dataset_manifest(manifest_path)
    except ValueError as error:
        return ValidationResult(
            False, None, (ValidationIssue("MANIFEST_INVALID", str(error)),)
        )

    issues: list[ValidationIssue] = []
    data_paths = tuple(manifest_path.parent / item for item in manifest.paths)
    try:
        actual_hash = content_fingerprint(data_paths)
    except ValueError as error:
        issues.append(ValidationIssue("CONTENT_UNREADABLE", str(error)))
        return ValidationResult(False, manifest.identity, tuple(issues))
    if actual_hash != manifest.content_hash:
        issues.append(
            ValidationIssue(
                "CONTENT_HASH_MISMATCH",
                "declared content_hash does not match the current file bytes",
            )
        )

    for field in manifest.fields:
        expected_observation = _FIELD_OBSERVATION_POLICY.get(field.name)
        if (
            expected_observation is not None
            and field.observed_at != expected_observation
        ):
            issues.append(
                ValidationIssue(
                    "FIELD_SEMANTIC_TIME",
                    f"{field.name} must be observed at {expected_observation}",
                )
            )
        if field.available_at.sort_key() < _OBSERVED_MINIMUM[field.observed_at]:
            issues.append(
                ValidationIssue(
                    "AVAILABLE_BEFORE_OBSERVED",
                    f"{field.name} is declared available before it can be observed",
                )
            )

    try:
        dataset = ds.dataset([str(item) for item in data_paths], format="parquet")
    except (OSError, pa.ArrowException) as error:
        issues.append(ValidationIssue("PARQUET_UNREADABLE", str(error)))
        return ValidationResult(False, manifest.identity, tuple(issues))

    issues.extend(_schema_issues(dataset.schema, manifest))
    if not any(issue.code == "SCHEMA_MISMATCH" for issue in issues):
        stats = _scan_primary_keys(dataset, (*manifest.entity_keys, manifest.time_key))
        if stats.duplicate_keys:
            issues.append(
                ValidationIssue(
                    "DUPLICATE_PRIMARY_KEY",
                    "entity/time primary key contains duplicates",
                )
            )
        if stats.row_count != manifest.row_count:
            issues.append(
                ValidationIssue(
                    "ROW_COUNT_MISMATCH",
                    f"declared row_count {manifest.row_count} != observed {stats.row_count}",
                )
            )
        if (
            stats.coverage_start != manifest.coverage_start
            or stats.coverage_end != manifest.coverage_end
        ):
            issues.append(
                ValidationIssue(
                    "COVERAGE_MISMATCH",
                    "declared coverage does not match observed time-key range",
                )
            )
    return ValidationResult(not issues, manifest.identity, tuple(issues))


def validate_plugin_cached(
    path: Path,
    cache_root: Path,
) -> tuple[ValidationResult, bool]:
    """Reuse a successful immutable-plugin audit while source metadata is unchanged."""
    manifest_path = Path(path).resolve()
    try:
        manifest = load_dataset_manifest(manifest_path)
        signature = _plugin_file_signature(manifest_path, manifest)
    except (OSError, ValueError):
        return validate_plugin(manifest_path), False
    cache_key = hashlib.sha256(
        f"{manifest_path}\0{manifest.identity}".encode("utf-8")
    ).hexdigest()
    receipt_path = Path(cache_root) / f"{cache_key}.json"
    receipt = _read_validation_receipt(receipt_path)
    if (
        receipt is not None
        and receipt.get("schema") == "plugin-validation-receipt.v1"
        and receipt.get("manifest_identity") == manifest.identity
        and receipt.get("file_signature") == signature
    ):
        return ValidationResult(True, manifest.identity, ()), True

    result = validate_plugin(manifest_path)
    if result.valid:
        _write_validation_receipt(
            receipt_path,
            {
                "schema": "plugin-validation-receipt.v1",
                "manifest_identity": manifest.identity,
                "file_signature": signature,
            },
        )
    else:
        receipt_path.unlink(missing_ok=True)
    return result, False


def _plugin_file_signature(
    manifest_path: Path,
    manifest: DatasetManifest,
) -> dict[str, object]:
    paths = (manifest_path, *(manifest_path.parent / item for item in manifest.paths))
    return {
        "files": [
            {
                "mtime_ns": path.stat().st_mtime_ns,
                "path": str(path.resolve(strict=True)),
                "size": path.stat().st_size,
            }
            for path in paths
        ],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _read_validation_receipt(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    digest = payload.get("content_digest")
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    if not isinstance(digest, str) or digest != _json_digest(body):
        return None
    return payload


def _write_validation_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["content_digest"] = _json_digest(body)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(body, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _schema_issues(
    schema: pa.Schema, manifest: DatasetManifest
) -> Iterable[ValidationIssue]:
    required = (
        *manifest.entity_keys,
        manifest.time_key,
        *(field.name for field in manifest.fields),
    )
    missing = [name for name in required if name not in schema.names]
    if missing:
        yield ValidationIssue(
            "SCHEMA_MISMATCH", f"missing columns: {', '.join(missing)}"
        )
        return
    for field in manifest.fields:
        actual = str(schema.field(field.name).type)
        if actual not in _ARROW_DTYPES[field.dtype]:
            yield ValidationIssue(
                "SCHEMA_MISMATCH",
                f"field {field.name} declares {field.dtype} but parquet contains {actual}",
            )


def _scan_primary_keys(dataset: ds.Dataset, keys: tuple[str, ...]) -> _ScanStats:
    time_index = len(keys) - 1
    duplicate = False
    row_count = 0
    coverage_start: date | None = None
    coverage_end: date | None = None
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as handle:
        connection = sqlite3.connect(handle.name)
        try:
            connection.execute("CREATE TABLE observed_keys (key TEXT PRIMARY KEY)")
            scanner = dataset.scanner(columns=list(keys), batch_size=65_536)
            for batch in scanner.to_batches():
                columns = [
                    batch.column(index).to_pylist() for index in range(len(keys))
                ]
                row_count += batch.num_rows
                dates = [_as_date(value) for value in columns[time_index]]
                if any(value is None for value in dates):
                    duplicate = True
                finite_dates = [value for value in dates if value is not None]
                if finite_dates:
                    batch_start = min(finite_dates)
                    batch_end = max(finite_dates)
                    coverage_start = (
                        batch_start
                        if coverage_start is None
                        else min(coverage_start, batch_start)
                    )
                    coverage_end = (
                        batch_end
                        if coverage_end is None
                        else max(coverage_end, batch_end)
                    )
                encoded = [repr(tuple(values)) for values in zip(*columns, strict=True)]
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO observed_keys(key) VALUES (?)",
                    ((value,) for value in encoded),
                )
                if connection.total_changes - before != len(encoded):
                    duplicate = True
            return _ScanStats(duplicate, row_count, coverage_start, coverage_end)
        finally:
            connection.close()


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None
