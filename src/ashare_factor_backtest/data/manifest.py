"""Versioned, immutable manifests for A-share research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
from pathlib import Path
import os
import stat
import tempfile
from typing import Any, Iterable, Mapping

from ashare_factor_backtest.contracts import DataVersion


FINGERPRINT_CHUNK_BYTES = 1024 * 1024


class ManifestStatus(str, Enum):
    PASS = "PASS"
    QUARANTINE = "QUARANTINE"
    FAIL = "FAIL"


class ManifestNotReadableError(RuntimeError):
    """Raised when a manifest is not eligible for normal dataset consumption."""


@dataclass(frozen=True)
class DataManifest:
    version: DataVersion
    source: str
    ingested_at: datetime
    coverage_start: date
    coverage_end: date
    row_count: int
    schema_hash: str
    content_hash: str
    parent_versions: tuple[DataVersion, ...]
    issues: tuple[str, ...]
    status: ManifestStatus

    def __post_init__(self) -> None:
        if not isinstance(self.version, DataVersion):
            raise TypeError("version must be a DataVersion")
        _require_nonempty("source", self.source)
        _require_utc("ingested_at", self.ingested_at)
        _require_date("coverage_start", self.coverage_start)
        _require_date("coverage_end", self.coverage_end)
        if self.coverage_start > self.coverage_end:
            raise ValueError("coverage_start must not be after coverage_end")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 0:
            raise ValueError("row_count must be a non-negative integer")
        _require_nonempty("schema_hash", self.schema_hash)
        _require_nonempty("content_hash", self.content_hash)
        if self.schema_hash != self.version.schema_hash:
            raise ValueError("schema_hash must match version.schema_hash")
        if self.content_hash != self.version.content_hash:
            raise ValueError("content_hash must match version.content_hash")
        _require_version_tuple("parent_versions", self.parent_versions)
        _require_string_tuple("issues", self.issues)
        if not isinstance(self.status, ManifestStatus):
            raise TypeError("status must be a ManifestStatus")

        object.__setattr__(self, "ingested_at", self.ingested_at.astimezone(timezone.utc))


Manifest = DataManifest


def content_fingerprint(paths: Iterable[Path]) -> str:
    """Hash the order-independent multiset of streamed file-content identities."""
    records: list[tuple[int, bytes]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            stable_path = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"fingerprint input does not exist: {path}") from error
        identifier = stable_path.as_posix()
        if identifier in seen:
            raise ValueError(f"duplicate fingerprint input: {stable_path}")
        seen.add(identifier)
        records.append(_fingerprint_file(stable_path))

    digest = hashlib.sha256()
    digest.update(b"astock_research.content_fingerprint.v2\0")
    for byte_size, content_hash in sorted(records):
        _update_fingerprint_field(digest, b"byte_size", byte_size.to_bytes(8, "big"))
        _update_fingerprint_field(digest, b"content_sha256", content_hash)
    return digest.hexdigest()


def _fingerprint_file(path: Path) -> tuple[int, bytes]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as error:
        raise ValueError(f"cannot open fingerprint input: {path}") from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"fingerprint input is not a regular file: {path}")

        content_digest = hashlib.sha256()
        bytes_read = 0
        while chunk := os.read(descriptor, FINGERPRINT_CHUNK_BYTES):
            bytes_read += len(chunk)
            content_digest.update(chunk)

        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise ValueError(f"fingerprint input changed while reading: {path}")
        if bytes_read != before.st_size:
            raise ValueError(f"fingerprint input size changed while reading: {path}")
        return bytes_read, content_digest.digest()
    except OSError as error:
        raise ValueError(f"cannot fingerprint input: {path}") from error
    finally:
        os.close(descriptor)


def _file_identity(file_stat: Any) -> tuple[int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def write_manifest(path: Path, manifest: DataManifest) -> None:
    """Atomically serialize a typed manifest with deterministic JSON formatting."""
    if not isinstance(manifest, DataManifest):
        raise TypeError("manifest must be a DataManifest")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_manifest(path: Path, *, allow_non_pass: bool = False) -> DataManifest:
    """Load a manifest, rejecting QUARANTINE and FAIL by default."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {source}: {error}") from error

    manifest = _manifest_from_dict(payload)
    if manifest.status is not ManifestStatus.PASS and not allow_non_pass:
        raise ManifestNotReadableError(
            f"manifest {source} is {manifest.status.value}; only PASS is readable by default"
        )
    return manifest


load_manifest = read_manifest


def _manifest_to_dict(manifest: DataManifest) -> dict[str, Any]:
    return {
        "content_hash": manifest.content_hash,
        "coverage_end": manifest.coverage_end.isoformat(),
        "coverage_start": manifest.coverage_start.isoformat(),
        "ingested_at": manifest.ingested_at.isoformat().replace("+00:00", "Z"),
        "issues": list(manifest.issues),
        "parent_versions": [_version_to_dict(version) for version in manifest.parent_versions],
        "row_count": manifest.row_count,
        "schema_hash": manifest.schema_hash,
        "source": manifest.source,
        "status": manifest.status.value,
        "version": _version_to_dict(manifest.version),
    }


def _manifest_from_dict(payload: Any) -> DataManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("manifest JSON must contain an object")
    try:
        return DataManifest(
            version=_version_from_dict(payload["version"]),
            source=payload["source"],
            ingested_at=_parse_datetime(payload["ingested_at"]),
            coverage_start=_parse_date(payload["coverage_start"]),
            coverage_end=_parse_date(payload["coverage_end"]),
            row_count=payload["row_count"],
            schema_hash=payload["schema_hash"],
            content_hash=payload["content_hash"],
            parent_versions=tuple(_version_from_dict(item) for item in payload["parent_versions"]),
            issues=tuple(payload["issues"]),
            status=ManifestStatus(payload["status"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid manifest payload: {error}") from error


def _version_to_dict(version: DataVersion) -> dict[str, str]:
    return {
        "as_of": version.as_of.isoformat(),
        "content_hash": version.content_hash,
        "dataset": version.dataset,
        "schema_hash": version.schema_hash,
        "version": version.version,
    }


def _version_from_dict(payload: Any) -> DataVersion:
    if not isinstance(payload, Mapping):
        raise ValueError("data version must be an object")
    return DataVersion(
        dataset=payload["dataset"],
        version=payload["version"],
        as_of=_parse_date(payload["as_of"]),
        schema_hash=payload["schema_hash"],
        content_hash=payload["content_hash"],
    )


def _parse_date(value: Any) -> date:
    if not isinstance(value, str):
        raise TypeError("date values must be ISO strings")
    return date.fromisoformat(value)


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("ingested_at must be an ISO datetime string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _update_fingerprint_field(
    digest: hashlib._Hash, label: bytes, value: bytes  # type: ignore[name-defined]
) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_date(name: str, value: date) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a date")


def _require_utc(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _require_version_tuple(name: str, value: tuple[DataVersion, ...]) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, DataVersion) for item in value):
        raise TypeError(f"{name} must be a tuple of DataVersion")


def _require_string_tuple(name: str, value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{name} must be a tuple of non-empty strings")
