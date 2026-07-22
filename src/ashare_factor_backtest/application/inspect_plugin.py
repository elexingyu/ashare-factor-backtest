"""Machine-facing inspection of immutable data plugin contracts."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Sequence

from ashare_factor_backtest.plugins.data_manifest import (
    DatasetManifest,
    load_dataset_manifest,
)
from ashare_factor_backtest.plugins.validator import validate_plugin


_NUMERIC_DTYPES = frozenset({"float32", "float64", "int32", "int64"})


class PluginInspectionService:
    def execute(
        self, manifest_path: Path, *, fields: Sequence[str] = ()
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        result = validate_plugin(Path(manifest_path))
        if not result.valid:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in result.issues
            )
            raise ValueError(f"data plugin validation failed: {details}")
        manifest = load_dataset_manifest(Path(manifest_path))
        return self.describe(manifest, fields=fields)

    def describe(
        self, manifest: DatasetManifest, *, fields: Sequence[str] = ()
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        """Describe a manifest already validated by a retained adapter."""
        selected = tuple(fields)
        if len(set(selected)) != len(selected):
            raise ValueError("plugin inspection fields must be unique")
        declared = {field.name: field for field in manifest.fields}
        unknown = sorted(set(selected).difference(declared))
        if unknown:
            raise ValueError(f"unknown plugin fields: {unknown}")
        inspected = (
            tuple(declared[name] for name in selected)
            if selected
            else tuple(sorted(manifest.fields, key=lambda item: item.name))
        )
        field_payloads = []
        warnings = []
        for field in inspected:
            eligible = (
                field.dtype in _NUMERIC_DTYPES
                and field.available_at.day_offset == 0
                and field.available_at.time <= time(15, 0)
            )
            if not eligible:
                warnings.append(
                    f"Field {field.name} requires an availability transform before "
                    "production next-open evaluation."
                )
            field_payloads.append(
                {
                    "available_at": {
                        "day_offset": field.available_at.day_offset,
                        "time": field.available_at.time.isoformat(timespec="minutes"),
                    },
                    "dtype": field.dtype,
                    "name": field.name,
                    "observed_at": field.observed_at,
                    "price_basis": (
                        field.price_basis.value if field.price_basis else None
                    ),
                    "production_next_open_eligible": eligible,
                    "unit_lineage": field.unit_lineage,
                }
            )
        return (
            {
                "coverage": {
                    "end": manifest.coverage_end.isoformat(),
                    "start": manifest.coverage_start.isoformat(),
                },
                "dataset": manifest.dataset,
                "fields": field_payloads,
                "frequency": manifest.frequency,
                "manifest_identity": manifest.identity,
                "provenance": dict(manifest.provenance or {}),
                "row_count": manifest.row_count,
                "timezone": manifest.timezone,
                "version": manifest.version,
            },
            tuple(warnings),
        )
