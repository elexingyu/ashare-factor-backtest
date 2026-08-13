"""Build compile-time field contracts from a production job manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ashare_factor_backtest.application.production_job import (
    ProductionJob,
    load_production_job,
)
from ashare_factor_backtest.evaluation.production_context import (
    build_production_field_catalog,
)
from ashare_factor_backtest.expression.catalog import FieldCatalog
from ashare_factor_backtest.expression.model import FieldSpec, ValueType
from ashare_factor_backtest.plugins.data_manifest import load_dataset_manifest


def build_job_field_catalog(
    job_path: Path,
    field_names: Iterable[str],
) -> FieldCatalog:
    """Build a return-free catalog for the requested fields in one job."""
    job = load_production_job(Path(job_path))
    required = tuple(dict.fromkeys(str(name) for name in field_names))
    if not required or any(not name or name != name.strip() for name in required):
        raise ValueError("job field catalog requires normalized source fields")
    plugin_specs = plugin_field_specs(job)
    plugin_fields = {field for binding in job.plugins for field in binding.fields}
    missing = sorted(set(required).intersection(plugin_fields).difference(plugin_specs))
    if missing:
        raise ValueError(f"production plugin fields lack contracts: {missing}")
    return build_production_field_catalog(
        field_names=set(required),
        date_range=(
            job.evaluation.start.isoformat(),
            job.evaluation.end.isoformat(),
        ),
        dataset_version=f"{job.dataset_version}_job_{job.contract_identity[:16]}",
        view=job.view,
        additional_field_specs={
            name: plugin_specs[name]
            for name in sorted(set(required).intersection(plugin_specs))
        },
    )


def plugin_field_specs(job: ProductionJob) -> dict[str, FieldSpec]:
    """Read declared plugin field metadata without loading market panels."""
    result: dict[str, FieldSpec] = {}
    for binding in job.plugins:
        manifest = load_dataset_manifest(binding.manifest)
        fields = {field.name: field for field in manifest.fields}
        missing = sorted(set(binding.fields).difference(fields))
        if missing:
            raise ValueError(
                f"production plugin binding fields missing from manifest: {missing}"
            )
        for name in binding.fields:
            field = fields[name]
            result[name] = FieldSpec(
                name=name,
                value_type=(
                    ValueType.PANEL_CATEGORY
                    if field.unit_lineage == "category"
                    else ValueType.PANEL_FLOAT
                ),
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
    return result
