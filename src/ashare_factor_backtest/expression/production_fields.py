"""Public daily field contract shared by evaluation and candidate generation."""

from __future__ import annotations

from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.catalog import FieldCatalog
from ashare_factor_backtest.expression.model import FieldSpec, ValueType


PRICE_FIELDS = ("open", "high", "low", "close")
ACTIVITY_FIELDS = ("volume", "amount")
PRODUCTION_FIELDS = PRICE_FIELDS + ACTIVITY_FIELDS


def build_production_field_catalog() -> FieldCatalog:
    catalog = FieldCatalog("production_daily_fields_v1")
    for name in PRICE_FIELDS:
        catalog.register(
            FieldSpec(
                name=name,
                value_type=ValueType.PANEL_FLOAT,
                available_at="15:00",
                price_basis=PriceBasis.HFQ_PIT,
                unit_lineage="price",
                dataset_version="canonical_daily",
                min_date="unknown",
                max_date="unknown",
                coverage_note=(
                    "public daily contract; availability checked before evaluation"
                ),
            )
        )
    for name, unit in (("volume", "shares"), ("amount", "currency")):
        catalog.register(
            FieldSpec(
                name=name,
                value_type=ValueType.PANEL_FLOAT,
                available_at="15:00",
                price_basis=None,
                unit_lineage=unit,
                dataset_version="canonical_daily",
                min_date="unknown",
                max_date="unknown",
                coverage_note=(
                    "public daily contract; availability checked before evaluation"
                ),
            )
        )
    return catalog
