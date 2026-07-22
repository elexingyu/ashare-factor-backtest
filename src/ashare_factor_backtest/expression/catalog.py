"""Deterministic field and operator catalogs."""

from __future__ import annotations

from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.expression.model import FieldSpec, OperatorSpec


class OperatorCatalog:
    def __init__(self, version: str) -> None:
        self.version = _version(version)
        self._specs: dict[str, OperatorSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._specs or spec.name in self._aliases:
            raise ExpressionError("UNKNOWN_OPERATOR", f"operator already registered: {spec.name}")
        for alias in spec.aliases:
            if alias in self._specs or alias in self._aliases:
                raise ExpressionError("UNKNOWN_OPERATOR", f"operator alias collision: {alias}")
        self._specs[spec.name] = spec
        self._aliases.update({alias: spec.name for alias in spec.aliases})

    def resolve(self, name: str) -> OperatorSpec:
        canonical = self._aliases.get(name, name)
        try:
            return self._specs[canonical]
        except KeyError as error:
            raise ExpressionError("UNKNOWN_OPERATOR", f"unknown operator: {name}") from error

    def specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def export_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "operators": [_operator_payload(spec) for spec in self.specs()],
        }


class FieldCatalog:
    def __init__(self, version: str) -> None:
        self.version = _version(version)
        self._specs: dict[str, FieldSpec] = {}

    def register(self, spec: FieldSpec) -> None:
        if spec.name in self._specs:
            raise ExpressionError("UNKNOWN_FIELD", f"field already registered: {spec.name}")
        self._specs[spec.name] = spec

    def resolve(self, name: str) -> FieldSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ExpressionError("UNKNOWN_FIELD", f"unknown field: {name}") from error

    def specs(self) -> tuple[FieldSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def version_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fields": [_field_payload(spec) for spec in self.specs()],
        }


def _operator_payload(spec: OperatorSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "aliases": list(spec.aliases),
        "version": spec.version,
        "category": spec.category,
        "input_types": [_type_payload(value) for value in spec.input_types],
        "output_type": spec.output_type.value,
        "causal_contract": spec.causal_contract,
        "nan_policy": spec.nan_policy,
        "parameter_domain": list(spec.parameter_domain),
        "lookback_rule": spec.lookback_rule,
        "commutative": spec.commutative,
        "complexity_cost": spec.complexity_cost,
        "examples": list(spec.examples),
        "requires_panel_input": spec.requires_panel_input,
    }


def _type_payload(value: object) -> str | list[str]:
    if isinstance(value, tuple):
        return [item.value for item in value]
    return value.value


def _field_payload(spec: FieldSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "value_type": spec.value_type.value,
        "available_at": spec.available_at,
        "price_basis": spec.price_basis.value if spec.price_basis else None,
        "unit_lineage": spec.unit_lineage,
        "dataset_version": spec.dataset_version,
        "min_date": spec.min_date,
        "max_date": spec.max_date,
        "coverage_note": spec.coverage_note,
    }


def _version(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("catalog version must be a normalized non-empty string")
    return value
