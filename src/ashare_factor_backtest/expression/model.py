"""Immutable data contracts for formula-factor expressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import TypeAlias

from ashare_factor_backtest.contracts import PriceBasis


_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ValueType(str, Enum):
    PANEL_FLOAT = "panel_float"
    PANEL_BOOL = "panel_bool"
    SCALAR_INT = "scalar_int"
    SCALAR_FLOAT = "scalar_float"


TypeConstraint: TypeAlias = ValueType | tuple[ValueType, ...]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    value_type: ValueType
    available_at: str
    price_basis: PriceBasis | None
    unit_lineage: str
    dataset_version: str
    min_date: str
    max_date: str
    coverage_note: str

    def __post_init__(self) -> None:
        _normalized_name("field name", self.name)
        for label in (
            "available_at", "unit_lineage", "dataset_version", "min_date",
            "max_date", "coverage_note",
        ):
            _nonempty(label, getattr(self, label))
        if not isinstance(self.value_type, ValueType):
            raise TypeError("value_type must be ValueType")
        if self.unit_lineage == "price" and not isinstance(self.price_basis, PriceBasis):
            raise ValueError("price fields must declare price_basis")
        if self.price_basis is not None and not isinstance(self.price_basis, PriceBasis):
            raise TypeError("price_basis must be PriceBasis or None")


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    aliases: tuple[str, ...]
    version: str
    category: str
    input_types: tuple[TypeConstraint, ...]
    output_type: ValueType
    causal_contract: str
    nan_policy: str
    parameter_domain: tuple[object, ...]
    lookback_rule: str
    commutative: bool
    complexity_cost: int
    examples: tuple[str, ...]
    requires_panel_input: bool = False

    def __post_init__(self) -> None:
        _normalized_name("operator name", self.name)
        for alias in self.aliases:
            _normalized_name("operator alias", alias)
        if len(set(self.aliases)) != len(self.aliases) or self.name in self.aliases:
            raise ValueError("operator aliases must be unique and differ from name")
        for label in (
            "version", "category", "causal_contract", "nan_policy", "lookback_rule"
        ):
            _nonempty(label, getattr(self, label))
        if not self.input_types:
            raise ValueError("input_types must contain type constraints")
        for constraint in self.input_types:
            allowed = constraint if isinstance(constraint, tuple) else (constraint,)
            if (
                not allowed
                or not all(isinstance(value, ValueType) for value in allowed)
                or len(set(allowed)) != len(allowed)
            ):
                raise ValueError("input_types must contain unique ValueType constraints")
        if not isinstance(self.output_type, ValueType):
            raise TypeError("output_type must be ValueType")
        if not isinstance(self.requires_panel_input, bool):
            raise TypeError("requires_panel_input must be bool")
        if self.requires_panel_input and not any(
            self.accepts(index, ValueType.PANEL_FLOAT)
            for index in range(len(self.input_types))
        ):
            raise ValueError("requires_panel_input needs a panel-compatible argument")
        if isinstance(self.complexity_cost, bool) or self.complexity_cost <= 0:
            raise ValueError("complexity_cost must be a positive integer")
        if not self.examples or any(not value.strip() for value in self.examples):
            raise ValueError("examples must be non-empty")

    def accepts(self, index: int, actual: ValueType) -> bool:
        constraint = self.input_types[index]
        allowed = constraint if isinstance(constraint, tuple) else (constraint,)
        return actual in allowed


@dataclass(frozen=True)
class FieldNode:
    name: str


@dataclass(frozen=True)
class ConstantNode:
    value: int | float


@dataclass(frozen=True)
class CallNode:
    operator: str
    arguments: tuple["ExprNode", ...]


ExprNode: TypeAlias = FieldNode | ConstantNode | CallNode


@dataclass(frozen=True)
class CompiledExpression:
    root: ExprNode
    canonical: str
    factor_id: str
    output_type: ValueType
    lookback: int
    complexity: int
    price_basis: PriceBasis | None
    unit_lineage: str
    fields: tuple[str, ...]
    warnings: tuple[str, ...]


def _normalized_name(label: str, value: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must use lowercase snake_case")


def _nonempty(label: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a normalized non-empty string")
