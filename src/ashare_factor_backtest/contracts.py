"""Immutable contracts shared by the A-share research infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal


class PriceBasis(str, Enum):
    RAW = "raw"
    HFQ_PIT = "hfq_pit"
    QFQ_SNAPSHOT = "qfq_snapshot"


class PriceBasisMismatch(ValueError):
    """Raised when data from different price coordinate systems is combined."""


def require_same_price_basis(*bases: PriceBasis) -> None:
    """Reject implicit joins across raw and adjusted price coordinate systems."""
    if not bases:
        raise ValueError("at least one price basis is required")
    if not all(isinstance(basis, PriceBasis) for basis in bases):
        raise TypeError("price basis must be a PriceBasis")
    if len(set(bases)) != 1:
        labels = ", ".join(basis.value for basis in bases)
        raise PriceBasisMismatch(f"price basis mismatch: {labels}")


@dataclass(frozen=True)
class DataVersion:
    dataset: str
    version: str
    as_of: date
    schema_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_nonempty("dataset", self.dataset)
        _require_nonempty("version", self.version)
        _require_date("as_of", self.as_of)
        _require_nonempty("schema_hash", self.schema_hash)
        _require_nonempty("content_hash", self.content_hash)


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    family_id: str
    direction: Literal["high", "low"]
    available_at: str
    price_basis: PriceBasis

    def __post_init__(self) -> None:
        _require_nonempty("factor_id", self.factor_id)
        _require_nonempty("family_id", self.family_id)
        if self.direction not in {"high", "low"}:
            raise ValueError("direction must be 'high' or 'low'")
        _require_normalized_nonempty("available_at", self.available_at)
        if not isinstance(self.price_basis, PriceBasis):
            raise TypeError("price_basis must be a PriceBasis")


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_normalized_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")


def _require_date(name: str, value: date) -> None:
    if not isinstance(value, date) or hasattr(value, "hour"):
        raise TypeError(f"{name} must be a date")
