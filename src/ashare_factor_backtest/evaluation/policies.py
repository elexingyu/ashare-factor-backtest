"""Evaluation policy value objects shared with optional research audits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionSelectionNullPolicy:
    permutations: int = 20
    maximum_portfolio_evaluations: int = 100_000
    maximum_empirical_p: float = 0.10
    minimum_changed_column_fraction: float = 0.50
    maximum_mean_coverage_drift: float = 0.01
    maximum_daily_coverage_drift: float = 0.10

    def __post_init__(self) -> None:
        if self.permutations <= 0 or self.maximum_portfolio_evaluations <= 0:
            raise ValueError("selection-null permutations and budget must be positive")
        for value in (
            self.maximum_empirical_p,
            self.minimum_changed_column_fraction,
            self.maximum_mean_coverage_drift,
            self.maximum_daily_coverage_drift,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("selection-null transform thresholds must be in [0, 1]")
        if not 0.0 < self.maximum_empirical_p < 1.0:
            raise ValueError("selection-null maximum empirical p must be in (0, 1)")
