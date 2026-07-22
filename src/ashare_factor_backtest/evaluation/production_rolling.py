"""Production-native rolling selection over the frozen execution context."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ashare_factor_backtest.evaluation.production_evaluator import ProductionLongOnlyResult
from ashare_factor_backtest.evaluation.production_execution_context import ProductionExecutionContext
from ashare_factor_backtest.evaluation.production_screen import (
    ProductionScreenPolicy,
    screen_production_stress_values,
    screen_production_values,
)


@dataclass(frozen=True)
class ProductionRollingWindow:
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        train_start = date.fromisoformat(self.train_start)
        train_end = date.fromisoformat(self.train_end)
        test_start = date.fromisoformat(self.test_start)
        test_end = date.fromisoformat(self.test_end)
        if not self.label.strip():
            raise ValueError("rolling window label must be nonempty")
        if train_start > train_end or test_start > test_end:
            raise ValueError("rolling window ranges are invalid")
        if train_end >= test_start:
            raise ValueError("rolling train must end strictly before test starts")


DEFAULT_PRODUCTION_WINDOWS = (
    ProductionRollingWindow(
        "2018_2020_to_2021", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"
    ),
    ProductionRollingWindow(
        "2019_2021_to_2022", "2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"
    ),
    ProductionRollingWindow(
        "2020_2022_to_2023", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"
    ),
    ProductionRollingWindow(
        "2021_2023_to_2024", "2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"
    ),
    ProductionRollingWindow(
        "2022_2024_to_2025", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"
    ),
)


@dataclass(frozen=True)
class ProductionRollingPolicy:
    windows: tuple[ProductionRollingWindow, ...] = DEFAULT_PRODUCTION_WINDOWS
    horizons: tuple[int, ...] = (5, 20, 60)
    minimum_coverage: float = 0.70
    minimum_periods: int = 400
    top_fraction: float = 0.20
    real_buy_cost: float = 0.0003
    real_sell_cost: float = 0.0012
    stress_buy_cost: float = 0.0005
    stress_sell_cost: float = 0.0020

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("production rolling requires at least one window")
        if len({window.label for window in self.windows}) != len(self.windows):
            raise ValueError("production rolling window labels must be unique")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("production rolling horizons must be positive")
        if not 0.0 < self.minimum_coverage <= 1.0 or self.minimum_periods <= 0:
            raise ValueError("production rolling coverage and periods are invalid")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("production rolling top_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ProductionRollingGate:
    required_folds: int = 5
    minimum_positive_folds: int = 3
    minimum_median_test_excess_sharpe: float = 0.0
    minimum_direction_mode_count: int = 3
    minimum_horizon_mode_count: int = 3


def audit_production_rolling(
    report: dict[str, Any],
    gate: ProductionRollingGate,
) -> dict[str, Any]:
    if report.get("policy", {}).get("execution_semantics") != "production_execution_context":
        return {
            "status": "invalid_evidence",
            "reason_codes": ["nonproduction_execution_semantics"],
        }
    folds = list(report.get("folds", []))
    if len(folds) != gate.required_folds:
        return {
            "status": "invalid_evidence",
            "reason_codes": [f"rolling_fold_count_not_{gate.required_folds}"],
        }
    try:
        trains = np.asarray(
            [fold["train"]["stress"]["excess_metrics"]["sharpe"] for fold in folds],
            dtype=float,
        )
        tests = np.asarray(
            [fold["test"]["stress"]["excess_metrics"]["sharpe"] for fold in folds],
            dtype=float,
        )
        directions = Counter(str(fold["direction"]) for fold in folds)
        horizons = Counter(int(fold["horizon"]) for fold in folds)
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid_evidence", "reason_codes": ["malformed_rolling_evidence"]}
    if not np.isfinite(trains).all() or not np.isfinite(tests).all():
        return {"status": "invalid_evidence", "reason_codes": ["nonfinite_rolling_evidence"]}
    positive = int((tests > 0).sum())
    median_test = float(np.median(tests))
    direction_mode = max(directions.values())
    horizon_mode = max(horizons.values())
    reasons = []
    if positive < gate.minimum_positive_folds:
        reasons.append(f"positive_folds_below_{gate.minimum_positive_folds}")
    if median_test <= gate.minimum_median_test_excess_sharpe:
        reasons.append("median_test_excess_sharpe_nonpositive")
    if direction_mode < gate.minimum_direction_mode_count:
        reasons.append(f"direction_mode_count_below_{gate.minimum_direction_mode_count}")
    if horizon_mode < gate.minimum_horizon_mode_count:
        reasons.append(f"horizon_mode_count_below_{gate.minimum_horizon_mode_count}")
    status = "rejected_rolling" if reasons else "rolling_survivor"
    research_profile = _build_rolling_research_profile(
        folds,
        test_excess_sharpes=tests,
        standalone_supported=not reasons,
    )
    return {
        "status": status,
        "reason_codes": reasons or ["production_rolling_gate_passed"],
        "positive_folds": positive,
        "median_test_excess_sharpe": median_test,
        "median_train_excess_sharpe": float(np.median(trains)),
        "direction_mode_count": direction_mode,
        "horizon_mode_count": horizon_mode,
        "decision_scope": "standalone_unconditional_support_only",
        "factor_record_action": "retain",
        "standalone_support": "unsupported" if reasons else "supported",
        "research_profile": research_profile,
    }


def _build_rolling_research_profile(
    folds: list[dict[str, Any]],
    *,
    test_excess_sharpes: np.ndarray,
    standalone_supported: bool,
) -> dict[str, Any]:
    labels = [str(fold.get("label", f"fold_{index}")) for index, fold in enumerate(folds)]
    directions = [str(fold["direction"]) for fold in folds]
    horizons = [int(fold["horizon"]) for fold in folds]
    sharpes = [float(value) for value in test_excess_sharpes]
    positive_sharpe_count = sum(value > 0.0 for value in sharpes)
    episode_count, maximum_streak = _positive_episode_summary(sharpes)
    recent = sharpes[-min(3, len(sharpes)) :]

    total_returns = _optional_test_excess_total_returns(folds)
    if total_returns is None:
        compounded_return = None
        positive_return_count = None
        gain_loss_ratio = None
        positive_concentration = None
    else:
        compounded_return = float(np.prod(np.asarray(total_returns) + 1.0) - 1.0)
        positives = [value for value in total_returns if value > 0.0]
        negatives = [value for value in total_returns if value < 0.0]
        positive_return_count = len(positives)
        gain_loss_ratio = (
            float(sum(positives) / abs(sum(negatives))) if negatives else None
        )
        positive_concentration = (
            float(max(positives) / sum(positives)) if positives else None
        )

    routes = [
        "factor_record",
        "standalone_followup" if standalone_supported else "standalone_gate_fail",
        "positive_oos_observed" if positive_sharpe_count else "no_positive_oos_observed",
        (
            "recent_positive_oos_observed"
            if any(value > 0.0 for value in recent)
            else "recent_positive_oos_absent"
        ),
    ]
    direction_counts = Counter(directions)
    horizon_counts = Counter(horizons)
    return {
        "schema": "production_rolling_research_profile.v1",
        "research_routes": routes,
        "temporal_profile": {
            "fold_labels": labels,
            "test_excess_sharpes": sharpes,
            "test_excess_total_returns": total_returns,
            "positive_sharpe_folds": positive_sharpe_count,
            "positive_sharpe_fold_share": float(positive_sharpe_count / len(sharpes)),
            "positive_sharpe_episode_count": episode_count,
            "maximum_positive_sharpe_streak": maximum_streak,
            "median_test_excess_sharpe": float(np.median(test_excess_sharpes)),
            "mean_test_excess_sharpe": float(np.mean(test_excess_sharpes)),
            "best_test_excess_sharpe": float(np.max(test_excess_sharpes)),
            "worst_test_excess_sharpe": float(np.min(test_excess_sharpes)),
            "recent_three_fold_count": len(recent),
            "recent_three_positive_sharpe_folds": sum(value > 0.0 for value in recent),
            "recent_three_median_test_excess_sharpe": float(np.median(recent)),
            "return_metrics_available": total_returns is not None,
            "positive_return_folds": positive_return_count,
            "compounded_oos_excess_total_return": compounded_return,
            "positive_to_negative_fold_return_ratio": gain_loss_ratio,
            "largest_positive_fold_return_share": positive_concentration,
            "direction_sequence": directions,
            "direction_unique_count": len(direction_counts),
            "direction_mode_count": max(direction_counts.values()),
            "horizon_sequence": horizons,
            "horizon_unique_count": len(horizon_counts),
            "horizon_mode_count": max(horizon_counts.values()),
            "pooled_daily_sharpe": None,
            "pooled_daily_sharpe_reason": "daily_return_path_not_loaded",
        },
    }


def _optional_test_excess_total_returns(
    folds: list[dict[str, Any]],
) -> list[float] | None:
    values: list[float] = []
    for fold in folds:
        try:
            value = float(fold["test"]["stress"]["excess_metrics"]["total_return"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        values.append(value)
    return values


def _positive_episode_summary(values: list[float]) -> tuple[int, int]:
    episodes = 0
    current = 0
    maximum = 0
    for value in values:
        if value > 0.0:
            current += 1
            maximum = max(maximum, current)
            if current == 1:
                episodes += 1
        else:
            current = 0
    return episodes, maximum


def evaluate_production_rolling(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionRollingPolicy,
    benchmark_cache: dict[tuple[Any, ...], Any] | None = None,
    result_observer: Callable[
        [ProductionRollingWindow, str, str, str, int, ProductionLongOnlyResult], object
    ]
    | None = None,
) -> dict[str, Any]:
    return _evaluate_production_rolling(
        values,
        execution_context,
        policy=policy,
        fold_evaluator=screen_production_values,
        cost_evidence="real_and_stress",
        benchmark_cache=benchmark_cache,
        result_observer=result_observer,
    )


def evaluate_production_rolling_selection_null(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionRollingPolicy,
) -> dict[str, Any]:
    """Replay rolling selection with the exact stress-only S4 resource contract."""
    return _evaluate_production_rolling(
        values,
        execution_context,
        policy=policy,
        fold_evaluator=screen_production_stress_values,
        cost_evidence="stress_only",
        benchmark_cache=None,
        result_observer=None,
    )


def _evaluate_production_rolling(
    values: pd.DataFrame,
    execution_context: ProductionExecutionContext,
    *,
    policy: ProductionRollingPolicy,
    fold_evaluator: Any,
    cost_evidence: str,
    benchmark_cache: dict[tuple[Any, ...], Any] | None,
    result_observer: Callable[
        [ProductionRollingWindow, str, str, str, int, ProductionLongOnlyResult], object
    ]
    | None,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    benchmark_cache = benchmark_cache if benchmark_cache is not None else {}
    for window in policy.windows:
        fold_policy = ProductionScreenPolicy(
            discovery=(window.train_start, window.train_end),
            validation=(window.test_start, window.test_end),
            horizons=policy.horizons,
            minimum_coverage=policy.minimum_coverage,
            minimum_periods=policy.minimum_periods,
            top_fraction=policy.top_fraction,
            real_buy_cost=policy.real_buy_cost,
            real_sell_cost=policy.real_sell_cost,
            stress_buy_cost=policy.stress_buy_cost,
            stress_sell_cost=policy.stress_sell_cost,
        )
        observer = None
        if result_observer is not None:
            def observe(
                segment: str,
                cost: str,
                direction: str,
                horizon: int,
                result: ProductionLongOnlyResult,
                *,
                _window: ProductionRollingWindow = window,
            ) -> object:
                return result_observer(
                    _window, segment, cost, direction, horizon, result
                )

            observer = observe
        evaluated = fold_evaluator(
            values,
            execution_context,
            policy=fold_policy,
            benchmark_cache=benchmark_cache,
            result_observer=observer,
        )
        folds.append(
            {
                "label": window.label,
                "direction": evaluated["selected_direction"],
                "horizon": evaluated["selected_horizon"],
                "train_window": [window.train_start, window.train_end],
                "test_window": [window.test_start, window.test_end],
                "train_variants": evaluated["discovery_variants"],
                "train": evaluated["discovery"],
                "test": evaluated["validation"],
            }
        )
    stress_test_sharpes = np.asarray(
        [fold["test"]["stress"]["excess_metrics"]["sharpe"] for fold in folds],
        dtype=float,
    )
    directions = Counter(str(fold["direction"]) for fold in folds)
    horizons = Counter(int(fold["horizon"]) for fold in folds)
    return {
        "policy": {
            "execution_semantics": "production_execution_context",
            "cost_evidence": cost_evidence,
            "windows": [
                {
                    "label": window.label,
                    "train": [window.train_start, window.train_end],
                    "test": [window.test_start, window.test_end],
                }
                for window in policy.windows
            ],
            "horizons": list(policy.horizons),
            "minimum_coverage": policy.minimum_coverage,
            "minimum_periods": policy.minimum_periods,
            "top_fraction": policy.top_fraction,
            "real_buy_cost": policy.real_buy_cost,
            "real_sell_cost": policy.real_sell_cost,
            "stress_buy_cost": policy.stress_buy_cost,
            "stress_sell_cost": policy.stress_sell_cost,
        },
        "summary": {
            "positive_folds": int((stress_test_sharpes > 0).sum()),
            "median_test_excess_sharpe": float(np.median(stress_test_sharpes)),
            "direction_mode_count": max(directions.values()),
            "horizon_mode_count": max(horizons.values()),
        },
        "folds": folds,
    }
