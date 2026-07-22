from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_backtest.evaluation.production_execution_context import (
    ProductionExecutionContext,
)
from ashare_factor_backtest.evaluation.production_rank_ic import (
    evaluate_production_rank_ic,
)


def _context(prices: np.ndarray) -> ProductionExecutionContext:
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    codes = pd.Index(["A", "B", "C"])
    shape = prices.shape
    return ProductionExecutionContext(
        dates=dates,
        codes=codes,
        valuation_open=np.asarray(prices, dtype=float),
        buyable=np.ones(shape, dtype=bool),
        sellable=np.ones(shape, dtype=bool),
        signal_eligible=np.ones(shape, dtype=bool),
    )


def test_rank_ic_uses_next_open_holding_interval_without_lookahead() -> None:
    prices = np.asarray(
        [
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [11.0, 12.0, 13.0],
            [12.1, 14.4, 16.9],
            [13.31, 17.28, 21.97],
        ]
    )
    values = pd.DataFrame(
        [[1.0, 2.0, 3.0]] * len(prices),
        index=pd.bdate_range("2024-01-02", periods=len(prices)),
        columns=["A", "B", "C"],
    )

    result = evaluate_production_rank_ic(
        values,
        _context(prices),
        horizon=1,
        signal_start="2024-01-02",
        signal_end="2024-01-08",
    )

    assert result["semantics"] == "signal_t_to_open_t_plus_1_to_t_plus_1_plus_h"
    assert result["observation_count"] == 3
    assert result["rank_ic_mean"] == pytest.approx(1.0)
    assert result["positive_rate"] == pytest.approx(1.0)
    assert result["average_cross_section_count"] == pytest.approx(3.0)


def test_rank_ic_respects_signal_date_pit_eligibility() -> None:
    prices = np.asarray(
        [
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [11.0, 12.0, 8.0],
            [12.1, 14.4, 6.4],
        ]
    )
    context = _context(prices)
    eligible = context.signal_eligible.copy()
    eligible[:, 2] = False
    context = ProductionExecutionContext(
        dates=context.dates,
        codes=context.codes,
        valuation_open=context.valuation_open,
        buyable=context.buyable,
        sellable=context.sellable,
        signal_eligible=eligible,
    )
    values = pd.DataFrame(
        [[1.0, 2.0, 100.0]] * len(prices),
        index=context.dates,
        columns=context.codes,
    )

    result = evaluate_production_rank_ic(
        values,
        context,
        horizon=1,
        signal_start="2024-01-02",
        signal_end="2024-01-05",
    )

    assert result["rank_ic_mean"] == pytest.approx(1.0)
    assert result["average_cross_section_count"] == pytest.approx(2.0)
