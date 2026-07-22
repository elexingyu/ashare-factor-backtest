from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_factor_backtest.evaluation.production_evaluator import (
    _rebalance_sleeve,
    evaluate_production_staggered_long_only,
)


def _frame(dates: pd.DatetimeIndex, codes: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": date,
                "hfq_open": 10.0,
                "signal_eligible": True,
                "is_suspended": False,
                "hfq_up_limit": 11.0,
                "hfq_down_limit": 9.0,
            }
            for code in codes
            for date in dates
        ]
    )


def test_randomized_rebalances_conserve_wealth_less_fees() -> None:
    rng = np.random.default_rng(20260722)

    for _ in range(1_000):
        size = int(rng.integers(1, 40))
        holdings = rng.uniform(0.0, 2.0, size=size)
        holdings[rng.random(size) < 0.65] = 0.0
        cash = float(rng.uniform(0.0, 2.0))
        target_count = int(rng.integers(0, size + 1))
        targets = np.sort(rng.choice(size, size=target_count, replace=False))
        buyable = rng.random(size) > 0.2
        sellable = rng.random(size) > 0.2
        buy_cost = float(rng.uniform(0.0, 0.003))
        sell_cost = float(rng.uniform(0.0, 0.006))
        before_wealth = float(holdings.sum() + cash)

        outcome = _rebalance_sleeve(
            holdings,
            cash,
            targets,
            buyable=buyable,
            sellable=sellable,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
        )

        after_wealth = float(holdings.sum() + outcome.cash)
        assert outcome.cash >= -1e-12
        assert float(holdings.min()) >= -1e-12
        assert after_wealth == pytest.approx(
            before_wealth - outcome.buy_fee - outcome.sell_fee,
            abs=1e-11,
        )


def test_repeated_target_is_retained_without_round_trip() -> None:
    dates = pd.bdate_range("2021-01-04", periods=6)
    codes = ("000001.SZ", "000002.SZ")
    factor = pd.DataFrame([[2.0, 1.0]] * len(dates), index=dates, columns=codes)

    result = evaluate_production_staggered_long_only(
        factor,
        _frame(dates, codes),
        direction="high",
        horizon=1,
        buy_cost=0.001,
        sell_cost=0.002,
        decision_start=str(dates[0].date()),
        decision_end=str(dates[-1].date()),
        top_fraction=0.5,
    )

    assert result.strategy.order_events[1].status == "retained"
    assert result.strategy.order_events[1].bought == ()
    assert result.strategy.order_events[1].sold == ()
    assert result.strategy.average_turnover < 0.3


def test_blocked_stock_becomes_residual_without_cancelling_other_orders() -> None:
    dates = pd.bdate_range("2021-01-04", periods=6)
    codes = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")
    frame = _frame(dates, codes)
    factor = pd.DataFrame(
        [[4.0, 3.0, 2.0, 1.0]] + [[2.0, 1.0, 4.0, 3.0]] * 5,
        index=dates,
        columns=codes,
    )
    blocked = frame["ts_code"].eq("000001.SZ") & frame["trade_date"].eq(dates[2])
    frame.loc[blocked, ["hfq_open", "hfq_up_limit", "hfq_down_limit"]] = np.nan
    frame.loc[blocked, "is_suspended"] = True

    result = evaluate_production_staggered_long_only(
        factor,
        frame,
        direction="high",
        horizon=1,
        buy_cost=0.0,
        sell_cost=0.0,
        decision_start=str(dates[0].date()),
        decision_end=str(dates[-1].date()),
        top_fraction=0.5,
    )
    partial = result.strategy.order_events[1]

    assert partial.status == "partial_fill"
    assert partial.blocked_sells == ("000001.SZ",)
    assert partial.sold == ("000002.SZ",)
    assert partial.bought == ("000003.SZ", "000004.SZ")
    assert partial.residual == ("000001.SZ",)
