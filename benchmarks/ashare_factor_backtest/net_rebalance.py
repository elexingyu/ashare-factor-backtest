"""Pure target-delta planning shared by the cross-engine benchmark adapter."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class TargetDeltaPlan:
    sell_values: dict[str, float]
    buy_values: dict[str, float]
    retained: tuple[str, ...]
    sell_fee: float
    buy_fee: float
    post_trade_cash: float


def qlib_open_cost(buy_cost: float) -> float:
    """Convert fee / total cash spent into Qlib's fee / security value."""
    if not 0.0 <= buy_cost < 1.0:
        raise ValueError("buy_cost must be in [0, 1)")
    return buy_cost / (1.0 - buy_cost)


def plan_target_delta_values(
    *,
    current_values: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    buy_cost: float,
    sell_cost: float,
) -> TargetDeltaPlan:
    """Plan executable security-value deltas using the public evaluator's fee units."""
    if not 0.0 <= sell_cost < 1.0:
        raise ValueError("sell_cost must be in [0, 1)")
    buy_rate = qlib_open_cost(buy_cost)
    current = _validated_non_negative(current_values, "current_values")
    weights = _validated_non_negative(target_weights, "target_weights")
    if sum(weights.values()) > 1.0 + 1e-12:
        raise ValueError("target weights must sum to at most one")

    if not isfinite(cash):
        raise ValueError("cash must be finite")
    scale_value = abs(cash) + sum(current.values())
    tolerance = max(1e-12, scale_value * 1e-12)
    if -tolerance <= cash < 0.0:
        cash = 0.0
    if cash < 0.0:
        raise ValueError("cash must be non-negative beyond numerical tolerance")
    wealth = cash + sum(current.values())
    desired = {symbol: wealth * weight for symbol, weight in weights.items()}
    symbols = sorted(set(current) | set(desired))
    sell_values = {
        symbol: current.get(symbol, 0.0) - desired.get(symbol, 0.0)
        for symbol in symbols
        if current.get(symbol, 0.0) - desired.get(symbol, 0.0) > tolerance
    }
    sell_notional = sum(sell_values.values())
    sell_fee = sell_notional * sell_cost
    available_cash = cash + sell_notional - sell_fee

    after_sales = {
        symbol: current.get(symbol, 0.0) - sell_values.get(symbol, 0.0)
        for symbol in symbols
    }
    deficits = {
        symbol: desired.get(symbol, 0.0) - after_sales.get(symbol, 0.0)
        for symbol in symbols
        if desired.get(symbol, 0.0) - after_sales.get(symbol, 0.0) > tolerance
    }
    required_cash = sum(value / (1.0 - buy_cost) for value in deficits.values())
    scale = min(1.0, available_cash / required_cash) if required_cash > 0.0 else 0.0
    buy_values = {
        symbol: value * scale
        for symbol, value in deficits.items()
        if value * scale > tolerance
    }
    bought_value = sum(buy_values.values())
    buy_fee = bought_value * buy_rate
    post_trade_cash = available_cash - bought_value - buy_fee
    if -tolerance < post_trade_cash < 0.0:
        post_trade_cash = 0.0
    if post_trade_cash < 0.0:
        raise RuntimeError("target-delta plan produced negative cash")
    retained = tuple(
        symbol
        for symbol in symbols
        if current.get(symbol, 0.0) > tolerance and desired.get(symbol, 0.0) > tolerance
    )
    return TargetDeltaPlan(
        sell_values=sell_values,
        buy_values=buy_values,
        retained=retained,
        sell_fee=sell_fee,
        buy_fee=buy_fee,
        post_trade_cash=post_trade_cash,
    )


def _validated_non_negative(
    values: Mapping[str, float], name: str
) -> dict[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    if any(not isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result
