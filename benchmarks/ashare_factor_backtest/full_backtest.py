"""Deterministic full-backtest fixture shared by production and Qlib runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


FULL_BACKTEST_SCHEMA = "ashare-factor-full-backtest.v1"
DEFAULT_EXPRESSION = "ts_pct_change(close,5)"


def generate_full_fixture(
    output_dir: Path,
    *,
    date_count: int = 1_500,
    security_count: int = 500,
    seed: int = 20260722,
) -> dict[str, Any]:
    if date_count < 220 or security_count < 10:
        raise ValueError("full benchmark requires at least 220 dates and 10 securities")
    target = Path(output_dir).resolve()
    production = target / "production"
    qlib_csv = target / "qlib_csv"
    production.mkdir(parents=True, exist_ok=True)
    qlib_csv.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2018-01-02", periods=date_count)
    symbols = _symbols(security_count)
    qlib_symbols = tuple(_to_qlib_symbol(symbol) for symbol in symbols)
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00015, 0.009, size=(date_count, 1))
    idiosyncratic = rng.normal(0.0, 0.013, size=(date_count, security_count))
    close_returns = market + idiosyncratic
    close = np.asarray(20.0 * np.exp(np.cumsum(close_returns, axis=0)), dtype=np.float32)
    overnight = rng.normal(0.0, 0.003, size=(date_count, security_count))
    open_price = np.asarray(close * np.exp(overnight), dtype=np.float32)
    high = np.maximum(open_price, close) * np.asarray(
        1.0 + rng.uniform(0.0, 0.012, size=close.shape), dtype=np.float32
    )
    low = np.minimum(open_price, close) * np.asarray(
        1.0 - rng.uniform(0.0, 0.012, size=close.shape), dtype=np.float32
    )
    volume = rng.integers(100_000, 8_000_000, size=close.shape).astype(np.float32)
    previous_close = np.vstack((close[:1], close[:-1]))
    change = np.asarray(close / previous_close - 1.0, dtype=np.float32)

    panel = _panel(
        dates,
        symbols,
        qlib_symbols,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        change=change,
    )
    panel_path = target / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    _write_qlib_csv(panel, qlib_csv)
    _write_production_data(
        production,
        panel,
        dates=dates,
        symbols=symbols,
        seed=seed,
    )
    job_path, fold_count = _write_job(
        production,
        dates=dates,
        security_count=security_count,
        seed=seed,
    )
    identity = _identity(dates, symbols, open_price, close)
    manifest = {
        "schema_version": FULL_BACKTEST_SCHEMA,
        "dataset_identity": identity,
        "seed": seed,
        "date_count": date_count,
        "security_count": security_count,
        "start_date": dates[0].date().isoformat(),
        "end_date": dates[-1].date().isoformat(),
        "expression": DEFAULT_EXPRESSION,
        "qlib_expression": "$signal_close/Ref($signal_close,5)-1",
        "panel_path": str(panel_path),
        "qlib_csv_path": str(qlib_csv),
        "production_job_path": str(job_path),
        "symbols": list(qlib_symbols),
        "rolling_fold_count": fold_count,
        "common_contract": {
            "direction": "high",
            "horizon": 1,
            "top_fraction": 0.20,
            "signal_time": "close_t",
            "execution_time": "open_t_plus_1",
            "buy_cost": 0.0003,
            "sell_cost": 0.0012,
            "terminal_liquidation": True,
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _symbols(count: int) -> tuple[str, ...]:
    sz_count = count // 2
    return tuple(
        [f"{position:06d}.SZ" for position in range(1, sz_count + 1)]
        + [f"{600000 + position:06d}.SH" for position in range(count - sz_count)]
    )


def _to_qlib_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    return f"{exchange}{code}"


def _panel(
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    qlib_symbols: tuple[str, ...],
    **matrices: np.ndarray,
) -> pd.DataFrame:
    rows = len(dates) * len(symbols)
    data: dict[str, Any] = {
        "date": np.repeat(dates.to_numpy(), len(symbols)),
        "symbol": np.tile(qlib_symbols, len(dates)),
        "ts_code": np.tile(symbols, len(dates)),
    }
    for name, values in matrices.items():
        data[name] = np.asarray(values, dtype=np.float32).reshape(rows)
    data["factor"] = np.ones(rows, dtype=np.float32)
    return pd.DataFrame(data)


def _write_qlib_csv(panel: pd.DataFrame, target: Path) -> None:
    qlib_panel = panel.copy()
    qlib_panel["signal_close"] = qlib_panel["close"]
    # Qlib's daily account always marks positions with `$close`. The common
    # contract is open-to-open, so keep the causal signal close in a dedicated
    # field and use open as the account valuation field.
    qlib_panel["close"] = qlib_panel["open"]
    columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "signal_close",
        "volume",
        "factor",
        "change",
    ]
    for symbol, frame in qlib_panel.loc[:, columns].groupby("symbol", sort=True):
        frame.to_csv(target / f"{symbol}.csv", index=False)


def _write_production_data(
    target: Path,
    panel: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    seed: int,
) -> None:
    bars = panel.rename(columns={"date": "trade_date", "volume": "vol"}).copy()
    bars["amount"] = bars["vol"] * bars["close"]
    bars["adj_factor"] = 1.0
    bars = bars.loc[
        :, ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]
    ]
    states = bars.loc[:, ["ts_code", "trade_date", "open"]].copy()
    states["is_suspended"] = False
    states["up_limit"] = states["open"] * 1.10
    states["down_limit"] = states["open"] * 0.90
    rng = np.random.default_rng(seed + 1)
    event_pool = np.arange(len(states))
    suspension_indexes = rng.choice(event_pool, size=max(1, len(states) // 5_000), replace=False)
    remaining = np.setdiff1d(event_pool, suspension_indexes)
    up_indexes = rng.choice(remaining, size=max(1, len(states) // 8_000), replace=False)
    remaining = np.setdiff1d(remaining, up_indexes)
    down_indexes = rng.choice(remaining, size=max(1, len(states) // 8_000), replace=False)
    states.loc[suspension_indexes, "is_suspended"] = True
    states.loc[up_indexes, "up_limit"] = states.loc[up_indexes, "open"]
    states.loc[down_indexes, "down_limit"] = states.loc[down_indexes, "open"]
    states["open_at_up_limit"] = states["open"].ge(states["up_limit"])
    states["open_at_down_limit"] = states["open"].le(states["down_limit"])

    hashes: list[str] = []
    for year in sorted(dates.year.unique()):
        bar_part = bars.loc[bars["trade_date"].dt.year.eq(year)]
        state_part = states.loc[states["trade_date"].dt.year.eq(year)]
        bar_path = target / "bars" / f"year={year}" / "part.parquet"
        state_path = target / "states" / f"year={year}" / "part.parquet"
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        bar_part.to_parquet(bar_path, index=False)
        state_part.to_parquet(state_path, index=False)
        hashes.extend((_sha256(bar_path), _sha256(state_path)))
    (target / "bars" / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "astock_production_yearly_bars_v1",
                "content_hash": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
                "analysis_range": [dates[0].date().isoformat(), dates[-1].date().isoformat()],
                "fixture": FULL_BACKTEST_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    later_listing = max(1, len(symbols) // 20)
    list_dates = ["20100101"] * len(symbols)
    for position in range(later_listing):
        list_dates[-position - 1] = dates[len(dates) // 5].strftime("%Y%m%d")
    exchanges = ["SZSE" if symbol.endswith(".SZ") else "SSE" for symbol in symbols]
    pd.DataFrame(
        {
            "ts_code": symbols,
            "exchange": exchanges,
            "market": ["主板"] * len(symbols),
            "list_date": list_dates,
            "delist_date": [None] * len(symbols),
        }
    ).to_parquet(target / "stock_master.parquet", index=False)
    st_rows = [
        {
            "ts_code": symbol,
            "start_date": dates[len(dates) // 3].strftime("%Y%m%d"),
            "end_date": dates[len(dates) // 3 + 10].strftime("%Y%m%d"),
        }
        for symbol in symbols[: max(1, len(symbols) // 100)]
    ]
    pd.DataFrame(st_rows).to_parquet(target / "st_intervals.parquet", index=False)
    pd.DataFrame(
        [
            {"exchange": exchange, "cal_date": day.strftime("%Y%m%d"), "is_open": 1}
            for exchange in ("SSE", "SZSE")
            for day in dates
        ]
    ).to_csv(target / "trade_calendar.csv", index=False)


def _write_job(
    target: Path,
    *,
    dates: pd.DatetimeIndex,
    security_count: int,
    seed: int,
) -> tuple[Path, int]:
    # The production universe requires 120 completed trading days after listing.
    # Keep that causal warmup outside every measured research window.
    evaluation_dates = dates[140:]
    fold_count = 5 if len(evaluation_dates) >= 1_000 else 2
    initial_train = max(60, len(evaluation_dates) // 2)
    test_length = max(20, (len(evaluation_dates) - initial_train) // fold_count)
    windows: list[dict[str, Any]] = []
    for fold in range(fold_count):
        test_start = initial_train + fold * test_length
        test_end = min(len(evaluation_dates) - 1, test_start + test_length - 1)
        train_start = max(0, test_start - initial_train)
        windows.append(
            {
                "label": f"fold_{fold + 1}",
                "train": [
                    evaluation_dates[train_start].date().isoformat(),
                    evaluation_dates[test_start - 1].date().isoformat(),
                ],
                "test": [
                    evaluation_dates[test_start].date().isoformat(),
                    evaluation_dates[test_end].date().isoformat(),
                ],
            }
        )
    split = int(len(evaluation_dates) * 0.60)
    minimum_periods = min(400, max(20, initial_train // 2))
    payload = {
        "schema_version": "production-job.v1",
        "job_id": f"full_backtest_{security_count}x{len(dates)}_{seed}",
        "dataset_version": f"full-backtest-{seed}",
        "data": {
            "bars_dir": "bars",
            "market_states_dir": "states",
            "stock_master": "stock_master.parquet",
            "st_intervals": "st_intervals.parquet",
            "trade_calendar": "trade_calendar.csv",
        },
        "plugins": [],
        "universe": {"view": "signal_eligible", "index_code": None},
        "evaluation": {
            "start": evaluation_dates[0].date().isoformat(),
            "end": evaluation_dates[-1].date().isoformat(),
            "max_lookback": 20,
            "symbol_batch_size": min(100, security_count),
            "cache_mib": 2048,
            "memory_limit_mib": 12288,
            "symbol_cap": None,
        },
        "research": {
            "evidence_mode": "engineering",
            "screen": {
                "discovery": [evaluation_dates[0].date().isoformat(), evaluation_dates[split - 1].date().isoformat()],
                "validation": [evaluation_dates[split].date().isoformat(), evaluation_dates[-1].date().isoformat()],
                "horizons": [1, 5, 20],
                "minimum_coverage": 0.70,
                "minimum_periods": minimum_periods,
                "top_fraction": 0.20,
                "real_buy_cost": 0.0003,
                "real_sell_cost": 0.0012,
                "stress_buy_cost": 0.0005,
                "stress_sell_cost": 0.0020,
            },
            "rolling": {
                "windows": windows,
                "gate": {
                    "required_folds": fold_count,
                    "minimum_positive_folds": 1,
                    "minimum_median_test_excess_sharpe": 0.0,
                    "minimum_direction_mode_count": 1,
                    "minimum_horizon_mode_count": 1,
                },
            },
            "selection_null": {
                "permutations": 20,
                "maximum_portfolio_evaluations": 100000,
                "maximum_empirical_p": 0.10,
                "minimum_changed_column_fraction": 0.50,
                "maximum_mean_coverage_drift": 0.01,
                "maximum_daily_coverage_drift": 0.10,
                "seed_namespace": f"full-backtest-{seed}",
            },
        },
    }
    path = target / "job.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, fold_count


def _identity(
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    open_price: np.ndarray,
    close: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(dates.view("i8"), dtype="<i8").tobytes())
    digest.update("\n".join(symbols).encode("ascii"))
    digest.update(np.asarray(open_price, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(close, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
