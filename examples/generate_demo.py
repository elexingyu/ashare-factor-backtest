"""Generate the deterministic, fully synthetic public demo dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "demo_daily"
SYMBOLS = (
    "000001.SZ",
    "000002.SZ",
    "000003.SZ",
    "000004.SZ",
    "600001.SH",
    "600002.SH",
    "600003.SH",
    "600004.SH",
)


def main() -> None:
    dates = pd.bdate_range("2020-01-02", "2021-06-30")
    bars = _bars(dates)
    states = _states(bars)
    _write_year_parts(bars, states)
    _write_sidecars(dates)
    _write_job()


def _bars(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    positions = np.arange(len(dates), dtype=float)
    for index, symbol in enumerate(SYMBOLS):
        direction = -1.0 if index % 2 else 1.0
        trend = direction * (index + 1) * positions / 2200.0
        cycle = 0.025 * np.sin(positions / (5.0 + index))
        open_price = (8.0 + index) * (1.0 + trend + cycle)
        close_price = open_price * (
            1.0 + direction * 0.002 + 0.003 * np.cos(positions / 7.0)
        )
        adj_factor = np.where(
            (index == 6) & (dates >= pd.Timestamp("2021-04-15")),
            2.0,
            1.0,
        )
        for position, day in enumerate(dates):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": day,
                    "open": float(open_price[position]),
                    "high": float(max(open_price[position], close_price[position]) * 1.01),
                    "low": float(min(open_price[position], close_price[position]) * 0.99),
                    "close": float(close_price[position]),
                    "vol": float(100_000 + 1_000 * index + 25 * position),
                    "amount": float(
                        (100_000 + 1_000 * index + 25 * position)
                        * close_price[position]
                    ),
                    "adj_factor": float(adj_factor[position]),
                }
            )
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"])


def _states(bars: pd.DataFrame) -> pd.DataFrame:
    states = bars.loc[:, ["ts_code", "trade_date", "open"]].copy()
    states["is_suspended"] = False
    states["up_limit"] = states["open"] * 1.10
    states["down_limit"] = states["open"] * 0.90
    suspension = states["ts_code"].eq("000002.SZ") & states["trade_date"].eq(
        pd.Timestamp("2021-02-15")
    )
    up_limit = states["ts_code"].eq("000003.SZ") & states["trade_date"].eq(
        pd.Timestamp("2021-04-01")
    )
    down_limit = states["ts_code"].eq("600002.SH") & states["trade_date"].eq(
        pd.Timestamp("2021-05-17")
    )
    states.loc[suspension, "is_suspended"] = True
    states.loc[up_limit, "up_limit"] = states.loc[up_limit, "open"]
    states.loc[down_limit, "down_limit"] = states.loc[down_limit, "open"]
    states["open_at_up_limit"] = states["open"].ge(states["up_limit"])
    states["open_at_down_limit"] = states["open"].le(states["down_limit"])
    return states


def _write_year_parts(bars: pd.DataFrame, states: pd.DataFrame) -> None:
    hashes: list[str] = []
    for year in sorted(bars["trade_date"].dt.year.unique()):
        bar_part = bars.loc[bars["trade_date"].dt.year.eq(year)].copy()
        state_part = states.loc[states["trade_date"].dt.year.eq(year)].copy()
        bar_path = OUTPUT / "bars" / f"year={year}" / "part.parquet"
        state_path = OUTPUT / "states" / f"year={year}" / "part.parquet"
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        bar_part.to_parquet(bar_path, index=False)
        state_part.to_parquet(state_path, index=False)
        hashes.extend((_sha256(bar_path), _sha256(state_path)))
    manifest = {
        "schema": "astock_production_yearly_bars_v1",
        "content_hash": hashlib.sha256("".join(hashes).encode("ascii")).hexdigest(),
        "analysis_range": ["2020-01-02", "2021-06-30"],
        "fixture": "fully_synthetic_cc0_v1",
    }
    (OUTPUT / "bars" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_sidecars(dates: pd.DatetimeIndex) -> None:
    master = pd.DataFrame(
        {
            "ts_code": SYMBOLS,
            "exchange": ["SZSE"] * 4 + ["SSE"] * 4,
            "market": ["主板"] * len(SYMBOLS),
            "list_date": ["20100101"] * 7 + ["20210115"],
            "delist_date": [None] * len(SYMBOLS),
        }
    )
    master.to_parquet(OUTPUT / "stock_master.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "600004.SH",
                "start_date": "20210301",
                "end_date": "20210310",
            }
        ]
    ).to_parquet(OUTPUT / "st_intervals.parquet", index=False)
    pd.DataFrame(
        [
            {
                "exchange": exchange,
                "cal_date": day.strftime("%Y%m%d"),
                "is_open": 1,
            }
            for exchange in ("SSE", "SZSE")
            for day in dates
        ]
    ).to_csv(OUTPUT / "trade_calendar.csv", index=False)


def _write_job() -> None:
    payload = {
        "schema_version": "production-job.v1",
        "job_id": "public_synthetic_demo_v1",
        "dataset_version": "public-synthetic-cc0-v1",
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
            "start": "2021-01-04",
            "end": "2021-06-30",
            "max_lookback": 20,
            "symbol_batch_size": 4,
            "cache_mib": 32,
            "memory_limit_mib": 1024,
            "symbol_cap": 8,
        },
        "research": {
            "evidence_mode": "engineering",
            "screen": {
                "discovery": ["2021-01-04", "2021-03-31"],
                "validation": ["2021-04-01", "2021-06-30"],
                "horizons": [1, 5, 20],
                "minimum_coverage": 0.70,
                "minimum_periods": 20,
                "top_fraction": 0.25,
                "real_buy_cost": 0.0003,
                "real_sell_cost": 0.0012,
                "stress_buy_cost": 0.0005,
                "stress_sell_cost": 0.0020,
            },
            "rolling": {
                "windows": [
                    {
                        "label": "early_2021",
                        "train": ["2021-01-04", "2021-02-26"],
                        "test": ["2021-03-01", "2021-04-30"],
                    },
                    {
                        "label": "late_2021",
                        "train": ["2021-03-01", "2021-04-30"],
                        "test": ["2021-05-03", "2021-06-30"],
                    },
                ],
                "gate": {
                    "required_folds": 2,
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
                "seed_namespace": "public-synthetic-demo-v1",
            },
        },
    }
    (OUTPUT / "job.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
