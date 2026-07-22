from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def test_full_backtest_fixture_contains_both_engine_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.full_backtest import generate_full_fixture

    manifest = generate_full_fixture(
        tmp_path / "fixture",
        date_count=240,
        security_count=12,
        seed=7,
    )

    assert manifest["date_count"] == 240
    assert manifest["security_count"] == 12
    assert Path(manifest["production_job_path"]).is_file()
    assert len(list((tmp_path / "fixture" / "qlib_csv").glob("*.csv"))) == 12
    panel = pd.read_parquet(manifest["panel_path"])
    assert set(("open", "close", "volume", "factor", "change")).issubset(panel)
    job = yaml.safe_load(Path(manifest["production_job_path"]).read_text())
    assert job["research"]["rolling"]["gate"]["required_folds"] == 2
    bars_manifest = json.loads(
        (tmp_path / "fixture" / "production" / "bars" / "manifest.json").read_text()
    )
    assert bars_manifest["schema"] == "astock_production_yearly_bars_v1"
