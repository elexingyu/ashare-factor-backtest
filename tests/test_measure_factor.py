from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ashare_factor_backtest.application.measure_factor import FactorMeasurementService
from ashare_factor_backtest.cli.public_main import main
from ashare_factor_backtest.evaluation.daily_factor_measurement import _weighted_return


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "examples" / "demo_daily" / "job.yaml"
EXPRESSION = "neg(ts_std(ts_pct_change(close,1),10))"


def test_daily_measurement_is_non_promoting_and_reusable(tmp_path: Path) -> None:
    service = FactorMeasurementService()
    first, _ = service.measure(
        JOB,
        EXPRESSION,
        direction="high",
        horizons=(1, 2, 3),
        rolling_windows=(20,),
        work_root=tmp_path,
    )
    second, _ = service.measure(
        JOB,
        EXPRESSION,
        direction="high",
        horizons=(1, 2, 3),
        rolling_windows=(20,),
        work_root=tmp_path,
    )

    assert first["return_data_read"] is True
    assert first["promotion_authority"] is False
    assert first["summary"]["return_clock"] == (
        "signal_t_to_open_t_plus_1_to_open_t_plus_2"
    )
    assert first["summary"]["direction"] == "high"
    assert first["measurement_identity"] == second["measurement_identity"]
    assert second["reused"] is True
    assert Path(first["artifacts"]["daily_trace"]["path"]).is_file()
    assert Path(first["artifacts"]["top20_membership"]["path"]).is_file()


def test_public_cli_exposes_daily_measurement(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "measure-factor",
            "--job",
            str(JOB),
            EXPRESSION,
            "--direction",
            "high",
            "--horizons",
            "1,2,3",
            "--rolling-windows",
            "20",
            "--work-root",
            str(tmp_path),
            "--json",
            "--run-id",
            "daily-measurement-test",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["command"] == "measure-factor"
    assert payload["status"] == "ok"
    assert payload["data"]["promotion_authority"] is False
    assert payload["data"]["summary"]["factor_return"]["observations"] > 0


def test_zero_book_and_missing_held_return_are_distinct() -> None:
    assert _weighted_return(
        np.asarray([0.0, 0.0]), np.asarray([np.nan, np.nan])
    ) == 0.0
    assert np.isnan(
        _weighted_return(
            np.asarray([0.5, -0.5]), np.asarray([0.01, np.nan])
        )
    )
