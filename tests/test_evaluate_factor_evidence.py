from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_factor_backtest.application.evaluate_factor import FactorEvaluationService
from ashare_factor_backtest.evaluation.production_frame_loader import (
    BatchedProductionFrameLoader,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_evaluation_reports_ic_and_stage_timings(tmp_path) -> None:
    result, _ = FactorEvaluationService().evaluate(
        ROOT / "examples" / "demo_daily" / "job.yaml",
        "ts_pct_change(close,5)",
        through="screen",
        work_root=tmp_path,
    )

    timings = result["timings_seconds"]
    assert timings["total"] >= sum(
        timings[name]
        for name in ("prepare", "factor", "execution_context", "screen")
    )
    rank_ic = result["screen"]["validation"]["rank_ic"]
    assert rank_ic["semantics"] == "signal_t_to_open_t_plus_1_to_t_plus_1_plus_h"
    assert rank_ic["observation_count"] > 0


def test_rolling_response_keeps_screen_evidence(tmp_path) -> None:
    result, _ = FactorEvaluationService().evaluate(
        ROOT / "examples" / "demo_daily" / "job.yaml",
        "ts_pct_change(close,5)",
        through="rolling",
        work_root=tmp_path,
    )

    assert result["screen"]["validation"]["rank_ic"]["observation_count"] > 0
    assert len(result["rolling"]["folds"]) == 2


def test_public_evaluation_reads_each_chunk_once(tmp_path, monkeypatch) -> None:
    requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original = BatchedProductionFrameLoader.iter_frames

    def counted(self, load_start, load_end):
        requests.append((pd.Timestamp(load_start), pd.Timestamp(load_end)))
        yield from original(self, load_start, load_end)

    monkeypatch.setattr(BatchedProductionFrameLoader, "iter_frames", counted)

    FactorEvaluationService().screen(
        ROOT / "examples" / "demo_daily" / "job.yaml",
        "ts_pct_change(close,5)",
        work_root=tmp_path,
    )

    assert len(requests) == 1
