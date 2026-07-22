from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_execution_context import (
    ChunkedProductionExecutionAccumulator,
    ExecutionCapturingFrameLoader,
    build_chunked_production_execution_context,
)
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


def _chunk() -> YearChunk:
    return YearChunk(
        year=2024,
        calculation_start=pd.Timestamp("2024-01-02"),
        calculation_end=pd.Timestamp("2024-01-03"),
        load_start=pd.Timestamp("2024-01-02"),
        load_end=pd.Timestamp("2024-01-03"),
        max_lookback=0,
        forward_tail=0,
    )


def _execution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A", "B", "A", "B"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "hfq_open": [10.0, 20.0, 11.0, 19.0],
            "signal_eligible": [True, True, True, False],
            "is_suspended": [False, False, False, False],
            "hfq_up_limit": [11.0, 22.0, 12.1, 20.9],
            "hfq_down_limit": [9.0, 18.0, 9.9, 17.1],
        }
    )


def test_execution_accumulator_requires_complete_unique_capture() -> None:
    chunk = _chunk()
    accumulator = ChunkedProductionExecutionAccumulator(
        (chunk,),
        eligibility_column="signal_eligible",
        price_storage_dtype="float32",
    )

    with pytest.raises(ValueError, match="not complete"):
        accumulator.build()

    list(accumulator.capture_frames(chunk, (_execution_frame(),)))
    context = accumulator.build()

    assert context.valuation_open.dtype == np.float32
    np.testing.assert_array_equal(context.codes, pd.Index(["A", "B"]))
    np.testing.assert_array_equal(
        context.valuation_open,
        np.asarray([[10.0, 20.0], [11.0, 19.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        context.signal_eligible,
        np.asarray([[True, True], [True, False]]),
    )
    with pytest.raises(ValueError, match="already captured"):
        list(accumulator.capture_frames(chunk, (_execution_frame(),)))


def test_execution_accumulator_does_not_publish_partial_capture() -> None:
    accumulator = ChunkedProductionExecutionAccumulator((_chunk(),))
    stream = accumulator.capture_frames(_chunk(), (_execution_frame(),))

    next(stream)
    stream.close()

    with pytest.raises(ValueError, match="not complete"):
        accumulator.build()


def test_captured_execution_context_matches_legacy_builder(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.full_backtest import generate_full_fixture

    manifest = generate_full_fixture(
        tmp_path / "fixture",
        date_count=240,
        security_count=12,
        seed=11,
    )
    expression = str(manifest["expression"])
    prepared = ProductionJobService().prepare(
        Path(manifest["production_job_path"]),
        expression=expression,
    )
    capturing_loader = ExecutionCapturingFrameLoader(
        prepared.frame_loader,
        prepared.chunks,
        eligibility_column=prepared.job.view,
        price_storage_dtype="float32",
    )

    evaluate_expression_by_year(
        expression,
        chunks=prepared.chunks,
        frame_loader=capturing_loader,
        dataset_version="shared_context_test",
        view=prepared.job.view,
        cache_max_bytes=32 * 1024 * 1024,
        required_fields={"close"},
    )
    captured = capturing_loader.execution_context()
    legacy = build_chunked_production_execution_context(
        chunks=prepared.chunks,
        frame_loader=prepared.frame_loader,
        price_storage_dtype="float32",
        eligibility_column=prepared.job.view,
    )

    np.testing.assert_array_equal(captured.dates, legacy.dates)
    np.testing.assert_array_equal(captured.codes, legacy.codes)
    np.testing.assert_array_equal(captured.valuation_open, legacy.valuation_open)
    np.testing.assert_array_equal(captured.buyable, legacy.buyable)
    np.testing.assert_array_equal(captured.sellable, legacy.sellable)
    np.testing.assert_array_equal(captured.signal_eligible, legacy.signal_eligible)
    assert capturing_loader.additional_field_specs == {}
    assert capturing_loader.additional_dataset_versions == {}
