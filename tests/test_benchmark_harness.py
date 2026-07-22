from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _result(name: str) -> dict[str, object]:
    return {
        "schema_version": "ashare-factor-benchmark.v1",
        "benchmark_id": "unit",
        "engine": {"name": name, "version": "1", "commit": "unit"},
        "environment": {
            "python": "3.13",
            "platform": "test",
            "processor": "test",
            "logical_cpu_count": 1,
            "memory_gib": 1.0,
        },
        "workload": {
            "dataset_identity": "a" * 64,
            "date_count": 20,
            "security_count": 2,
            "expression_count": 1,
            "expressions_sha256": "b" * 64,
            "semantics": "test",
            "output_contract": ["factor_values"],
        },
        "cache_state": "warm",
        "measurements": {
            "repetitions": 1,
            "wall_seconds": [1.0],
            "cpu_seconds": [1.0],
            "peak_rss_mib": 1.0,
            "output_digest": "c" * 64,
        },
        "parity": {
            "reference_engine": "pending",
            "comparable": False,
            "exact": False,
            "maximum_absolute_error": 0.0,
            "reason": "pending",
        },
    }


def test_benchmark_fixture_and_comparison_are_self_contained(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.compare_cross_framework import compare
    from benchmarks.ashare_factor_backtest.cross_framework import (
        generate_fixture,
        write_outputs,
    )

    manifest = generate_fixture(tmp_path / "data", date_count=20, security_count=2)
    assert pd.read_parquet(manifest["panel_path"])["close"].dtype == np.float32

    ours = tmp_path / "ours"
    qlib = tmp_path / "qlib"
    ours.mkdir()
    qlib.mkdir()
    values = (np.arange(40, dtype=float).reshape(20, 2),)
    write_outputs(ours / "outputs.npz", values)
    write_outputs(qlib / "outputs.npz", values)
    for directory, name in ((ours, "ours"), (qlib, "qlib")):
        (directory / "result.json").write_text(
            json.dumps(_result(name)), encoding="utf-8"
        )

    summary = compare(ours, qlib, tmp_path / "summary.json")

    assert summary["comparable"] is True
    assert summary["qlib_over_ours_wall_ratio"] == 1.0


def test_benchmark_report_renders_chinese_and_english(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.render_cross_framework_report import render

    ours = _result("ours")
    qlib = _result("qlib")
    ours_path = tmp_path / "ours.json"
    qlib_path = tmp_path / "qlib.json"
    summary_path = tmp_path / "summary.json"
    ours_path.write_text(json.dumps(ours), encoding="utf-8")
    qlib_path.write_text(json.dumps(qlib), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "comparable": True,
                "qlib_over_ours_wall_ratio": 1.0,
                "maximum_absolute_error": 0.0,
            }
        ),
        encoding="utf-8",
    )

    chinese = render(ours_path, qlib_path, summary_path, language="zh")
    english = render(ours_path, qlib_path, summary_path, language="en")

    assert "墙钟时间中位数" in chinese
    assert "工作负载包含 2 只证券、20 个交易日" in chinese
    assert "Median wall time" in english
    assert "2 securities, 20 dates" in english
