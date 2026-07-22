from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


def _common_result(name: str, wall_seconds: float) -> dict[str, object]:
    result = _result(name)
    result["workload"] = {
        "benchmark_id": "common-full-backtest-v1",
        "semantics": "fixed",
    }
    result["measurements"] = {"wall_seconds": [wall_seconds]}
    result["evidence"] = {
        "target_selection_digest": "same-selection",
        "strategy_metrics": {"total_return": 0.1, "sharpe": 1.2},
        "rank_ic": {"rank_ic_mean": 0.03},
        "execution_adapter": "target_delta_continuous_value_v2",
        "turnover_coordinate": "traded_security_value_over_previous_portfolio_value",
    }
    return result


def test_complete_backtest_comparison_requires_parity_before_speed_claim(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.compare_common_backtest import compare

    ours = tmp_path / "ours"
    qlib = tmp_path / "qlib"
    ours.mkdir()
    qlib.mkdir()
    arrays = {
        "return_dates_ns": np.arange(3, dtype=np.int64),
        "factor": np.arange(6, dtype=float).reshape(3, 2),
        "gross_returns": np.array([0.0, 0.01, -0.02]),
        "net_returns": np.array([0.0, 0.009, -0.021]),
        "turnover_rates": np.array([0.0, 0.4, 0.2]),
        "cost_rates": np.array([0.0, 0.001, 0.001]),
    }
    np.savez_compressed(ours / "outputs.npz", **arrays)
    np.savez_compressed(qlib / "outputs.npz", **arrays)
    (ours / "result.json").write_text(
        json.dumps(_common_result("ours", 1.0)), encoding="utf-8"
    )
    (qlib / "result.json").write_text(
        json.dumps(_common_result("qlib", 2.0)), encoding="utf-8"
    )

    comparable = compare(ours, qlib, tmp_path / "comparable.json")

    assert comparable["comparable"] is True
    assert comparable["speed_claim_allowed"] is True
    assert comparable["qlib_over_ours_wall_ratio"] == 2.0

    mismatched = arrays | {"net_returns": np.array([0.0, 0.009, -0.03])}
    np.savez_compressed(qlib / "outputs.npz", **mismatched)
    rejected = compare(ours, qlib, tmp_path / "rejected.json")

    assert rejected["comparable"] is False
    assert rejected["speed_claim_allowed"] is False
    assert rejected["qlib_over_ours_wall_ratio"] is None
    assert rejected["first_mismatches"]["net_returns"]["position"] == 2


def test_complete_backtest_comparison_rejects_different_environments(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.compare_common_backtest import compare

    ours = tmp_path / "ours"
    qlib = tmp_path / "qlib"
    ours.mkdir()
    qlib.mkdir()
    arrays = {
        "return_dates_ns": np.arange(2, dtype=np.int64),
        "factor": np.ones((2, 2)),
        "gross_returns": np.zeros(2),
        "net_returns": np.zeros(2),
        "turnover_rates": np.zeros(2),
        "cost_rates": np.zeros(2),
    }
    np.savez_compressed(ours / "outputs.npz", **arrays)
    np.savez_compressed(qlib / "outputs.npz", **arrays)
    ours_result = _common_result("ours", 1.0)
    qlib_result = _common_result("qlib", 2.0)
    qlib_result["environment"] = dict(qlib_result["environment"])
    qlib_result["environment"]["python"] = "3.12"
    (ours / "result.json").write_text(json.dumps(ours_result), encoding="utf-8")
    (qlib / "result.json").write_text(json.dumps(qlib_result), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "environments differ: python"):
        compare(ours, qlib, tmp_path / "summary.json")


def test_target_delta_plan_retains_matching_positions_without_round_trip(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.net_rebalance import plan_target_delta_values

    plan = plan_target_delta_values(
        current_values={"A": 50.0, "B": 50.0},
        cash=0.0,
        target_weights={"A": 0.5, "B": 0.5},
        buy_cost=0.001,
        sell_cost=0.002,
    )

    assert plan.sell_values == {}
    assert plan.buy_values == {}
    assert plan.retained == ("A", "B")
    assert plan.post_trade_cash == 0.0


def test_target_delta_plan_sells_only_excess_and_scales_buys_pro_rata(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.net_rebalance import plan_target_delta_values

    plan = plan_target_delta_values(
        current_values={"A": 60.0, "B": 40.0},
        cash=0.0,
        target_weights={"A": 0.5, "C": 0.5},
        buy_cost=0.001,
        sell_cost=0.002,
    )

    assert plan.sell_values == {"A": 10.0, "B": 40.0}
    assert plan.buy_values["C"] == pytest.approx(49.9 * 0.999)
    assert plan.retained == ("A",)
    assert plan.post_trade_cash == pytest.approx(0.0, abs=1e-12)
    assert plan.sell_fee == pytest.approx(0.1)
    assert plan.buy_fee == pytest.approx(plan.buy_values["C"] * 0.001 / 0.999)


def test_qlib_open_cost_converts_cash_spend_cost_to_traded_value_cost(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.net_rebalance import qlib_open_cost

    assert qlib_open_cost(0.001) == pytest.approx(0.001 / 0.999)


def test_target_delta_plan_treats_large_account_roundoff_as_zero(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.net_rebalance import plan_target_delta_values

    plan = plan_target_delta_values(
        current_values={"A": 60_000_000.0, "B": 40_000_000.0},
        cash=0.0,
        target_weights={"A": 0.5, "C": 0.5},
        buy_cost=0.0003,
        sell_cost=0.0012,
    )

    assert plan.post_trade_cash == 0.0

    carried = plan_target_delta_values(
        current_values={"A": 50_000_000.0, "C": 49_985_010.0},
        cash=-1e-8,
        target_weights={"A": 0.5, "C": 0.5},
        buy_cost=0.0003,
        sell_cost=0.0012,
    )
    assert carried.post_trade_cash >= 0.0


def test_archived_complete_backtest_claim_matches_versioned_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "benchmarks" / "results" / "full_backtest_v1"
    ours = json.loads((evidence / "ours_common.json").read_text(encoding="utf-8"))
    qlib = json.loads((evidence / "qlib_common.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (evidence / "common_summary.json").read_text(encoding="utf-8")
    )
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert ours["environment"] == qlib["environment"]
    assert ours["engine"]["commit"].startswith("7f11920")
    assert qlib["engine"]["commit"].startswith("d5379c5")
    assert summary["comparable"] is True
    assert summary["speed_claim_allowed"] is True
    assert round(summary["ours_median_wall_seconds"], 3) == 0.804
    assert round(summary["qlib_median_wall_seconds"], 3) == 25.539
    assert round(summary["qlib_over_ours_wall_ratio"], 2) == 31.77
    assert "**`31.77x`**" in readme


def test_v02_complete_backtest_claim_matches_versioned_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "benchmarks" / "results" / "full_backtest_v2"
    ours = json.loads((evidence / "ours_common.json").read_text(encoding="utf-8"))
    qlib = json.loads((evidence / "qlib_common.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (evidence / "common_summary.json").read_text(encoding="utf-8")
    )
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert ours["environment"] == qlib["environment"]
    assert ours["engine"]["commit"].startswith("9eda124")
    assert qlib["engine"]["commit"].startswith("d5379c5")
    assert summary["comparable"] is True
    assert summary["speed_claim_allowed"] is True
    assert summary["first_mismatches"] == {}
    assert round(summary["ours_median_wall_seconds"], 3) == 1.330
    assert round(summary["qlib_median_wall_seconds"], 3) == 19.070
    assert round(summary["qlib_over_ours_wall_ratio"], 2) == 14.34
    assert "**`14.34x`**" in readme


def _full_research_result(
    *, wall_seconds: list[float], peak_rss_mib: float
) -> dict[str, object]:
    return {
        "engine": {"name": "engine", "commit": "revision"},
        "environment": {
            "python": "3.12.13",
            "platform": "test",
            "processor": "arm64",
            "logical_cpu_count": 8,
            "memory_gib": 24.0,
        },
        "workload": {"date_count": 1500, "security_count": 500},
        "measurements": {
            "wall_seconds": wall_seconds,
            "peak_rss_mib": peak_rss_mib,
        },
        "evidence": {
            "screen": {"rank_ic": 0.01},
            "rolling": {"folds": [{"return": 0.02}]},
            "gate": {"status": "candidate"},
            "warnings": [],
        },
    }


def test_full_research_comparison_requires_evidence_and_performance_gate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.ashare_factor_backtest.compare_full_research_evidence import (
        compare,
    )

    baseline = _full_research_result(
        wall_seconds=[9.5, 10.0, 9.8], peak_rss_mib=460.0
    )
    candidate = _full_research_result(
        wall_seconds=[6.5, 6.7, 6.6], peak_rss_mib=400.0
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    accepted = compare(
        baseline_path,
        candidate_path,
        tmp_path / "accepted.json",
        maximum_median_seconds=8.3105,
        maximum_peak_rss_mib=650.0,
    )

    assert accepted["semantic_evidence_exact"] is True
    assert accepted["retention_allowed"] is True
    assert accepted["wall_time_improvement_fraction"] > 0.30

    candidate["evidence"]["screen"]["rank_ic"] = 0.02
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    rejected = compare(
        baseline_path,
        candidate_path,
        tmp_path / "rejected.json",
        maximum_median_seconds=8.3105,
        maximum_peak_rss_mib=650.0,
    )

    assert rejected["semantic_evidence_exact"] is False
    assert rejected["retention_allowed"] is False


def test_archived_shared_context_claim_matches_versioned_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "benchmarks" / "results" / "shared_context_v1"
    candidate = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (evidence / "comparison.json").read_text(encoding="utf-8")
    )
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert candidate["engine"]["commit"].startswith("7899b44")
    assert comparison["environment_exact"] is True
    assert comparison["workload_exact"] is True
    assert comparison["semantic_evidence_exact"] is True
    assert comparison["retention_allowed"] is True
    assert round(comparison["candidate"]["median_wall_seconds"], 3) == 7.256
    assert round(comparison["wall_time_improvement_fraction"] * 100, 1) == 25.8
    assert "**`7.256 s`**" in readme
