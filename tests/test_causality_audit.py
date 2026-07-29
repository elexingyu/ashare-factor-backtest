from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

from ashare_factor_backtest.application.audit_causality import CausalityAuditService
from ashare_factor_backtest.cli.public_main import main
from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.causality import audit_prefix_invariance
from ashare_factor_backtest.expression.evaluator import EvaluationContext
from ashare_factor_backtest.expression.model import OperatorSpec, ValueType
from ashare_factor_backtest.expression.operators.registry import (
    build_production_operator_catalog,
)
from ashare_factor_backtest.expression.production_fields import (
    build_production_field_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


def _context() -> EvaluationContext:
    dates = pd.bdate_range("2024-01-02", periods=12)
    columns = pd.Index(["A", "B"])
    base = np.arange(1.0, 25.0).reshape(12, 2)
    fields = {
        "open": pd.DataFrame(base, index=dates, columns=columns),
        "high": pd.DataFrame(base + 1.0, index=dates, columns=columns),
        "low": pd.DataFrame(base - 1.0, index=dates, columns=columns),
        "close": pd.DataFrame(base + 0.5, index=dates, columns=columns),
        "volume": pd.DataFrame(base * 100.0, index=dates, columns=columns),
        "amount": pd.DataFrame(base * 1000.0, index=dates, columns=columns),
    }
    return EvaluationContext(
        fields=fields,
        dataset_versions={"fixture": "v1"},
        universe_policy="all",
        date_range=(dates[0].date().isoformat(), dates[-1].date().isoformat()),
        universe_size=pd.Series(2, index=dates),
        universe_mask=pd.DataFrame(True, index=dates, columns=columns),
        evaluation_price_basis=PriceBasis.HFQ_PIT,
    )


def test_prefix_invariance_accepts_backward_only_expression() -> None:
    operators, functions = build_production_operator_catalog()

    report = audit_prefix_invariance(
        "ts_mean(close,3)",
        _context(),
        operators=operators,
        fields=build_production_field_catalog(),
        functions=functions,
        cache_max_bytes=8 * 1024 * 1024,
        cutoff="2024-01-09",
    )

    assert report.passed is True
    assert report.mismatch_cells == 0
    assert report.compared_cells == 12
    assert report.max_abs_error == 0.0


def test_prefix_invariance_rejects_operator_that_reads_the_next_row() -> None:
    operators, functions = build_production_operator_catalog()
    operators.register(
        OperatorSpec(
            name="hidden_lead",
            aliases=(),
            version="1",
            category="time_series",
            input_types=(ValueType.PANEL_FLOAT,),
            output_type=ValueType.PANEL_FLOAT,
            causal_contract="same_date_or_backward_only",
            nan_policy="preserve",
            parameter_domain=(),
            lookback_rule="max_children",
            commutative=False,
            complexity_cost=1,
            examples=("hidden_lead(close)",),
        )
    )
    functions = dict(functions)
    functions["hidden_lead"] = lambda panel: panel.shift(-1)

    report = audit_prefix_invariance(
        "hidden_lead(close)",
        _context(),
        operators=operators,
        fields=build_production_field_catalog(),
        functions=functions,
        cache_max_bytes=8 * 1024 * 1024,
        cutoff="2024-01-09",
    )

    assert report.passed is False
    assert report.mismatch_cells == 2
    assert report.first_mismatch_date == "2024-01-09"


def test_public_causality_audit_writes_a_small_certificate(tmp_path) -> None:
    result, warnings = CausalityAuditService().audit(
        ROOT / "examples" / "demo_daily" / "job.yaml",
        "ts_pct_change(close,5)",
        work_root=tmp_path,
    )

    assert result["passed"] is True
    assert result["status"] == "prefix_invariance_verified"
    assert result["mismatch_cells"] == 0
    assert result["compared_cells"] > 0
    assert result["timings_seconds"]["total"] > 0
    certificate = Path(result["certificate_path"])
    assert certificate.is_file()
    assert certificate.stat().st_size < 10_000
    assert warnings


def test_public_cli_exposes_causality_audit(tmp_path, capsys) -> None:
    code = main(
        [
            "audit-causality",
            "--job",
            str(ROOT / "examples" / "demo_daily" / "job.yaml"),
            "ts_pct_change(close,5)",
            "--work-root",
            str(tmp_path),
            "--json",
            "--run-id",
            "causality-test",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["command"] == "audit-causality"
    assert payload["data"]["passed"] is True
    assert payload["data"]["status"] == "prefix_invariance_verified"
