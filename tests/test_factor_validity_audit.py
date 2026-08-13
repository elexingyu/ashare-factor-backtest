from __future__ import annotations

import json
from pathlib import Path

from ashare_factor_backtest.application.audit_factor_validity import (
    FactorValidityAuditService,
)
from ashare_factor_backtest.cli.public_main import main


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "examples" / "demo_daily" / "job.yaml"
EXPRESSION = "neg(ts_std(ts_pct_change(close,1),10))"


def test_factor_validity_uses_pit_universe_and_is_return_blind(tmp_path: Path) -> None:
    result, warnings = FactorValidityAuditService().audit(
        JOB,
        EXPRESSION,
        work_root=tmp_path,
    )

    assert result["status"] == "factor_validity_verified"
    assert result["return_data_read"] is False
    assert result["coverage"]["denominator"].startswith(
        "finite cells inside the job's PIT universe"
    )
    assert result["coverage"]["universe_cells"] == 863
    assert result["coverage"]["factor_coverage"] == 1.0
    assert result["checks"]["causality"]["passed"] is True
    assert result["checks"]["target_separation"]["passed"] is True
    assert Path(result["artifact_path"]).is_file()
    assert warnings


def test_factor_validity_identity_does_not_depend_on_work_root(tmp_path: Path) -> None:
    first, _ = FactorValidityAuditService().audit(
        JOB,
        EXPRESSION,
        work_root=tmp_path / "first",
    )
    second, _ = FactorValidityAuditService().audit(
        JOB,
        EXPRESSION,
        work_root=tmp_path / "second",
    )

    assert first["audit_identity"] == second["audit_identity"]


def test_public_cli_exposes_factor_validity_audit(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "audit-factor",
            "--job",
            str(JOB),
            EXPRESSION,
            "--work-root",
            str(tmp_path),
            "--json",
            "--run-id",
            "factor-validity-test",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["command"] == "audit-factor"
    assert payload["status"] == "ok"
    assert payload["data"]["return_data_read"] is False
    assert payload["data"]["coverage"]["universe_cells"] == 863
