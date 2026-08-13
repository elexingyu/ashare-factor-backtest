from __future__ import annotations

import json
from pathlib import Path

import yaml

from ashare_factor_backtest.application.production_job import load_production_job
from ashare_factor_backtest.cli.public_main import main


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_batch_cli_preserves_candidate_order(tmp_path: Path, capsys) -> None:
    expressions = [
        "ts_pct_change(close,5)",
        "cs_rank(ts_pct_change(close,5))",
    ]
    request = tmp_path / "expressions.json"
    request.write_text(
        json.dumps(
            {
                "schema": "ashare-factor-expression-batch.v1",
                "expressions": expressions,
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "evaluate-batch",
            "--job",
            str(ROOT / "examples/demo_daily/job.yaml"),
            "--expressions-file",
            str(request),
            "--through",
            "screen",
            "--work-root",
            str(tmp_path / "work"),
            "--json",
            "--run-id",
            "v024-batch-test",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["command"] == "evaluate-batch.screen"
    assert payload["data"]["candidate_count"] == 2
    assert [row["expression"] for row in payload["data"]["candidates"]] == expressions
    assert payload["data"]["completed_count"] == 2


def test_daily_factor_job_separates_account_and_decay_horizons(
    tmp_path: Path,
) -> None:
    source = ROOT / "examples/demo_daily/job.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    screen = payload["research"]["screen"]
    screen.pop("horizons")
    screen.update(
        {
            "mode": "daily_factor",
            "fixed_direction": "high",
            "decay_horizons": [5, 10, 20, 60],
        }
    )
    job_path = tmp_path / "daily-factor-job.yaml"
    job_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    job = load_production_job(job_path)

    assert job.research is not None
    assert job.research.screen.mode == "daily_factor"
    assert job.research.screen.horizons == (1,)
    assert job.research.screen.fixed_direction == "high"
    assert job.research.screen.decay_horizons == (5, 10, 20, 60)
