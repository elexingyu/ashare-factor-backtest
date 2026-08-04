from __future__ import annotations

import json
from pathlib import Path
import shutil
import tomllib

from ashare_factor_backtest.cli.public_main import main


ROOT = Path(__file__).resolve().parents[1]


def test_inspect_job_exposes_stable_engine_and_data_contract(capsys) -> None:
    code = main(
        [
            "inspect-job",
            "--job",
            str(ROOT / "examples" / "demo_daily" / "job.yaml"),
            "--json",
            "--run-id",
            "inspect-job-test",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    assert payload["data"]["engine_version"] == package_version
    assert payload["data"]["machine_protocol"] == "ashare-backtest.protocol.v1"
    assert len(payload["data"]["job_identity"]) == 64
    assert payload["data"]["dataset_version"] == "public-synthetic-cc0-v1"
    assert payload["data"]["universe"]["view"] == "signal_eligible"


def test_inspect_job_identity_does_not_depend_on_install_path(
    tmp_path: Path, capsys
) -> None:
    source = ROOT / "examples" / "demo_daily"
    first = tmp_path / "first"
    second = tmp_path / "second"
    shutil.copytree(source, first)
    shutil.copytree(source, second)

    identities = []
    for index, job in enumerate((first / "job.yaml", second / "job.yaml")):
        code = main(
            [
                "inspect-job",
                "--job",
                str(job),
                "--json",
                "--run-id",
                f"inspect-relocated-{index}",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        identities.append(
            (payload["data"]["contract_identity"], payload["data"]["job_identity"])
        )

    assert identities[0] == identities[1]
