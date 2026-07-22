from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_public_namespace_does_not_require_private_repository() -> None:
    script = """
import json
import sys
from ashare_factor_backtest.cli.public_main import main
code = main(['compile', 'cs_rank(ts_pct_change(close,5))', '--json', '--run-id', 'wheel'])
print(json.dumps({'code': code, 'private_loaded': any(name.startswith('src.astock_research') for name in sys.modules)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = completed.stdout.splitlines()
    payload = json.loads(lines[0])
    audit = json.loads(lines[1])
    assert payload["status"] == "ok"
    assert audit == {"code": 0, "private_loaded": False}


def test_compile_reports_hfq_cross_section_scale_warning() -> None:
    script = """
import json
from ashare_factor_backtest.cli.public_main import main
code = main(['compile', 'cs_rank(close)', '--json', '--run-id', 'hfq-warning'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["warnings"] == [
        "hfq_price_level_is_not_cross_sectionally_comparable"
    ]
