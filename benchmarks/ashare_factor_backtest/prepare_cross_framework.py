"""Generate deterministic inputs for the cross-framework benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.ashare_factor_backtest.cross_framework import generate_fixture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dates", type=int, default=1_500)
    parser.add_argument("--securities", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    result = generate_fixture(
        arguments.output_dir,
        date_count=arguments.dates,
        security_count=arguments.securities,
        seed=arguments.seed,
    )
    print(json.dumps(result, sort_keys=True))
