"""Shared contract for the A-Share Factor Backtest Engine versus Qlib benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from typing import Sequence

import numpy as np
import pandas as pd


CROSS_FRAMEWORK_SCHEMA = "ashare-factor-cross-framework.v1"
EXPRESSION_PAIRS = (
    {"ours": "close", "qlib": "$close", "valid_from": 0},
    {"ours": "ts_delay(close,5)", "qlib": "Ref($close,5)", "valid_from": 5},
    {
        "ours": "ts_pct_change(close,5)",
        "qlib": "$close/Ref($close,5)-1",
        "valid_from": 5,
    },
    {"ours": "ts_mean(close,20)", "qlib": "Mean($close,20)", "valid_from": 19},
)


def generate_fixture(
    output_dir: Path,
    *,
    date_count: int = 1_500,
    security_count: int = 500,
    seed: int = 20260722,
) -> dict[str, object]:
    """Generate one deterministic float32 panel and Qlib-compatible CSV files."""
    if date_count < 20 or security_count <= 0:
        raise ValueError("benchmark fixture requires at least 20 dates and one security")
    target = Path(output_dir)
    csv_dir = target / "qlib_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2018-01-02", periods=date_count)
    symbols = tuple(f"SH{600000 + position:06d}" for position in range(security_count))
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0002, 0.018, size=(date_count, security_count))
    close = np.asarray(20.0 * np.exp(np.cumsum(innovations, axis=0)), dtype=np.float32)

    frames: list[pd.DataFrame] = []
    for position, symbol in enumerate(symbols):
        frame = pd.DataFrame(
            {"date": dates, "symbol": symbol, "close": close[:, position]}
        )
        frame.to_csv(csv_dir / f"{symbol}.csv", index=False)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel_path = target / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    identity = _panel_identity(dates, symbols, close)
    manifest = {
        "schema_version": CROSS_FRAMEWORK_SCHEMA,
        "dataset_identity": identity,
        "seed": seed,
        "date_count": date_count,
        "security_count": security_count,
        "start_date": dates[0].date().isoformat(),
        "end_date": dates[-1].date().isoformat(),
        "panel_path": str(panel_path.resolve()),
        "qlib_csv_path": str(csv_dir.resolve()),
        "expressions": list(EXPRESSION_PAIRS),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def normalized_outputs(
    outputs: Sequence[np.ndarray],
    pairs: Sequence[dict[str, object]] = EXPRESSION_PAIRS,
) -> tuple[np.ndarray, ...]:
    if len(outputs) != len(pairs):
        raise ValueError("output count does not match expression contract")
    normalized: list[np.ndarray] = []
    for values, pair in zip(outputs, pairs, strict=True):
        array = np.asarray(values, dtype=np.float64).copy()
        array[: int(pair["valid_from"]), :] = np.nan
        normalized.append(array)
    return tuple(normalized)


def output_digest(outputs: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in outputs:
        array = np.asarray(values, dtype="<f8")
        finite = np.isfinite(array)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(finite.tobytes(order="C"))
        digest.update(np.where(finite, array, 0.0).tobytes(order="C"))
    return digest.hexdigest()


def write_outputs(path: Path, outputs: Sequence[np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"factor_{i}": value for i, value in enumerate(outputs)})


def read_outputs(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path) as archive:
        return tuple(archive[key] for key in sorted(archive.files))


def benchmark_result(
    *,
    manifest: dict[str, object],
    engine_name: str,
    engine_version: str,
    engine_commit: str,
    wall_seconds: Sequence[float],
    cpu_seconds: Sequence[float],
    outputs: Sequence[np.ndarray],
    cache_state: str,
) -> dict[str, object]:
    expressions = manifest["expressions"]
    expression_bytes = json.dumps(expressions, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "ashare-factor-benchmark.v1",
        "benchmark_id": str(manifest["dataset_identity"])[:16],
        "engine": {
            "name": engine_name,
            "version": engine_version,
            "commit": engine_commit,
        },
        "environment": _environment(),
        "workload": {
            "dataset_identity": manifest["dataset_identity"],
            "date_count": manifest["date_count"],
            "security_count": manifest["security_count"],
            "expression_count": len(expressions),
            "expressions_sha256": hashlib.sha256(expression_bytes).hexdigest(),
            "expression_contract": expressions,
            "semantics": "native_persistent_store_to_factor_matrix_complete_windows",
            "output_contract": ["factor_values"],
        },
        "cache_state": cache_state,
        "measurements": {
            "repetitions": len(wall_seconds),
            "wall_seconds": list(wall_seconds),
            "cpu_seconds": list(cpu_seconds),
            "peak_rss_mib": _peak_rss_mib(),
            "output_digest": output_digest(outputs),
        },
        "parity": {
            "reference_engine": "pending-cross-framework-comparison",
            "comparable": False,
            "exact": False,
            "maximum_absolute_error": 0.0,
            "reason": "parity has not been evaluated",
        },
    }


def _panel_identity(
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    close: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(dates.view("i8"), dtype="<i8").tobytes())
    digest.update("\n".join(symbols).encode("ascii"))
    digest.update(np.asarray(close, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.machine(),
        "logical_cpu_count": os.cpu_count() or 1,
        "memory_gib": _physical_memory_bytes() / (1024**3),
    }


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
