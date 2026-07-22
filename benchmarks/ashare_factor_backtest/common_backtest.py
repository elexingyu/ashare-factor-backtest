"""Shared output contract for the fixed-policy complete backtest benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np
import pandas as pd


def target_selection_digest(values: pd.DataFrame, *, top_fraction: float) -> str:
    digest = hashlib.sha256()
    for date, row in values.iterrows():
        valid = row.dropna().sort_index()
        count = max(1, int(np.ceil(len(valid) * top_fraction))) if len(valid) else 0
        selected = valid.sort_values(ascending=False, kind="stable").iloc[:count].index
        digest.update(pd.Timestamp(date).date().isoformat().encode("ascii"))
        digest.update(b"\0")
        digest.update("\n".join(sorted(str(item) for item in selected)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def array_digest(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value, dtype="<f8")
        finite = np.isfinite(array)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(finite.tobytes(order="C"))
        digest.update(np.where(finite, array, 0.0).tobytes(order="C"))
    return digest.hexdigest()


def write_arrays(path: Path, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_json(path: Path, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def environment() -> dict[str, object]:
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
