"""Validation for publishable benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "ashare-factor-benchmark.v1"
CACHE_STATES = frozenset({"cold", "warm", "resume"})


def validate_result(payload: Mapping[str, Any]) -> None:
    _require_equal(payload, "schema_version", SCHEMA_VERSION)
    _require_string(payload, "benchmark_id")
    engine = _require_mapping(payload, "engine")
    for key in ("name", "version", "commit"):
        _require_string(engine, key)

    environment = _require_mapping(payload, "environment")
    for key in ("python", "platform", "processor"):
        _require_string(environment, key)
    _require_positive_number(environment, "logical_cpu_count")
    _require_positive_number(environment, "memory_gib")

    workload = _require_mapping(payload, "workload")
    _require_string(workload, "dataset_identity")
    for key in ("date_count", "security_count", "expression_count"):
        _require_positive_number(workload, key)
    _require_digest(workload, "expressions_sha256")
    _require_string(workload, "semantics")
    output_contract = workload.get("output_contract")
    if not _is_sequence(output_contract) or not output_contract:
        raise ValueError("workload.output_contract must be a non-empty sequence")

    cache_state = payload.get("cache_state")
    if cache_state not in CACHE_STATES:
        raise ValueError(f"cache_state must be one of {sorted(CACHE_STATES)}")

    measurements = _require_mapping(payload, "measurements")
    repetitions = measurements.get("repetitions")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("measurements.repetitions must be a positive integer")
    for key in ("wall_seconds", "cpu_seconds"):
        values = measurements.get(key)
        if not _is_sequence(values) or len(values) != repetitions:
            raise ValueError(f"measurements.{key} must match repetitions")
        if any(not _is_nonnegative_number(value) for value in values):
            raise ValueError(f"measurements.{key} must contain nonnegative numbers")
    _require_positive_number(measurements, "peak_rss_mib")
    _require_digest(measurements, "output_digest")

    parity = _require_mapping(payload, "parity")
    _require_string(parity, "reference_engine")
    if not isinstance(parity.get("comparable"), bool):
        raise ValueError("parity.comparable must be boolean")
    if not isinstance(parity.get("exact"), bool):
        raise ValueError("parity.exact must be boolean")
    error = parity.get("maximum_absolute_error")
    if not _is_nonnegative_number(error):
        raise ValueError("parity.maximum_absolute_error must be nonnegative")
    _require_string(parity, "reason")


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_equal(payload: Mapping[str, Any], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"{key} must equal {expected!r}")


def _require_positive_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not _is_nonnegative_number(value) or float(value) <= 0:
        raise ValueError(f"{key} must be positive")
    return float(value)


def _require_digest(payload: Mapping[str, Any], key: str) -> str:
    value = _require_string(payload, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase sha256 digest")
    return value


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
