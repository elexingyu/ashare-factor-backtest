"""Stable JSON response envelope."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = "factor-factory.protocol.v1"


@dataclass(frozen=True)
class MachineEnvelope:
    command: str
    status: Literal["ok", "error"]
    run_id: str
    data: dict[str, Any]
    warnings: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    schema_version: str = PROTOCOL_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "command": self.command,
            "status": self.status,
            "run_id": self.run_id,
            "data": self.data,
            "warnings": list(self.warnings),
            "next_actions": list(self.next_actions),
        }

    def json_line(self) -> str:
        return json.dumps(
            self.payload(),
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
