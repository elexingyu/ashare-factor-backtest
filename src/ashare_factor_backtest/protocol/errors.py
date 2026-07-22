"""Stable process exit codes and machine error payloads."""

from __future__ import annotations

from dataclasses import dataclass


EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_INTERNAL_ERROR = 70


@dataclass(frozen=True)
class ProtocolError:
    code: str
    message: str

    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.message}}
