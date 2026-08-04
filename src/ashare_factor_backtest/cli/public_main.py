"""Small machine-first CLI for the public single-factor evaluation surface."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys
import uuid
from typing import Sequence

from ashare_factor_backtest.application.audit_causality import CausalityAuditService
from ashare_factor_backtest.application.compile_expression import CompileExpressionService
from ashare_factor_backtest.application.evaluate_factor import FactorEvaluationService
from ashare_factor_backtest.application.production_job import ProductionJobService
from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.protocol.envelope import MachineEnvelope
from ashare_factor_backtest.protocol.errors import (
    EXIT_INPUT_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_OK,
    ProtocolError,
)


PUBLIC_PROTOCOL_VERSION = "ashare-backtest.protocol.v1"
PUBLIC_ENGINE_VERSION = "0.2.2"
PUBLIC_COMMANDS = (
    "capabilities",
    "schema",
    "doctor",
    "compile",
    "inspect-job",
    "audit-causality",
    "evaluate",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as error:
        return int(error.code)

    run_id = args.run_id or uuid.uuid4().hex
    try:
        exit_code = EXIT_OK
        service = CompileExpressionService()
        if args.command == "capabilities":
            envelope = _envelope(
                "capabilities",
                "ok",
                run_id,
                {
                    "commands": list(PUBLIC_COMMANDS),
                    "engine_version": PUBLIC_ENGINE_VERSION,
                    "excluded_capabilities": [
                        "factor_search",
                        "multi_factor_portfolio",
                        "live_trading",
                    ],
                    "machine_protocol": PUBLIC_PROTOCOL_VERSION,
                    "scope": "single_factor_evaluation",
                },
                next_actions=("schema", "doctor"),
            )
        elif args.command == "schema":
            envelope = _envelope(
                "schema",
                "ok",
                run_id,
                service.schema(),
                next_actions=("compile",),
            )
        elif args.command == "doctor":
            envelope = _envelope(
                "doctor",
                "ok",
                run_id,
                {
                    "core": "ready",
                    "field_catalog_version": service.field_catalog_version,
                    "operator_catalog_version": service.operator_catalog_version,
                },
                warnings=("No dataset was requested or validated.",),
                next_actions=("compile",),
            )
        elif args.command == "compile":
            result = service.execute(args.expression)
            warnings = tuple(result.pop("warnings"))
            envelope = _envelope(
                "compile",
                "ok",
                run_id,
                result,
                warnings=warnings,
                next_actions=("evaluate",),
            )
        elif args.command == "inspect-job":
            with redirect_stdout(sys.stderr):
                result, warnings = ProductionJobService().inspect(Path(args.job))
            envelope = _envelope(
                "inspect-job",
                "ok",
                run_id,
                {
                    **result,
                    "engine_version": PUBLIC_ENGINE_VERSION,
                    "machine_protocol": PUBLIC_PROTOCOL_VERSION,
                },
                warnings=warnings,
                next_actions=("compile", "audit-causality", "evaluate"),
            )
        elif args.command == "audit-causality":
            with redirect_stdout(sys.stderr):
                result, warnings = CausalityAuditService().audit(
                    Path(args.job),
                    args.expression,
                    work_root=Path(args.work_root),
                )
            passed = bool(result["passed"])
            envelope = _envelope(
                "audit-causality",
                "ok" if passed else "error",
                run_id,
                result,
                warnings=warnings,
                next_actions=(
                    ("evaluate",)
                    if passed
                    else ("inspect_certificate", "fix_expression_or_operator")
                ),
            )
            if not passed:
                exit_code = EXIT_INTERNAL_ERROR
        else:
            with redirect_stdout(sys.stderr):
                result, warnings = FactorEvaluationService().evaluate(
                    Path(args.job),
                    args.expression,
                    through=args.through,
                    work_root=Path(args.work_root),
                )
            envelope = _envelope(
                f"evaluate.{args.through}",
                "ok",
                run_id,
                _compact_evaluation_receipt(result),
                warnings=warnings,
                next_actions=("inspect_artifact",),
            )
        _emit(envelope)
        return exit_code
    except ExpressionError as error:
        _emit(
            _envelope(
                args.command,
                "error",
                run_id,
                ProtocolError(error.code, str(error)).payload(),
                next_actions=("fix_expression", "schema"),
            )
        )
        return EXIT_INPUT_ERROR
    except (TypeError, ValueError) as error:
        _emit(
            _envelope(
                args.command,
                "error",
                run_id,
                ProtocolError("INVALID_INPUT", str(error)).payload(),
                next_actions=("schema",),
            )
        )
        return EXIT_INPUT_ERROR
    except Exception as error:
        _emit(
            _envelope(
                args.command,
                "error",
                run_id,
                ProtocolError("INTERNAL_ERROR", str(error)).payload(),
                next_actions=("doctor",),
            )
        )
        return EXIT_INTERNAL_ERROR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ashare-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capabilities", "schema", "doctor"):
        child = subparsers.add_parser(command)
        _common_arguments(child)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("expression")
    _common_arguments(compile_parser)
    inspect_parser = subparsers.add_parser("inspect-job")
    inspect_parser.add_argument("--job", required=True)
    _common_arguments(inspect_parser)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--job", required=True)
    evaluate_parser.add_argument("expression")
    evaluate_parser.add_argument(
        "--through", choices=("screen", "rolling"), default="rolling"
    )
    evaluate_parser.add_argument("--work-root", required=True)
    _common_arguments(evaluate_parser)
    audit_parser = subparsers.add_parser("audit-causality")
    audit_parser.add_argument("--job", required=True)
    audit_parser.add_argument("expression")
    audit_parser.add_argument("--work-root", required=True)
    _common_arguments(audit_parser)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--run-id")


def _envelope(
    command: str,
    status: str,
    run_id: str,
    data: dict[str, object],
    *,
    warnings: tuple[str, ...] = (),
    next_actions: tuple[str, ...] = (),
) -> MachineEnvelope:
    return MachineEnvelope(
        command=command,
        status=status,
        run_id=run_id,
        data=data,
        warnings=warnings,
        next_actions=next_actions,
        schema_version=PUBLIC_PROTOCOL_VERSION,
    )


def _emit(envelope: MachineEnvelope) -> None:
    sys.stdout.write(envelope.json_line())


def _compact_evaluation_receipt(data: dict[str, object]) -> dict[str, object]:
    receipt = {key: value for key, value in data.items() if key != "rolling"}
    rolling = data.get("rolling")
    if isinstance(rolling, dict):
        receipt["rolling_summary"] = rolling.get("summary", {})
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
