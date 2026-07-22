"""Sync public sources in the monorepo or audit a standalone checkout."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL = MONOREPO_ROOT / "src" / "astock_research"
IS_MONOREPO = CANONICAL.is_dir()
TARGET = (
    MONOREPO_ROOT / "packages" / "ashare-factor-backtest" / "src" / "ashare_factor_backtest"
    if IS_MONOREPO
    else PACKAGE_ROOT / "src" / "ashare_factor_backtest"
)
BENCHMARK_CANONICAL = MONOREPO_ROOT / "benchmarks" / "ashare_factor_backtest"
BENCHMARK_TARGET = (
    MONOREPO_ROOT
    / "packages"
    / "ashare-factor-backtest"
    / "benchmarks"
    / "ashare_factor_backtest"
    if IS_MONOREPO
    else PACKAGE_ROOT / "benchmarks" / "ashare_factor_backtest"
)
ENTRYPOINTS = ("cli/public_main.py",)
BENCHMARK_MODULES = (
    "__init__.py",
    "contract.py",
    "cross_framework.py",
    "prepare_cross_framework.py",
    "run_cross_framework_ours.py",
    "run_qlib_baseline.py",
    "compare_cross_framework.py",
    "render_cross_framework_report.py",
)
PRIVATE_NAMESPACE = "src.astock_research"
PUBLIC_NAMESPACE = "ashare_factor_backtest"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not IS_MONOREPO:
        _validate_generated(TARGET)
        _validate_generated(BENCHMARK_TARGET)
        print("standalone source audit passed; this checkout is canonical")
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix="ashare-factor-backtest-sync-") as temp:
            expected = Path(temp) / "ashare_factor_backtest"
            _materialize(expected)
            differences = _differences(expected, TARGET)
            expected_benchmarks = Path(temp) / "benchmarks" / "ashare_factor_backtest"
            _materialize_benchmarks(expected_benchmarks)
            differences.extend(_differences(expected_benchmarks, BENCHMARK_TARGET))
        if differences:
            print("public package source is not synchronized:")
            for difference in differences:
                print(f"- {difference}")
            return 1
        return 0
    _materialize(TARGET)
    _materialize_benchmarks(BENCHMARK_TARGET)
    print(
        f"synchronized {len(_discover_modules())} public modules and "
        f"{len(BENCHMARK_MODULES)} benchmark modules"
    )
    return 0


def _discover_modules() -> tuple[str, ...]:
    queue = list(ENTRYPOINTS)
    seen: set[str] = set()
    while queue:
        relative = queue.pop(0)
        if relative in seen:
            continue
        path = CANONICAL / relative
        if not path.is_file():
            raise ValueError(f"public source module does not exist: {relative}")
        seen.add(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith(f"{PRIVATE_NAMESPACE}."):
                continue
            module_path = node.module.removeprefix(f"{PRIVATE_NAMESPACE}.").replace(
                ".", "/"
            )
            module_file = f"{module_path}.py"
            package_init = f"{module_path}/__init__.py"
            if (CANONICAL / module_file).is_file():
                queue.append(module_file)
            elif (CANONICAL / package_init).is_file():
                queue.append(package_init)
            else:
                raise ValueError(
                    f"cannot resolve public internal import {node.module} in {relative}"
                )
            package_dir = CANONICAL / module_path
            if package_dir.is_dir():
                for alias in node.names:
                    child = package_dir / f"{alias.name}.py"
                    if child.is_file():
                        queue.append(str(child.relative_to(CANONICAL)))
    return tuple(sorted(seen))


def _materialize(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    modules = _discover_modules()
    package_dirs = {Path(".")}
    for relative in modules:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        package_dirs.update(destination.relative_to(target).parents)
        source = (CANONICAL / relative).read_text(encoding="utf-8")
        rewritten = source.replace(PRIVATE_NAMESPACE, PUBLIC_NAMESPACE)
        if PRIVATE_NAMESPACE in rewritten:
            raise ValueError(f"private namespace remains after rewrite: {relative}")
        destination.write_text(rewritten, encoding="utf-8")
    for package_dir in sorted(package_dirs, key=lambda item: (len(item.parts), str(item))):
        init = target / package_dir / "__init__.py"
        if not init.exists():
            init.write_text('"""Public A-share factor evaluation package."""\n', encoding="utf-8")
    _validate_generated(target)


def _validate_generated(target: Path) -> None:
    forbidden = (
        PRIVATE_NAMESPACE,
        f"{PUBLIC_NAMESPACE}.search",
        "from spec.",
        "/Volumes/",
        "/Users/",
    )
    violations: list[str] = []
    for path in sorted(target.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                violations.append(f"{path.relative_to(target)}:{value}")
    if violations:
        raise ValueError("forbidden public source references: " + ", ".join(violations))


def _materialize_benchmarks(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in BENCHMARK_MODULES:
        source_path = BENCHMARK_CANONICAL / relative
        if not source_path.is_file():
            raise ValueError(f"benchmark module does not exist: {relative}")
        source = source_path.read_text(encoding="utf-8")
        rewritten = source.replace(PRIVATE_NAMESPACE, PUBLIC_NAMESPACE)
        (target / relative).write_text(rewritten, encoding="utf-8")
    _validate_generated(target)


def _differences(expected: Path, actual: Path) -> list[str]:
    expected_files = {
        path.relative_to(expected): path.read_bytes()
        for path in expected.rglob("*.py")
    }
    actual_files = (
        {
            path.relative_to(actual): path.read_bytes()
            for path in actual.rglob("*.py")
        }
        if actual.is_dir()
        else {}
    )
    differences = [
        f"missing {path}" for path in sorted(expected_files.keys() - actual_files.keys())
    ]
    differences.extend(
        f"unexpected {path}" for path in sorted(actual_files.keys() - expected_files.keys())
    )
    differences.extend(
        f"changed {path}"
        for path in sorted(expected_files.keys() & actual_files.keys())
        if expected_files[path] != actual_files[path]
    )
    return differences


if __name__ == "__main__":
    raise SystemExit(main())
