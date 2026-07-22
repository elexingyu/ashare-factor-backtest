"""Render the README benchmark section from validated JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from benchmarks.ashare_factor_backtest.contract import validate_result


START = "<!-- benchmark:start -->"
END = "<!-- benchmark:end -->"


def render(
    ours_path: Path,
    qlib_path: Path,
    summary_path: Path,
    *,
    language: str = "en",
) -> str:
    if language not in {"en", "zh"}:
        raise ValueError(f"unsupported report language: {language}")
    ours = json.loads(Path(ours_path).read_text(encoding="utf-8"))
    qlib = json.loads(Path(qlib_path).read_text(encoding="utf-8"))
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    validate_result(ours)
    validate_result(qlib)
    if not summary["comparable"] or summary["qlib_over_ours_wall_ratio"] is None:
        raise ValueError("cannot render a speed table without numerical parity")
    if ours["workload"] != qlib["workload"]:
        raise ValueError("benchmark workloads differ")
    dirty = "+dirty" in str(ours["engine"]["commit"])
    workload = ours["workload"]
    if language == "zh":
        status = "预发布（源码目录含未提交修改）" if dirty else "可复现发布版本"
        lines = [
            START,
            f"**证据状态：** {status}。",
            "",
            "| 引擎 | 墙钟时间中位数 | 峰值内存 |",
            "| --- | ---: | ---: |",
            _engine_row(ours),
            _engine_row(qlib),
            "",
            f"在该工作负载下，Qlib 与本引擎的墙钟时间比为 "
            f"`{summary['qlib_over_ours_wall_ratio']:.2f}x`。工作负载包含 "
            f"{workload['security_count']} 只证券、{workload['date_count']} 个交易日和 "
            f"{workload['expression_count']} 条语义对齐的表达式。输出最大绝对误差为 "
            f"`{summary['maximum_absolute_error']:.3g}`，所有有限值位置完全一致。",
            "",
            "该结果是在同一 Python 环境中，以单进程、缓存预热方式，从各自原生存储读取数据并生成因子矩阵。"
            "它不是完整回测速度对比，也不代表其他公式、数据规模或机器仍有相同比例。引用前请阅读 "
            "`BENCHMARKS.md` 和已归档的 JSON 结果。",
            END,
        ]
        return "\n".join(lines)

    status = "Pre-release (dirty source tree)" if dirty else "Release reproducible"
    lines = [
        START,
        f"**Evidence status:** {status}.",
        "",
        "| Engine | Median wall time | Peak RSS |",
        "| --- | ---: | ---: |",
        _engine_row(ours),
        _engine_row(qlib),
        "",
        f"On this workload, Qlib / this engine wall-time ratio was "
        f"`{summary['qlib_over_ours_wall_ratio']:.2f}x`. The workload contains "
        f"{workload['security_count']} securities, {workload['date_count']} dates and "
        f"{workload['expression_count']} mapped expressions. Maximum absolute output "
        f"difference was `{summary['maximum_absolute_error']:.3g}` and all finite masks "
        "matched.",
        "",
        "This is a warm-cache, single-worker, native-store-to-factor-matrix benchmark "
        "on the same Python environment. It is not a complete backtest comparison and "
        "does not imply the same ratio for other formulas, data sizes or machines. "
        "See `BENCHMARKS.md` and the versioned JSON artifacts before quoting it.",
        END,
    ]
    return "\n".join(lines)


def update_readme(readme_path: Path, section: str) -> None:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END)
    if start < 0 or end < start:
        raise ValueError("README benchmark markers are missing")
    end += len(END)
    path.write_text(text[:start] + section + text[end:], encoding="utf-8")


def _engine_row(payload: dict[str, object]) -> str:
    measurements = payload["measurements"]
    return (
        f"| {payload['engine']['name']} | "
        f"{median(measurements['wall_seconds']):.3f} s | "
        f"{measurements['peak_rss_mib']:.0f} MiB |"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--qlib", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=("zh", "en"), default="en")
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    markdown = render(
        arguments.ours,
        arguments.qlib,
        arguments.summary,
        language=arguments.language,
    )
    if arguments.readme:
        update_readme(arguments.readme, markdown)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(markdown + "\n", encoding="utf-8")
    if not arguments.readme and not arguments.output:
        print(markdown)
