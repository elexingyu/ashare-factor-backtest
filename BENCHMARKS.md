# Benchmark Methodology

Performance claims are accepted only when both engines use the same deterministic source values, mapped expression semantics, valid-window masks, Python environment, worker count and output contract. The current comparison measures each engine's native persistent store through production of four factor matrices. It does not measure the A-share execution simulator, IC, costs or rolling evidence.

The mapped expressions are `close`, five-day delay, five-day percentage change and twenty-day mean. Qlib's `Rank` is a time-series operator, while this project's `cs_rank` is cross-sectional, so rank is deliberately excluded.

The committed JSON files under `benchmarks/results/cross_framework_v1/` record all repetitions, hardware, source revisions, workload identity, output hashes and numerical parity. The README section is generated, not handwritten:

```bash
uv run python -m benchmarks.ashare_factor_backtest.render_cross_framework_report \
  --ours benchmarks/results/cross_framework_v1/ours.json \
  --qlib benchmarks/results/cross_framework_v1/qlib.json \
  --summary benchmarks/results/cross_framework_v1/summary.json \
  --language zh \
  --readme README.md

uv run python -m benchmarks.ashare_factor_backtest.render_cross_framework_report \
  --ours benchmarks/results/cross_framework_v1/ours.json \
  --qlib benchmarks/results/cross_framework_v1/qlib.json \
  --summary benchmarks/results/cross_framework_v1/summary.json \
  --language en \
  --readme README_EN.md
```

The deterministic panel generator and both runners are included under `benchmarks/ashare_factor_backtest/`. Qlib is an optional benchmark dependency and is intentionally absent from the runtime wheel. A release benchmark must be rerun from a clean tagged source tree; a result whose revision ends in `+dirty` is visibly labeled pre-release.
