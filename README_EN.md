# A-Share Factor Backtest Engine

[简体中文](README.md) | English

An A-share single-factor backtesting engine designed for AI agents and automated research workflows. Given an expression and a data job, the CLI performs safe compilation, applies A-share trading constraints, runs rolling evaluation and returns stable machine-readable results.

> This project is research software, not investment advice. Backtests and synthetic examples do not establish future profitability.

## Why This Project Exists

General-purpose backtesting frameworks commonly leave data preparation, historical universes, adjustment policy, price limits, suspensions and reporting conventions to each user. This project makes those error-prone A-share rules reusable contracts. A person or AI agent can submit a WorldQuant-like factor expression without rebuilding the same backtest plumbing for every experiment.

It is not a factor search engine and does not promise to discover profitable factors. Version 0.1 focuses on turning one existing expression into a fast, reproducible and auditable single-factor backtest.

## Quick Start

The project builds as a standalone wheel. From this directory, run:

```bash
uv sync --locked --all-groups
uv run ashare-backtest doctor --json
uv run ashare-backtest compile 'cs_rank(ts_pct_change(close,5))' --json
uv run ashare-backtest evaluate \
  --job examples/demo_daily/job.yaml \
  'cs_rank(ts_pct_change(close,5))' \
  --through rolling \
  --work-root /tmp/ashare-factor-demo \
  --json
```

Each CLI invocation writes exactly one JSON line to standard output, making it suitable for Codex, Claude Code, CI pipelines and other programs. Diagnostics go to standard error and do not corrupt the machine protocol.

The bundled dataset is entirely synthetic. It covers a listing boundary, an ST interval, a suspension, an open-limit-up event and an open-limit-down event. It verifies execution semantics and does not demonstrate alpha.

## Core Capabilities

- **Expression compilation:** allowlisted fields and operators reject unknown fields, invalid parameters and forward-looking references.
- **A-share point-in-time semantics:** factor observation, historical universe membership and next-open execution are kept distinct.
- **Adjustment/execution separation:** adjusted series may drive factor values while executable trading and returns retain the corresponding real-price semantics.
- **Trading constraints:** listing status, ST, suspension, open price limits, two-sided costs and long-only portfolios.
- **Rolling evaluation:** train/test evidence, strategy and benchmark metrics, excess metrics and coverage diagnostics.
- **Reproducible artifacts:** expression, data identity, job contract and evaluation semantics jointly identify reusable results.
- **AI-first interface:** `capabilities`, `schema`, `doctor`, `compile` and `evaluate` expose a versioned JSON protocol.

## Scope

Version 0.1 includes expression evaluation, data contracts, single-factor quantile backtesting, A-share execution constraints and rolling evidence.

The first release deliberately excludes factor generation/search, multi-factor portfolio optimization, live trading, private market data and a general event-driven order system. Those systems can call this engine from above without becoming part of its trusted single-factor evaluation core.

## Performance Evidence

The table below is generated from versioned JSON artifacts. The comparison fixes source values, mapped expression semantics, valid-value masks, Python environment, worker count and output scope. A speed ratio is rendered only after numerical parity passes.

The first comparison measures both engines reading their native persistent stores and producing four factor matrices. It excludes A-share execution simulation, IC, costs and rolling evaluation. It supports a claim about the current expression-computation path, not the same speedup for a complete backtest.

<!-- benchmark:start -->
**Evidence status:** Release reproducible.

| Engine | Median wall time | Peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest-public-evaluator | 0.086 s | 372 MiB |
| microsoft-qlib-local-provider-kernels-1 | 0.903 s | 690 MiB |

On this workload, Qlib / this engine wall-time ratio was `10.50x`. The workload contains 500 securities, 1500 dates and 4 mapped expressions. Maximum absolute output difference was `1.45e-05` and all finite masks matched.

This is a warm-cache, single-worker, native-store-to-factor-matrix benchmark on the same Python environment. It is not a complete backtest comparison and does not imply the same ratio for other formulas, data sizes or machines. See `BENCHMARKS.md` and the versioned JSON artifacts before quoting it.
<!-- benchmark:end -->

See [BENCHMARKS.md](BENCHMARKS.md) for the complete methodology, raw JSON artifacts and reproduction commands.

## License

The software is licensed under Apache-2.0. The synthetic fixture is separately dedicated under CC0 1.0; see [examples/DATA_LICENSE.md](examples/DATA_LICENSE.md).
