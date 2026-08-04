# A-Share Factor Backtest Engine

[简体中文](README.md) | English

[![CI](https://github.com/elexingyu/ashare-factor-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/elexingyu/ashare-factor-backtest/actions/workflows/ci.yml)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

An A-share single-factor backtesting engine designed for AI agents and automated research workflows. Given an expression and a data job, the CLI performs safe compilation, applies A-share trading constraints, runs rolling evaluation and returns stable machine-readable results.

**Performance summary:** on the fixed complete single-factor workload with 500 securities and 1,500 dates, the project measured `1.330 s` versus Qlib's `19.070 s`, or **`14.34x`** faster. The ratio is emitted only after all 11 parity checks pass. See [Performance Evidence](#performance-evidence).

> This project is research software, not investment advice. Backtests and synthetic examples do not establish future profitability.

## Why This Project Exists

General-purpose backtesting frameworks commonly leave data preparation, historical universes, adjustment policy, price limits, suspensions and reporting conventions to each user. This project makes those error-prone A-share rules reusable contracts. A person or AI agent can submit a WorldQuant-like factor expression without rebuilding the same backtest plumbing for every experiment.

It is not a factor search engine and does not promise to discover profitable factors. Version 0.2 focuses on turning one existing expression into a fast, reproducible and auditable single-factor backtest.

## Quick Start

The project builds as a standalone wheel. From this directory, run:

```bash
uv sync --locked --all-groups
uv run ashare-backtest doctor --json
uv run ashare-backtest compile 'cs_rank(ts_pct_change(close,5))' --json
uv run ashare-backtest inspect-job --job examples/demo_daily/job.yaml --json
uv run ashare-backtest audit-causality \
  --job examples/demo_daily/job.yaml \
  'cs_rank(ts_pct_change(close,5))' \
  --work-root /tmp/ashare-factor-demo \
  --json
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
- **Causality audit:** an independent prefix-invariance check recomputes a historical
  prefix. The command fails and saves a machine-readable certificate if adding future
  rows changes past factor values.
- **A-share point-in-time semantics:** factor observation, historical universe membership and next-open execution are kept distinct.
- **Temporal category plugin:** external entity/category/effective-date/expiry-date
  tables can be materialized as PIT fields. Missing, overlapping or multiply assigned
  observations fail closed; vendor-specific industry data is not coupled to the core.
- **Optional within-group operators:** the Python extension surface provides
  `group_demean`, `group_rank` and `group_zscore` for comparisons within each PIT
  category. Categories have a distinct type and cannot be accidentally treated as
  continuous values by an AI-generated expression. These operators are not yet part
  of the default CLI catalog.
- **Dual price coordinates:** factor values and continuous valuation use point-in-time back-adjusted series, while tradability and order prices use raw opens and raw exchange limits.
- **Adjustment-scale warnings:** back-adjusted prices are suitable for returns, ratios and time-series normalization; the compiler warns when raw price levels, averages or spreads can contaminate cross-sectional selection.
- **Trading constraints:** listing status, ST, suspension, open price limits, per-security partial fills, net rebalancing, two-sided costs and long-only portfolios.
- **Rolling evaluation:** train/test evidence, Rank IC, strategy and benchmark metrics, excess metrics and coverage diagnostics.
- **Reproducible artifacts:** expression, data identity, job contract and evaluation semantics jointly identify reusable results.
- **AI-first interface:** `capabilities`, `schema`, `doctor`, `compile`, `inspect-job`,
  `audit-causality` and `evaluate` expose a versioned JSON protocol.

`inspect-job` returns the engine version, protocol, data-asset identities, universe,
and execution-contract identity before evaluation. An external Agent can therefore
freeze a reusable experiment identity without importing internal Python modules.

`audit-causality` is an independent release-time or new-operator check and does not
slow down ordinary `evaluate` calls. It verifies that expression execution does not
depend on future rows; it cannot prove that an upstream provider did not place revised
data into history. The certificate therefore says `prefix_invariance_verified`, not
"all look-ahead bias is impossible."

## Scope

Version 0.2 includes expression evaluation, data contracts, single-factor quantile backtesting, A-share execution constraints and rolling evidence.

The current release deliberately excludes factor generation/search, multi-factor portfolio optimization, live trading, private market data and a general event-driven order system. Those systems can call this engine from above without becoming part of its trusted single-factor evaluation core.

## Execution Contract

A signal is frozen on day `T` and executed at the `T+1` open. Each sleeve trades only the delta between current holdings and the new equal-weight target: overlapping names are retained; executable sell deltas proceed; suspended or limit-down sells remain as residual positions without cancelling other orders; and only realized cash funds buyable target deficits. Reports include planned and actual turnover, blocked buy/sell orders, target-tracking error and terminal residual value.

This is an A-share execution-constraint proxy with raw order coordinates and a dividend-reinvestment total-return valuation proxy, not a broker-level share ledger. It does not yet store actual share counts, so board lots, per-order minimum commissions, corporate-action cash/share flows, queue priority and order-book impact remain outside the current public scope.

## Performance Evidence

Every performance claim uses deterministic synthetic data, a clean source revision and archived JSON evidence. The comparison program emits a speed ratio only when both engines use the same Python environment and pass pre-registered parity checks for factor values, daily selections, returns, Sharpe and Rank IC.

### Complete Single-Factor Backtest vs Qlib

Version 0.2 uses per-security target-delta rebalancing: overlapping holdings are retained and only target differences trade. The Qlib side uses an independent adapter for the same order contract. A speed ratio is emitted only after factor values, selections, daily returns, turnover, costs and Rank IC pass parity.

The common workload contains 500 securities, 1,500 dates and one five-day price-change expression. Both engines start from persistent data, calculate the factor, select a daily cross-section, rebalance at the next open, charge two-sided costs, liquidate at the end and write NAV, return, drawdown, turnover, Rank IC and artifact outputs. Each engine uses one warmup followed by five measured single-process runs.

| Engine | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest `9eda124` | **1.330 s** | 568 MiB |
| Microsoft Qlib `d5379c5` | 19.070 s | 1,038 MiB |

For this fixed-policy complete backtest, the Qlib / project wall-time ratio is **`14.34x`**. All 11 parity checks pass with no first mismatch: maximum daily net-return error is `3.04e-15`, turnover error is `4.98e-12`, total-return error is `2.84e-14`, Sharpe error is `8.38e-14`, and mean Rank IC error is `5.74e-08`.

The ratio applies only to this workload and cannot be generalized to every expression, dataset size or machine. The shared contract excludes board lots, minimum commissions, price limits, suspensions and corporate actions. Whether the synthetic factor makes money is orthogonal to a benchmark of correctness and runtime. The v0.1 full-liquidation evidence and its `31.77x` historical ratio remain versioned, but do not describe v0.2.

### Full A-Share Research Path

The production path additionally applies point-in-time listing status, ST, suspension, open price limits, stressed costs, screening windows and rolling evaluation. On the same 500 × 1,500 fixture, one expression triggers five rolling folds, 90 portfolio simulations and 12 Rank IC evaluations. Sharing the factor and execution data pass reduced median end-to-end time from `9.777 s` to **`7.256 s`**, a **25.8%** improvement, while process peak RSS fell from 459 MiB to **411 MiB**.

Median stage times after the optimization are approximately `4.569 s` for the shared data/factor/execution-panel pass, `0.028 s` for final execution-matrix materialization, `2.045 s` for rolling evaluation and `0.592 s` for screening. The automated gate confirms exact equality of the old and new `workload` and complete `evidence` JSON. No Qlib ratio is reported for this layer because there is no fully aligned Qlib implementation of the same A-share constraints and rolling evidence contract.

### Expression-Computation Microbenchmark

The table below is generated from versioned JSON artifacts. The comparison fixes source values, mapped expression semantics, valid-value masks, Python environment, worker count and output scope. A speed ratio is rendered only after numerical parity passes.

The earlier microbenchmark measures both engines reading their native persistent stores and producing four factor matrices. It excludes execution simulation, IC, costs and rolling evaluation. It remains as low-level computation evidence and must not be confused with the complete-backtest result above.

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
