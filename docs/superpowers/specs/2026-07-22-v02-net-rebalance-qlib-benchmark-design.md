# v0.2 Net-Rebalance Qlib Benchmark Design

## Goal

Re-establish a defensible complete-backtest performance comparison for v0.2 after the public evaluator changed from full liquidation/rebuy to per-security target-delta rebalancing.

## Options considered

1. **Use Qlib's built-in interactive order generator.** Smallest patch, but its cash reservation and fee conventions do not exactly match this project's continuous-value evaluator. A parity failure would be difficult to attribute.
2. **Use a benchmark-only target-delta adapter (selected).** Compute desired per-security value, executable sells, cash-scaled buys and terminal liquidation explicitly, then express those deltas as Qlib orders. This keeps Qlib's native account/exchange execution while making the shared contract inspectable.
3. **Keep the v0.1 full-rebalance comparison.** Rejected because it no longer describes v0.2 and would make the speed claim misleading.

Strict share-account execution is intentionally not added to the public CLI in this release. The public data contract currently has daily rectangular fields but no generic point-in-time corporate-action event stream. Publishing a private-data-dependent command would create a feature users cannot reproduce.

## Shared contract

- Deterministic 500-security by 1,500-date fixture for the release benchmark.
- `ts_pct_change(close,5)` expression in both engines.
- Daily top 20%, descending signal.
- Signal frozen on `T`; execution at `T+1` open.
- Horizon 1 for the cross-engine comparison.
- Existing holdings are retained; only target-value deltas trade.
- Sell orders execute before buys.
- Buy cash is scaled pro rata when fees prevent full target attainment.
- No lot rounding, minimum commission, limit or suspension constraints in this comparison.
- Final holdings are liquidated on the common terminal date.
- One process, one warmup, five measured repetitions for the release result.

The evaluator defines `buy_cost` as a fraction of total cash spent. Qlib defines `open_cost` as a fraction of traded security value. The adapter therefore uses `qlib_open_cost = buy_cost / (1 - buy_cost)` so bought security value and cash consumption match exactly. Sell cost already uses the same notional convention.

## Parity evidence

The comparison gate must require:

- exact return dates, factor finite masks, factor values and target-selection digest;
- daily gross and net return tolerances;
- daily turnover and cost-rate tolerances, both normalized by the previous portfolio value to match Qlib's report coordinate;
- total return, Sharpe and mean Rank IC tolerances;
- first-mismatch diagnostics when a daily series fails.

No speed ratio is rendered when any required check fails. Increasing tolerances is not an acceptable substitute for explaining a semantic mismatch.

## Validation ladder

1. Unit tests for deterministic target-delta planning, retained holdings and fee conversion.
2. A 12-security by 240-date fixture with one warmup and one measured repetition.
3. The existing 500-security by 1,500-date fixture only after the small parity gate passes.
4. Update `BENCHMARKS.md`, release readiness and both READMEs only from clean, saved evidence.

## Boundaries

This benchmark evaluates the public fast continuous-value lane. It does not claim broker-level share accounting, corporate-action cashflows, tax, queue priority, impact or capacity. Those belong to the strict audit lane and require a separate public event-data contract.
