# Changelog

## 0.2.7 - 2026-08-13

- Correct immutable daily measurement so zero-exposure trading days contribute zero
  return to the calendar-day series instead of being dropped.
- Keep nonzero books with missing held-security returns explicitly missing; add
  `gross_exposure`, active-book counts and active-day return diagnostics so callers can
  distinguish no position from unavailable valuation evidence.
- Add job `evidence_mode` and `symbol_cap` to immutable measurement metadata. The
  corrected full demo replay did not change its previously reported return statistics.

## 0.2.6 - 2026-08-13

- Add `measure-factor`, an immutable daily factor measurement for external Agents.
- Freeze direction and the `signal t -> t+1 open -> t+2 open` clock; report continuous
  rank returns, Rank IC, turnover, fixed Top20%, quintiles, horizon decay, yearly and
  rolling diagnostics without adding account or admission semantics.
- Save content-addressed daily traces and Top20% memberships so descriptive profiling
  and later research gates can reuse the same evidence instead of recomputing it.

## 0.2.5 - 2026-08-13

- Add the public return-blind `audit-factor` command for external research Agents.
- Report expression and data identities, field clocks, coverage inside the actual PIT
  universe, target-column isolation and prefix invariance in one compact artifact.
- Keep Alpha thresholds, factor search, research-stage decisions and library admission
  outside the backtester.

## 0.2.4 - 2026-08-13

- Add the versioned `evaluate-batch` command for Agent-owned expression batches.
  Candidates share one data and execution-context pass while retaining independent
  screen, rolling, rejection and artifact evidence.
- Add the `daily_factor` research mode so daily portfolio rebalancing is not
  conflated with diagnostic factor-decay horizons.
- Expose job-bound field catalogs, including validated PIT plugin fields, during
  compilation without loading return data.
- Keep factor generation, candidate selection, Alpha Library management and
  multi-factor portfolio construction outside this package.

## 0.2.2 - 2026-08-04

- Reconcile the optional prefix-invariance audit with the newer dynamic-universe,
  temporal-category and expanded production-operator implementation.
- Add the machine-readable `inspect-job` command so external Agents can freeze the
  engine version, protocol, job identity, data assets and execution contract before
  reserving an experiment.
- Make contract and job identities portable across installation directories by
  hashing data content and semantics rather than absolute local paths.
- Preserve the one-way integration boundary: factor-search Agents may call this CLI,
  while this package remains independent of all search and Agent implementations.

## 0.2.1 - 2026-07-29

- Add the optional `audit-causality` command and a compact
  `causality-certificate.v1` artifact.
- Detect expression implementations whose historical values change when future
  rows are removed, without adding work to ordinary `evaluate` calls.
- Add a regression fixture that proves a deliberately hidden one-row lead is
  rejected while a backward-only rolling expression remains invariant.

## 0.2.0 - 2026-07-22

- Replace whole-sleeve liquidation and rebuy with per-security target-delta rebalancing.
- Retain overlapping holdings without synthetic round trips or duplicate fees.
- Continue executable orders when another security is suspended or limit-down, while carrying the blocked position as an explicit residual.
- Preserve unsellable terminal holdings at marked value instead of writing them down to zero.
- Report retained, bought, sold, blocked and residual names plus planned/actual turnover and target-tracking error.
- Preserve raw OHLC and adjustment factors beside point-in-time back-adjusted OHLC.
- Use raw opens and raw exchange limits for order/tradability coordinates while retaining continuous adjusted opens for fast valuation.
- Emit a machine-readable warning when scale-dependent back-adjusted price levels enter cross-sectional selection, while leaving scale-invariant return and ratio expressions unaffected.
- Version the new execution semantics so v0.1 artifacts cannot be reused silently.
- Replace the archived v0.1 full-liquidation comparison with a v0.2 target-delta benchmark: all 11 parity checks pass and the clean 500-by-1,500 workload measures 1.330 seconds versus Qlib's 19.070 seconds (`14.34x`) with lower peak RSS.

## 0.1.2 - 2026-07-22

- Reuse the factor data pass to build A-share execution matrices without a second persistent-data read.
- Add lifecycle and array-parity tests for the streaming execution-context collector.
- Add an automated retention gate for semantic evidence, runtime and peak memory.
- Reduce the versioned full A-share rolling benchmark median from 9.777 to 7.256 seconds while preserving evidence exactly.

## 0.1.1 - 2026-07-22

- Add production Rank IC evidence with explicit next-open horizon semantics.
- Preserve screen evidence and stage-level timings in full rolling evaluations.
- Add a deterministic, complete 500-security by 1,500-date benchmark against Qlib.
- Require matching environments and output parity before reporting cross-framework speed.

## 0.1.0 - 2026-07-22

- Add a safe, typed factor-expression compiler and production operator catalog.
- Add explicit A-share PIT universe, suspension and open-limit execution semantics.
- Add next-open long-only evaluation with costs, IC, drawdown, turnover and rolling evidence.
- Add a one-line JSON CLI for AI callers: `doctor`, `capabilities`, `schema`, `compile` and `evaluate`.
- Add a CC0 synthetic fixture, standalone wheel, reproducible benchmark harness and CI for Python 3.11-3.13.
- Publish Chinese-first and English README files generated from the same versioned benchmark evidence.
