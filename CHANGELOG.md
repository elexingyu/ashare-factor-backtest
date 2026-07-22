# Changelog

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
