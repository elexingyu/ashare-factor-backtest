# v0.2 Net-Rebalance Qlib Benchmark Implementation Plan

1. Add failing unit tests for a benchmark-only target-delta planner and fee-coordinate conversion.
2. Implement the pure planner and wire it into the Qlib order generator without changing runtime package semantics.
3. Extend both benchmark outputs with daily turnover and cost arrays plus first-mismatch diagnostics.
4. Run package tests and Ruff.
5. Build the exact Qlib commit in an isolated Python environment and run the 12x240 parity probe.
6. If and only if the probe passes, run the 500x1500 release benchmark and save artifacts under a new `full_backtest_v2` directory.
7. Update release documentation from the resulting evidence, mirror changes to the standalone public repository without overwriting unrelated work, and verify the wheel in an isolated environment.
