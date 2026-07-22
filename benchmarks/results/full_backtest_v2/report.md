# v0.2 Complete Backtest Benchmark

## Contract

- Fixture: 500 synthetic securities, 1,500 dates, seed `20260722`.
- Expression: `ts_pct_change(close,5)`.
- Portfolio: daily top 20%, horizon 1, signal at close T, trade at open T+1.
- Execution: per-security target-value deltas, retained holdings preserved, sells before cash-scaled buys, terminal liquidation.
- Costs: 3 bps buy cash-spend rate and 12 bps sell notional rate.
- Environment: Python 3.12.13, arm64 macOS, 24 GiB memory, 15 logical CPUs, one process.
- Measurement: one warmup and five measured repetitions.
- Revisions: project `9eda1243f2d4850db298b03df102bdbce3566c76`; Qlib `d5379c520f66a39953bad76234a7019a72796fd0`.

## Result

| Engine | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest | 1.330 s | 567.6 MiB |
| Microsoft Qlib | 19.070 s | 1,037.5 MiB |

The Qlib / project median wall-time ratio is `14.3382x`.

All 11 parity checks passed with no first mismatch. Maximum daily gross-return error is `5.5303e-15`, net-return error is `3.0427e-15`, turnover-rate error is `4.9841e-12`, cost-rate error is `5.1296e-15`, total-return error is `2.8422e-14`, Sharpe error is `8.3822e-14`, and mean Rank IC error is `5.7432e-08`.

This is a fixed-contract synthetic benchmark, not a claim that every workload is 14.34 times faster. The common contract excludes board lots, minimum commissions, price limits, suspensions and corporate actions. Raw repetitions and exact evidence are stored beside this report.
