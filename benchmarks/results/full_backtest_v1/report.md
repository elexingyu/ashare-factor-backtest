# Full Backtest Benchmark v1

Evidence date: 2026-07-22

- Workload: 500 securities, 1,500 dates, one expression, single process.
- Project: `7f1192021c91bf6506894be9de351967201fbdd8`.
- Qlib: `d5379c520f66a39953bad76234a7019a72796fd0`.
- Environment: Python 3.12.13, macOS arm64, 24 GiB memory, 15 logical CPUs.
- Repetitions: one warmup followed by five measurements per engine.

## Common Complete Backtest

| Engine | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest | 0.804 s | 599.7 MiB |
| Microsoft Qlib | 25.539 s | 1,086.6 MiB |

The Qlib / project wall-time ratio is `31.7659x`. The comparison gate passed every check: dates, finite factor values and daily target selections match exactly. Maximum daily net-return error is `9.2939e-08`; total-return error is `1.3229e-05`; Sharpe error is `1.5599e-04`; and mean Rank IC error is `5.7432e-08`.

## A-Share Production Path

The public rolling evaluator completed five folds, 90 portfolio simulations and 12 Rank IC evaluations in a median `9.777 s`, with process peak RSS of `459.3 MiB`. Median stages were factor/chunk loading `3.860 s`, execution-context construction `3.301 s`, rolling evaluation `2.033 s` and screen evaluation `0.580 s`.

These synthetic results establish reproducible correctness and engineering throughput for the measured contracts. They do not establish factor profitability or imply the same speed ratio for other expressions, universes, machines or execution semantics.
