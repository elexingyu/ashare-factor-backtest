# Shared Production Context Benchmark v1

Evidence date: 2026-07-22

- Baseline commit: `7f1192021c91bf6506894be9de351967201fbdd8`.
- Candidate commit: `7899b4447eb2ee0077f8426a2fdaa6fcd1e87f78`.
- Workload: 500 securities, 1,500 dates, five rolling folds, 90 portfolio simulations and 12 Rank IC evaluations.
- Environment: Python 3.12.13, macOS arm64, 24 GiB memory, 15 logical CPUs.
- Repetitions: one warmup followed by five measurements.

## Result

| Version | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| v0.1.1 two data passes | 9.777 s | 459.3 MiB |
| v0.1.2 shared data pass | 7.256 s | 410.9 MiB |

Median runtime improved by `25.7827%`. Candidate wall times were `[7.6012, 7.9043, 7.2562, 6.9232, 6.9150]s`.

The environment and workload matched exactly. The complete `screen`, `rolling`, `gate` and warning evidence trees also matched exactly. The pre-registered retention limits were median runtime at most `8.3105 s` and peak RSS at most `650 MiB`; both passed.

The optimization eliminates a second persistent-data read. The existing factor pass now also extracts execution panels from each streamed batch. This changes engineering throughput only; factor values, selections, returns, costs, Rank IC and rolling decisions are unchanged.
