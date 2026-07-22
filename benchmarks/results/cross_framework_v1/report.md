<!-- benchmark:start -->
**Evidence status:** Pre-release (dirty source tree).

| Engine | Median wall time | Peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest-public-evaluator | 0.089 s | 372 MiB |
| microsoft-qlib-local-provider-kernels-1 | 0.903 s | 690 MiB |

On this workload, Qlib / this engine wall-time ratio was `10.18x`. The workload contains 500 securities, 1500 dates and 4 mapped expressions. Maximum absolute output difference was `1.45e-05` and all finite masks matched.

This is a warm-cache, single-worker, native-store-to-factor-matrix benchmark on the same Python environment. It is not a complete backtest comparison and does not imply the same ratio for other formulas, data sizes or machines. See `BENCHMARKS.md` and the versioned JSON artifacts before quoting it.
<!-- benchmark:end -->
