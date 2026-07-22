# Benchmark Methodology

Performance claims are accepted only when both engines use the same deterministic source values, mapped expression semantics, valid-window masks, Python environment, process count and output contract. Cross-framework speed is reported only after numerical parity passes. The raw evidence is versioned under `benchmarks/results/`.

## Complete Backtest Benchmark

The `full_backtest_v1` fixture contains 500 synthetic securities and 1,500 business dates beginning on 2018-01-02. Its seed is `20260722`; the dataset identity is `d9972bc1d1abdcf49ce6f39f32c053d7f3151e30c3da919f81e2e349d3e38793`.

The common contract is deliberately narrow enough to implement identically in both engines:

- expression: `ts_pct_change(close,5)`, mapped to `$signal_close/Ref($signal_close,5)-1` in Qlib;
- signal observed at close T, daily top 20% selected, execution at open T+1;
- one-day open-to-open holding return;
- buy cost 3 bps and sell cost 12 bps, including terminal liquidation;
- outputs: factor matrix, daily target selections, gross/net returns, NAV metrics, drawdown, turnover, Rank IC and serialized artifacts.

Qlib's account marks positions with its `$close` field. To preserve the shared open-to-open valuation contract without leaking the signal, the fixture stores the causal close signal as `$signal_close` and maps Qlib's valuation `$close` to the same-day open. The runner uses Qlib's native `D.features` and `backtest_daily`; its order generator performs full liquidation and rebuy so turnover and costs match this project's fixed-policy evaluator.

Before the final timing run, these tolerances were frozen:

| Check | Requirement |
| --- | ---: |
| dates, finite masks, finite factor values, target selections | exact |
| maximum gross-return error | `1e-10` |
| maximum net-return error | `1e-6` |
| total-return error | `1e-4` |
| Sharpe error | `1e-3` |
| mean Rank IC error | `1e-6` |

Both engines ran in the same Python 3.12.13 environment on an arm64 macOS machine with 24 GiB memory and 15 logical CPUs. Each runner used one process, one warmup and five measured repetitions. The project source was clean commit `7f1192021c91bf6506894be9de351967201fbdd8`; Qlib was clean commit `d5379c520f66a39953bad76234a7019a72796fd0`.

| Engine | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest | 0.804 s | 600 MiB |
| Microsoft Qlib | 25.539 s | 1,087 MiB |

All parity checks passed. The Qlib / project median wall-time ratio is `31.7659x`. See `benchmarks/results/full_backtest_v1/common_summary.json` for exact errors and `ours_common.json` / `qlib_common.json` for all repetitions and stage timings.

## Full A-Share Research Benchmark

The same fixture also contains deterministic point-in-time listing, ST, suspension and price-limit events. Running the public `evaluate --through rolling` path performs:

- persistent data reads and expression evaluation;
- point-in-time universe and A-share execution-context construction;
- real and stressed two-sided cost simulations;
- screen selection plus five rolling train/test folds;
- 54 strategy and 36 benchmark portfolio simulations;
- 12 Rank IC evaluations;
- atomic JSON artifact writes.

One warmup plus five measured repetitions produced a median of `9.777 s` and process peak RSS of `459 MiB`. This is an absolute production-throughput measurement, not a Qlib speed comparison: publishing a ratio here would require an independently aligned implementation of all A-share constraints and rolling evidence semantics.

## Reproduction

The runtime package does not depend on Qlib. For comparison, clone Qlib and install both projects into the same temporary Python 3.12 environment:

```bash
git clone https://github.com/microsoft/qlib.git /tmp/qlib
git -C /tmp/qlib checkout d5379c520f66a39953bad76234a7019a72796fd0
uv venv --python 3.12 /tmp/ashare-benchmark-env
uv pip install --python /tmp/ashare-benchmark-env/bin/python -e /tmp/qlib
uv pip install --python /tmp/ashare-benchmark-env/bin/python -e .
```

Generate the deterministic fixture and Qlib binary store:

```bash
PY=/tmp/ashare-benchmark-env/bin/python
DATA=/tmp/ashare-full-benchmark/data
QLIB=/tmp/ashare-full-benchmark/qlib_data

$PY -m benchmarks.ashare_factor_backtest.prepare_full_backtest \
  --output-dir "$DATA" --dates 1500 --securities 500 --seed 20260722
$PY /tmp/qlib/scripts/dump_bin.py dump_all \
  --data_path "$DATA/qlib_csv" --qlib_dir "$QLIB" \
  --freq day --exclude_fields date,symbol
```

Run both common benchmarks and enforce the parity gate:

```bash
$PY -m benchmarks.ashare_factor_backtest.run_common_backtest_ours \
  --manifest "$DATA/manifest.json" --output-dir /tmp/bench-ours \
  --warmup-repetitions 1 --repetitions 5
$PY -m benchmarks.ashare_factor_backtest.run_common_backtest_qlib \
  --manifest "$DATA/manifest.json" --qlib-data-dir "$QLIB" \
  --qlib-commit d5379c520f66a39953bad76234a7019a72796fd0 \
  --output-dir /tmp/bench-qlib --warmup-repetitions 1 --repetitions 5
$PY -m benchmarks.ashare_factor_backtest.compare_common_backtest \
  --ours-dir /tmp/bench-ours --qlib-dir /tmp/bench-qlib \
  --output /tmp/bench-summary.json
```

Run the A-share production path:

```bash
$PY -m benchmarks.ashare_factor_backtest.run_full_backtest_ours \
  --manifest "$DATA/manifest.json" --output-dir /tmp/bench-full \
  --warmup-repetitions 1 --repetitions 5
```

## Expression Microbenchmark

The older `cross_framework_v1` comparison reads each engine's native persistent store and produces four mapped factor matrices: close, five-day delay, five-day percentage change and twenty-day mean. It excludes trading, costs, IC and rolling evaluation. Its archived Qlib / project ratio is `10.50x`; it is retained as a low-level expression-path benchmark and is not the headline full-backtest claim.

Qlib is an optional benchmark dependency and is intentionally absent from the runtime wheel. Any result whose project revision is dirty, whose environments differ, or whose parity gate fails must not report a speed ratio.
