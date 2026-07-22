# Benchmark Methodology

Performance claims are accepted only when both engines use the same deterministic source values, mapped expression semantics, valid-window masks, Python environment, process count and output contract. Cross-framework speed is reported only after numerical parity passes. The raw evidence is versioned under `benchmarks/results/`.

## Complete Backtest Benchmark

The v0.2 `full_backtest_v2` fixture contains 500 synthetic securities and 1,500 business dates beginning on 2018-01-02. Its seed is `20260722`; the dataset identity is `e89042bdb1f1104d2164cc095fcba669a0453be88d50aba835387ed7ff13f1bb`.

The common contract is deliberately narrow enough to implement identically in both engines:

- expression: `ts_pct_change(close,5)`, mapped to `$signal_close/Ref($signal_close,5)-1` in Qlib;
- signal observed at close T, daily top 20% selected, execution at open T+1;
- one-day open-to-open holding return;
- buy cost 3 bps and sell cost 12 bps, including terminal liquidation;
- per-security target-delta rebalancing: retained names are not sold and repurchased;
- outputs: factor matrix, daily target selections, gross/net returns, NAV metrics, drawdown, turnover, cost rates, Rank IC and serialized artifacts.

Qlib's account marks positions with its `$close` field. To preserve the shared open-to-open valuation contract without leaking the signal, the fixture stores the causal close signal as `$signal_close` and maps Qlib's valuation `$close` to the same-day open. The runner uses Qlib's native `D.features`, exchange, account and `backtest_daily`. A benchmark-only order adapter computes the same target-value deltas, sells before buys and scales buys pro rata when fees consume cash. Because this project defines buy cost as a fraction of total cash spent while Qlib defines it as a fraction of traded security value, the adapter applies `qlib_open_cost = buy_cost / (1 - buy_cost)`.

Before the final timing run, these tolerances were frozen:

| Check | Requirement |
| --- | ---: |
| dates, finite masks, finite factor values, target selections | exact |
| maximum gross-return error | `1e-10` |
| maximum net-return error | `1e-6` |
| maximum turnover-rate error | `1e-6` |
| maximum cost-rate error | `1e-6` |
| total-return error | `1e-4` |
| Sharpe error | `1e-3` |
| mean Rank IC error | `1e-6` |

Both engines ran in the same Python 3.12.13 environment on an arm64 macOS machine with 24 GiB memory and 15 logical CPUs. Each runner used one process, one warmup and five measured repetitions. The project source was clean commit `9eda1243f2d4850db298b03df102bdbce3566c76`; Qlib was clean commit `d5379c520f66a39953bad76234a7019a72796fd0`.

| Engine | Median wall time | Process peak RSS |
| --- | ---: | ---: |
| ashare-factor-backtest | 1.330 s | 568 MiB |
| Microsoft Qlib | 19.070 s | 1,038 MiB |

All 11 parity checks passed with no first mismatch. The Qlib / project median wall-time ratio is `14.3382x`. Maximum daily net-return error is `3.0427e-15`; turnover-rate error is `4.9841e-12`; total-return error is `2.8422e-14`; Sharpe error is `8.3822e-14`; and mean Rank IC error is `5.7432e-08`. See `benchmarks/results/full_backtest_v2/common_summary.json` for exact errors and `ours_common.json` / `qlib_common.json` for every repetition and stage timing.

The v0.1 full-liquidation/rebuy result remains archived under `full_backtest_v1`. Its `31.7659x` ratio describes the old execution contract and must not be attributed to v0.2.

## Full A-Share Research Benchmark

The same fixture also contains deterministic point-in-time listing, ST, suspension and price-limit events. Running the public `evaluate --through rolling` path performs:

- persistent data reads and expression evaluation;
- point-in-time universe and A-share execution-context construction;
- real and stressed two-sided cost simulations;
- screen selection plus five rolling train/test folds;
- 54 strategy and 36 benchmark portfolio simulations;
- 12 Rank IC evaluations;
- atomic JSON artifact writes.

The v0.1.1 baseline used one warmup plus five measured repetitions and produced a median of `9.777 s` with process peak RSS of `459 MiB`. In v0.1.2, the factor and execution paths consume the same streamed frame batches. The same experiment produced measured wall times `[7.6012, 7.9043, 7.2562, 6.9232, 6.9150]s`, a median of `7.256 s`, and process peak RSS of `411 MiB`.

The automated retention gate reports a `25.7827%` median improvement with exact environment, workload and complete semantic-evidence equality. The original factor pass (`3.860 s`) and second execution-context pass (`3.301 s`) became a shared data/factor/panel pass (`4.569 s`) plus final matrix materialization (`0.028 s`). See `benchmarks/results/shared_context_v1/` for all repetitions and the gate output.

This remains an absolute production-throughput measurement, not a Qlib speed comparison: publishing a ratio here would require an independently aligned implementation of all A-share constraints and rolling evidence semantics.

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
$PY -m benchmarks.ashare_factor_backtest.compare_full_research_evidence \
  --baseline benchmarks/results/full_backtest_v1/ours_full_research.json \
  --candidate /tmp/bench-full/result.json \
  --output /tmp/bench-full/comparison.json \
  --maximum-median-seconds 8.3105 --maximum-peak-rss-mib 650
```

## Expression Microbenchmark

The older `cross_framework_v1` comparison reads each engine's native persistent store and produces four mapped factor matrices: close, five-day delay, five-day percentage change and twenty-day mean. It excludes trading, costs, IC and rolling evaluation. Its archived Qlib / project ratio is `10.50x`; it is retained as a low-level expression-path benchmark and is not the headline full-backtest claim.

Qlib is an optional benchmark dependency and is intentionally absent from the runtime wheel. Any result whose project revision is dirty, whose environments differ, or whose parity gate fails must not report a speed ratio.
