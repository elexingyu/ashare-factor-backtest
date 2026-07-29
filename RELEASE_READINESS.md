# Release Readiness

## Version 0.2.1 Status

- Add one optional `audit-causality` command without changing the ordinary
  `evaluate` path or the v0.2 execution contract.
- The prefix-invariance regression catches a deliberately hidden one-row lead
  and accepts a backward-only rolling expression.
- The bundled demo compares 512 factor cells with zero mismatches in about
  0.20 seconds and writes a compact `causality-certificate.v1` artifact.
- Package tests (`30 passed`), Ruff, source build and wheel build pass on the
  release candidate.
- The built `0.2.1` wheel was installed into a clean Python 3.13 environment;
  `doctor` and the bundled `audit-causality` demo both returned successful
  machine-protocol responses.

## Version 0.2 Status

- Local tests, lint and source/wheel builds pass for the per-security target-delta execution contract and explicit raw/adjusted price coordinates.
- Regression coverage includes retained overlapping holdings, independent partial fills, blocked residual positions, terminal residual valuation and randomized wealth conservation.
- The three real PIT CSI500 anchor expressions were rerun under execution semantics v7; all three selected a different direction or horizon than the v0.1 contract, so v0.1 research selections are legacy evidence only. Semantics v8 adds raw order/tradability coordinates without changing the continuous-value portfolio model: a real CSI500 `ts_pct_change(close,1)` regression compared 649 numeric screen fields with a maximum absolute difference of `1.875e-13`.
- The v0.2 target-delta complete-backtest benchmark passed all 11 parity checks against Qlib `d5379c5`. Clean commit `9eda124` measured a 1.330 s median and 568 MiB peak RSS versus Qlib's 19.070 s and 1,038 MiB, a `14.34x` wall-time ratio. The v0.1 `31.77x` full-liquidation result remains historical evidence only.
- A freshly built 0.2.0 wheel was installed in an isolated Python 3.13 environment; `doctor` and the bundled demo `evaluate --through screen` completed under semantics v8 with the raw/adjusted price split and execution diagnostics present.
- GitHub Actions run `29919533330` passed test, lint, source/wheel build and clean-wheel smoke jobs on Linux with Python 3.11, 3.12 and 3.13. Version 0.2.0 is ready for release.

## Version 0.1 Historical Evidence

- Public runtime import closure contains no private search modules, credentials or personal absolute paths.
- Standalone wheel installs outside the private monorepo and completes `doctor`, `compile` and rolling demo evaluation.
- Clean local environments on Python 3.11, 3.12 and 3.13 all complete the rolling demo evaluation.
- Public fixture covers listing history, ST, suspension, open limit-up and open limit-down boundaries.
- Apache-2.0 project license, CC0 fixture dedication and direct dependency notices are present.
- Runtime, benchmark harness, source distribution, CI, contribution, security and compatibility documents are included.
- Qlib comparison requires numerical parity before a speed ratio can be rendered.
- The public benchmark was rerun from clean commit `2ea0c74750baaf6c8c84dac1767c40ace4dec8c6`; the generated evidence contains no `+dirty` marker.
- The complete-backtest benchmark was rerun from clean commit `7f1192021c91bf6506894be9de351967201fbdd8` against Qlib `d5379c520f66a39953bad76234a7019a72796fd0` in the same Python 3.12.13 environment. Every parity check passed before the `31.77x` ratio was reported.
- The v0.1.2 shared data pass reduced the full A-share rolling median from `9.777 s` to `7.256 s` and peak RSS from 459 MiB to 411 MiB; the automated retention gate confirmed exact workload and complete semantic-evidence equality.
- The 0.1.1 wheel was installed in a fresh Python 3.12 environment; `doctor`, `compile` and rolling `evaluate` passed with screen evidence, Rank IC and stage timings present in the machine response.
- GitHub Actions run `29894930464` passed test, lint, build and clean-wheel smoke jobs on Linux with Python 3.11, 3.12 and 3.13.
- GitHub Actions run `29898393255` repeated the same Python 3.11/3.12/3.13 matrix successfully for the 0.1.1 source and complete-backtest evidence.
- GitHub Actions run `29900507339` passed the same Python 3.11/3.12/3.13 matrix for the v0.1.2 shared-data-pass implementation and archived performance evidence.
- The PyPI project endpoint for `ashare-factor-backtest` returned 404 on 2026-07-22, so the name appeared available at audit time.

## Optional Next Distribution Step

Publish to PyPI only after confirming the desired name is still available and configuring a trusted publisher or scoped project token.

Search/generation, multi-factor combination, live trading and private data remain explicitly outside v0.2.
