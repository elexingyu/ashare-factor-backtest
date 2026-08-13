# Release Readiness

## Version 0.2.8 Status

- Add return-blind daily book preconditions to `audit-factor` for data-first robustness
  workflows without importing Agent or admission policy.
- Verify that the audit reports breadth, zero-exposure frequency, zero net exposure and
  unit active gross exposure while `return_data_read` remains false.
- Verify package tests, Ruff, source sync, source/wheel builds and clean-wheel smoke.

## Version 0.2.7 Historical Status

- Correct the zero-exposure calendar-day return rule without imputing missing returns
  for nonzero held books.
- Version the measurement engine as v2, retain v1 artifacts, and replay the complete
  demo configuration; the demo statistics remain unchanged because its two missing
  dates were held-price gaps rather than zero books.
- Verify source sync, package tests, Ruff, source/wheel builds and a clean-wheel
  `measure-factor` smoke before release.

## Version 0.2.6 Historical Status

- Add `measure-factor` with an explicit direction and immutable daily measurement
  contract, separate from executable-account and admission logic.
- Persist and hash the daily trace and Top20% membership artifacts; verify deterministic
  reuse and Agent Candidate Profile integration.
- Verify package tests, Ruff, source sync, source/wheel builds and a clean-wheel
  `measure-factor` smoke before release.

## Version 0.2.5 Status

- Add `audit-factor` as a return-blind public command whose coverage denominator is
  the job's actual PIT universe rather than a rectangular panel.
- Bind the compact artifact to expression, job, data assets, coverage and causality
  certificate identities without importing any search or Agent code.
- Verify package tests, Ruff, source sync, source/wheel builds and a clean-wheel
  `audit-factor` smoke before release.

## Version 0.2.4 Status

- Publish `evaluate-batch` as a versioned JSON command while preserving the
  single-expression economic contract and per-candidate artifacts.
- Verify that a two-expression batch reads the synthetic production chunk once and
  reproduces both single-expression screen results exactly.
- Verify that `daily_factor` uses one daily account sleeve and treats 5/10/20/60-day
  horizons as decay diagnostics rather than fixed holding periods.
- Verify package tests, Ruff, source/wheel builds, clean-wheel CLI commands, and the
  external Agent subprocess adapter before pushing the standalone repository.

## Version 0.2.2 Status

- Reconcile the v0.2.1 prefix-invariance audit with the current canonical dynamic
  universe, temporal category and expanded operator implementation.
- Add `inspect-job` as the stable pre-evaluation identity handshake for external
  factor-search Agents, without adding an Agent dependency to this package.
- Verify that copying the same job and assets to another installation directory keeps
  both contract and job identities unchanged.
- Release verification must include package tests, Ruff, source/wheel builds, a clean
  wheel install, `inspect-job`, `audit-causality` and rolling `evaluate` smoke tests.

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
