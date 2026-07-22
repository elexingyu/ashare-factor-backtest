# Release Readiness

## Passed

- Public runtime import closure contains no private search modules, credentials or personal absolute paths.
- Standalone wheel installs outside the private monorepo and completes `doctor`, `compile` and rolling demo evaluation.
- Clean local environments on Python 3.11, 3.12 and 3.13 all complete the rolling demo evaluation.
- Public fixture covers listing history, ST, suspension, open limit-up and open limit-down boundaries.
- Apache-2.0 project license, CC0 fixture dedication and direct dependency notices are present.
- Runtime, benchmark harness, source distribution, CI, contribution, security and compatibility documents are included.
- Qlib comparison requires numerical parity before a speed ratio can be rendered.
- The public benchmark was rerun from clean commit `2ea0c74750baaf6c8c84dac1767c40ace4dec8c6`; the generated evidence contains no `+dirty` marker.
- The complete-backtest benchmark was rerun from clean commit `7f1192021c91bf6506894be9de351967201fbdd8` against Qlib `d5379c520f66a39953bad76234a7019a72796fd0` in the same Python 3.12.13 environment. Every parity check passed before the `31.77x` ratio was reported.
- The full A-share rolling path completed five folds, 90 portfolio simulations and 12 Rank IC evaluations in a median `9.777 s` on the versioned 500-security by 1,500-date fixture.
- The 0.1.1 wheel was installed in a fresh Python 3.12 environment; `doctor`, `compile` and rolling `evaluate` passed with screen evidence, Rank IC and stage timings present in the machine response.
- GitHub Actions run `29894930464` passed test, lint, build and clean-wheel smoke jobs on Linux with Python 3.11, 3.12 and 3.13.
- The PyPI project endpoint for `ashare-factor-backtest` returned 404 on 2026-07-22, so the name appeared available at audit time.

## Optional Next Distribution Step

Publish to PyPI only after confirming the desired name is still available and configuring a trusted publisher or scoped project token.

Search/generation, multi-factor combination, live trading and private data remain explicitly outside v0.1.
