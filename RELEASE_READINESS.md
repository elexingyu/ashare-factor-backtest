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
- The PyPI project endpoint for `ashare-factor-backtest` returned 404 on 2026-07-22, so the name appeared available at audit time.

## Required Before Public Release

1. Run GitHub Actions on Linux for Python 3.11, 3.12 and 3.13.
2. Review the generated README table against its JSON artifacts, then tag `v0.1.0`.
3. Publish to PyPI only after confirming the desired name is still available.

Search/generation, multi-factor combination, live trading and private data remain explicitly outside v0.1.
