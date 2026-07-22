# Third-Party Dependencies

The wheel declares dependencies but does not vendor their source code.

| Dependency | Role | License family |
|---|---|---|
| NumPy | arrays and numerical operations | BSD-3-Clause and compatible bundled components |
| pandas | labeled panels and time-series operations | BSD-3-Clause |
| PyArrow | Parquet and columnar data | Apache-2.0 |
| Numba / llvmlite | compiled numerical kernels | BSD / Apache-2.0 with LLVM exception |
| PyYAML | versioned manifests | MIT |

Transitive packages and exact versions are recorded in `uv.lock`. Distributors remain responsible for preserving notices required by the dependency versions they redistribute.
