# Shared Production Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the second production-data read in one-expression screen/rolling evaluation while preserving all factor and backtest evidence.

**Architecture:** Wrap the existing batched frame loader with a transparent iterator that forwards each frame to the unchanged factor builder and simultaneously records the four execution panels. A memory-bounded accumulator finalizes each yearly chunk and materializes the existing `ProductionExecutionContext` only after every expected chunk completes.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyArrow, pytest, uv, existing production benchmark harness.

## Global Constraints

- Keep the public CLI, job schema, factor semantics and result schema backward compatible.
- Do not retain all raw production DataFrames after a batch is consumed.
- Do not add runtime dependencies or change expression/operator semantics.
- Factor values, finite masks, selections, returns, costs, Rank IC, rolling folds and gates must remain unchanged.
- Formal median runtime must be at most 8.3105 seconds and peak RSS at most 650 MiB on the versioned 500 by 1,500 fixture.
- Work directly on the current `main` branch; do not create a worktree or subagent.

---

### Task 1: Specify single-pass capture behavior with failing tests

**Files:**
- Create: `tests/test_shared_production_context.py`
- Modify: `tests/test_evaluate_factor_evidence.py`

**Interfaces:**
- Consumes: existing `YearChunk`, `BatchedProductionFrameLoader`, `FactorEvaluationService` and `ProductionExecutionContext`.
- Produces test requirements for `ChunkedProductionExecutionAccumulator.capture_frames`, `ChunkedProductionExecutionAccumulator.build` and `ExecutionCapturingFrameLoader.iter_frames`.

- [x] **Step 1: Add an accumulator lifecycle test**

Create a minimal execution frame and a `YearChunk`. Assert that `build()` before capture raises, a fully consumed `capture_frames()` produces the expected context, and capturing the same chunk again raises.

```python
accumulator = ChunkedProductionExecutionAccumulator(
    (chunk,), eligibility_column="signal_eligible", price_storage_dtype="float32"
)
with pytest.raises(ValueError, match="not complete"):
    accumulator.build()
list(accumulator.capture_frames(chunk, (frame,)))
context = accumulator.build()
assert context.valuation_open.dtype == np.float32
with pytest.raises(ValueError, match="already captured"):
    list(accumulator.capture_frames(chunk, (frame,)))
```

- [x] **Step 2: Add a captured-versus-legacy context test**

Generate the 240-date, 12-security fixture, prepare its production job, evaluate one expression through `ExecutionCapturingFrameLoader`, and compare every context array with `build_chunked_production_execution_context`.

```python
np.testing.assert_array_equal(captured.dates, legacy.dates)
np.testing.assert_array_equal(captured.codes, legacy.codes)
np.testing.assert_array_equal(captured.valuation_open, legacy.valuation_open)
np.testing.assert_array_equal(captured.buyable, legacy.buyable)
np.testing.assert_array_equal(captured.sellable, legacy.sellable)
np.testing.assert_array_equal(captured.signal_eligible, legacy.signal_eligible)
```

- [x] **Step 3: Add a public-service physical-read-count test**

Monkeypatch `BatchedProductionFrameLoader.iter_frames`, run the demo screen evaluator, and require exactly one iterator request per yearly chunk.

```python
original = BatchedProductionFrameLoader.iter_frames
requests = []
def counted(self, start, end):
    requests.append((pd.Timestamp(start), pd.Timestamp(end)))
    yield from original(self, start, end)
monkeypatch.setattr(BatchedProductionFrameLoader, "iter_frames", counted)
FactorEvaluationService().screen(job_path, "ts_pct_change(close,5)", work_root=tmp_path)
assert len(requests) == 1
```

- [x] **Step 4: Run the focused tests and verify RED**

Run: `uv run pytest -q tests/test_shared_production_context.py tests/test_evaluate_factor_evidence.py`

Expected: failures because the accumulator/wrapper do not exist and the public service currently requests each chunk twice.

---

### Task 2: Implement the bounded execution accumulator and loader wrapper

**Files:**
- Modify: `src/ashare_factor_backtest/evaluation/production_execution_context.py`
- Test: `tests/test_shared_production_context.py`

**Interfaces:**
- Produces: `ChunkedProductionExecutionAccumulator(chunks, *, price_storage_dtype, eligibility_column)`.
- Produces: `capture_frames(chunk, frames) -> Iterator[pd.DataFrame]` and `build() -> ProductionExecutionContext`.
- Produces: `ExecutionCapturingFrameLoader(frame_loader, chunks, *, price_storage_dtype, eligibility_column)` with `iter_frames`, `__call__`, `execution_context` and metadata proxying.

- [x] **Step 1: Add the accumulator skeleton and validation**

Store immutable expected chunk keys, completed keys and four lists of finalized chunk panels. Reject empty/duplicate chunk contracts, unfinished builds and duplicate capture.

```python
class ChunkedProductionExecutionAccumulator:
    def __init__(self, chunks, *, price_storage_dtype=np.float64,
                 eligibility_column="signal_eligible"):
        self._chunks = tuple(chunks)
        self._expected = {_chunk_key(chunk): chunk for chunk in self._chunks}
        if not self._chunks or len(self._expected) != len(self._chunks):
            raise ValueError("production execution capture chunks must be unique")
        self._completed = set()
        self._collected = {name: [] for name in _EXECUTION_PANEL_NAMES}
```

- [x] **Step 2: Implement streaming capture and normal-completion commit**

For each yielded frame, build execution panels and retain only those panels. Combine symbol batches only after the source iterator finishes normally, slice to the chunk calculation range, and then commit the chunk. An exception must leave no completed chunk.

```python
def capture_frames(self, chunk, frames):
    key = _chunk_key(chunk)
    if key in self._completed:
        raise ValueError("production execution chunk already captured")
    parts = {name: [] for name in _EXECUTION_PANEL_NAMES}
    for frame in frames:
        panels = _execution_panels(frame, eligibility_column=self._eligibility_column)
        for name, panel in panels.items():
            parts[name].append(panel)
        yield frame
    finalized = _combine_symbol_batch_panels(parts)
    self._commit_chunk(chunk, finalized)
```

- [x] **Step 3: Implement final context materialization**

Require every expected chunk, concatenate finalized panels across dates, reject duplicate dates, and call the existing `_from_panels` so numerical semantics remain centralized.

- [x] **Step 4: Implement the transparent loader wrapper**

Map exact `(load_start, load_end)` requests to expected chunks, delegate to the source loader's iterator or callable, and wrap that iterable with `capture_frames`. Explicit properties copy `additional_field_specs` and `additional_dataset_versions`.

- [x] **Step 5: Run accumulator tests and verify GREEN**

Run: `uv run pytest -q tests/test_shared_production_context.py`

Expected: all accumulator, parity, metadata and lifecycle tests pass.

- [x] **Step 6: Commit the component**

```bash
git add src/ashare_factor_backtest/evaluation/production_execution_context.py tests/test_shared_production_context.py
git commit -m "Add streaming execution context capture"
```

---

### Task 3: Route the public evaluator through the shared pass

**Files:**
- Modify: `src/ashare_factor_backtest/application/evaluate_factor.py`
- Modify: `tests/test_evaluate_factor_evidence.py`

**Interfaces:**
- Consumes: `ExecutionCapturingFrameLoader.execution_context()` from Task 2.
- Preserves: `FactorEvaluationService.evaluate(path, expression, *, through="rolling", work_root) -> tuple[dict[str, object], tuple[str, ...]]`.

- [x] **Step 1: Wrap the prepared loader before expression evaluation**

```python
capturing_loader = ExecutionCapturingFrameLoader(
    prepared.frame_loader,
    prepared.chunks,
    price_storage_dtype="float32",
    eligibility_column=job.view,
)
evaluated = evaluate_expression_by_year(
    expression,
    chunks=prepared.chunks,
    frame_loader=capturing_loader,
    dataset_version=f"{job.dataset_version}_{prepared.job_identity[:16]}",
    view=job.view,
    cache_max_bytes=job.evaluation.cache_mib * 1024 * 1024,
    required_fields=set(referenced_fields(expression)),
    spill_to_disk=True,
)
execution = capturing_loader.execution_context()
```

Remove the second call to `build_chunked_production_execution_context`. Continue reporting a separate `execution_context` timing for final context materialization. Do not add a new result field because the existing stage artifact identity excludes timings and the public response schema should remain stable.

- [x] **Step 2: Run focused service tests**

Run: `uv run pytest -q tests/test_evaluate_factor_evidence.py tests/test_shared_production_context.py`

Expected: one physical iterator request per chunk; screen/rolling evidence and Rank IC tests pass.

- [x] **Step 3: Run the full suite and lint**

Run: `uv run pytest -q && uv run ruff check src tests tools benchmarks`

Expected: zero failures and zero lint findings.

- [x] **Step 4: Commit the service integration**

```bash
git add src/ashare_factor_backtest/application/evaluate_factor.py tests/test_evaluate_factor_evidence.py
git commit -m "Reuse factor data pass for execution context"
```

---

### Task 4: Prove semantic parity and measure formal performance

**Files:**
- Create: `benchmarks/ashare_factor_backtest/compare_full_research_evidence.py`
- Modify: `tests/test_benchmark_harness.py`
- Create: `benchmarks/results/shared_context_v1/result.json`
- Create: `benchmarks/results/shared_context_v1/report.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `BENCHMARKS.md`
- Modify: private `TODO/20260722_ashare_factor_backtest_open_source_v01.md`

**Interfaces:**
- Consumes: versioned fixture `/tmp/ashare-full-bench-500x1500-v2/data/manifest.json`.
- Compares against: `benchmarks/results/full_backtest_v1/ours_full_research.json`.

- [x] **Step 1: Run a 240 by 12 smoke benchmark**

Run one warmup and one measurement with the generated small fixture. Confirm completion, one data pass per chunk and peak RSS below the configured job limit.

- [x] **Step 2: Compare complete semantic evidence against the baseline**

Remove only `timings_seconds`, `peak_rss_mib`, artifact paths/identities and content digests from both result trees. Require the remaining `gate`, `screen`, `rolling`, workload and warning evidence to compare exactly.

```python
IGNORED_KEYS = {
    "artifact_identity",
    "artifact_path",
    "content_digest",
    "peak_rss_mib",
    "stage_artifacts",
    "timings_seconds",
    "upstream_artifact_identities",
}

def semantic_evidence(value):
    if isinstance(value, dict):
        return {
            key: semantic_evidence(item)
            for key, item in sorted(value.items())
            if key not in IGNORED_KEYS
        }
    if isinstance(value, list):
        return [semantic_evidence(item) for item in value]
    return value
```

Run: `uv run pytest -q tests/test_benchmark_harness.py`

Expected: the comparator test passes for diagnostic-only differences and fails for a changed Rank IC or return value.

- [x] **Step 3: Run the formal benchmark**

```bash
/tmp/ashare-factor-qlib-bench-312/bin/python \
  -m benchmarks.ashare_factor_backtest.run_full_backtest_ours \
  --manifest /tmp/ashare-full-bench-500x1500-v2/data/manifest.json \
  --output-dir /tmp/ashare-shared-context-final \
  --warmup-repetitions 1 --repetitions 5
```

Expected retention gate: median `<=8.3105s`, peak RSS `<=650 MiB`, semantic evidence exact.

- [x] **Step 4: Record the result without moving the goalposts**

If the gate passes, archive JSON and update both README languages and benchmark documentation. If it fails, revert the production integration while retaining the test/experiment report and record the observed bottleneck in the private TODO.

- [x] **Step 5: Run release verification**

Run: `uv run pytest -q && uv run ruff check src tests tools benchmarks && uv build --out-dir dist`

Expected: tests, lint and build pass; the working tree contains only intended evidence/documentation changes.

- [ ] **Step 6: Commit verified evidence**

```bash
git add benchmarks/results/shared_context_v1 README.md README_EN.md BENCHMARKS.md
git commit -m "Publish shared context performance evidence"
```
