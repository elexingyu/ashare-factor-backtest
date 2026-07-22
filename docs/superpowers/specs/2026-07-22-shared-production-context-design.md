# Shared Production Context Design

## Goal

Reduce the public A-share rolling evaluator's end-to-end runtime by eliminating its second persistent-data read while preserving every factor and backtest semantic.

The measured v0.1.1 baseline is 9.7770 seconds median for 500 securities by 1,500 dates. Factor/chunk loading and execution-context construction account for approximately 7.16 seconds combined.

## Chosen Approach

Use a streaming execution-data tap around the existing production frame loader.

For each yearly chunk and symbol batch:

1. The underlying loader builds the production frame once.
2. The frame continues unchanged into the existing factor-context builder.
3. A side collector extracts only `hfq_open`, eligibility, suspension and price-limit fields.
4. The collector converts those fields into bounded per-chunk execution panels.
5. Raw batch frames are released under the existing memory discipline.
6. After all chunks complete, the collector combines the panels into the existing `ProductionExecutionContext` type.

The factor compiler, expression evaluator, screen, rolling evaluator and public JSON contracts remain unchanged.

## Alternatives Rejected

### Cache complete frames

This is simpler but retains all raw production columns across the factor pass. It risks doubling memory and undermines the current bounded-batch architecture.

### Replace both paths with a new unified data model

This could provide a higher long-term ceiling, but it would change too many ownership boundaries at once and increase correctness risk. It is not justified before measuring the smaller streaming change.

## Components

- A reusable execution accumulator owns batch observation, chunk finalization and final context materialization.
- A loader wrapper delegates to the existing loader while sending each yielded frame to the accumulator.
- The current standalone execution builder is rewritten to use the same accumulator, preventing two independent implementations.
- `FactorEvaluationService` wraps its prepared loader for factor evaluation and obtains the execution context from the completed accumulator instead of rereading storage.

The wrapper must proxy `additional_field_specs` and `additional_dataset_versions` so plugin and sidecar expression behavior remains unchanged.

## Error Handling

- A chunk may be captured only once.
- A chunk must finish normally before its execution panels become available.
- Missing required execution columns, duplicate date/security keys, duplicate chunk dates and mismatched chunk ranges remain hard errors.
- Requesting the final execution context before all expected chunks finish is a hard error.
- Exceptions during factor evaluation must not expose a partial context as valid.

## Verification

Tests are written before implementation and must prove:

- a loader batch is physically requested once, not once per consumer;
- shared and legacy execution contexts are array-identical;
- public screen and rolling evidence remains identical after timing fields are removed;
- plugin metadata survives the wrapper;
- incomplete or repeated chunk capture is rejected.

The formal benchmark uses the versioned 500 by 1,500 fixture, Python 3.12.13, one warmup and five measurements. The implementation is retained only if median runtime is at most 8.3105 seconds and peak RSS is at most 650 MiB, with all semantic evidence unchanged from the v0.1.1 baseline.

## Scope Boundary

This change optimizes one-expression full evaluation. It does not yet share factor panels across multiple expressions, add operators, alter the search system or claim any improvement in alpha discovery.
