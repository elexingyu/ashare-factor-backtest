# Contributing

## Development

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check src tests tools
uv build
```

Behavior changes require a failing test first. Changes to execution semantics must include a deterministic fixture or golden test covering dates, security axes, costs and blocked trades. Performance claims require a benchmark artifact containing the workload identity, cache state, hardware, output digest and parity evidence.

Do not submit proprietary market data, API tokens, private strategy results or examples that imply investment performance. The bundled demo is synthetic and exists only to test software behavior.
