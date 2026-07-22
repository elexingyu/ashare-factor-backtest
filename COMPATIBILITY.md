# Compatibility Policy

The public CLI, dataset manifest, job manifest and artifact schemas are versioned independently.

- Patch releases preserve valid commands and schema meaning.
- Minor releases may add optional fields and commands but do not silently change existing execution semantics.
- Breaking changes require a new schema or protocol version and a documented migration path.
- Artifact cache reuse is allowed only when expression, data identity, job contract and evaluation-semantics identity all match.

The current machine protocol is `ashare-backtest.protocol.v1`. The current production job input remains `production-job.v1`; it will be renamed only through an explicit compatibility release.
