# Architecture Decision Records

Short records of decisions that a future session **must not silently reverse**. Design detail
lives in `SPEC.md`; state lives in `PROGRESS.md`; these files record *why* a choice was made
and what it costs.

Add a record when a decision would be expensive to rediscover or easy to undo by accident.
Keep it to roughly a screen. Use the next free number; never renumber an existing file.

| # | Decision | Status |
|---|---|---|
| [0001](0001-dataset-slices-and-paths.md) | Dataset slices: `dev` for building, `full` for conclusions; paths live in config | Accepted |
| [0002](0002-llm-induces-deterministic-applies.md) | The LLM induces vocabulary; deterministic code applies it at scale | Accepted |
| [0003](0003-two-insight-families-and-tiers.md) | Two insight families (sentiment, attention) and three coverage tiers | Accepted |
| [0004](0004-terminal-first-no-web-ui.md) | Terminal-first presentation; no web UI | Accepted |
| [0005](0005-artifact-and-doc-discipline.md) | Artifact and documentation discipline | Accepted |
| [0006](0006-induction-gate-truth-is-surrounding-text.md) | The induction gate's truth is surrounding text, not human labels | Accepted |
| [0007](0007-panel-cache-fingerprint.md) | The cached panel is keyed by a fingerprint checked on every load | Accepted |

## Format

```
# NNNN. Title
Status: Accepted | Superseded by NNNN
Date: YYYY-MM-DD
Context:      what forced a choice
Decision:     what was chosen, in one or two sentences
Consequences: what this buys, and what it costs or forbids
```
