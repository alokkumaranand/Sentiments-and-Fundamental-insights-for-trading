# 0005. Artifact and documentation discipline

**Status:** Accepted · **Date:** 2026-10-12

## Context

The work is deliberately broken into phases with human confirmation at each boundary, and it
must be resumable in a **different session** — possibly by an agent with no memory of this one.
Without a written contract, a new session would either re-derive decisions already made or
silently reverse them.

## Decision

Four documents, each with one job and no overlap:

| File | Job | Updated when |
|---|---|---|
| `SPEC.md` | design truth: what the system is and why | a design decision changes |
| `docs/PROGRESS.md` | state: current phase, next action, environment traps, open decisions | at every phase boundary |
| `docs/adr/NNNN-*.md` | decisions a future session must not silently reverse | when such a decision is made |
| `reviews/PHASE_<n>_REVIEW.md` | the gate: what was built, how to verify it, what needs approval | at the end of each phase |

Supporting rules:

- every run writes `artifacts/runs/<run_id>/` containing `profile.json` and, from Phase 2,
  `results/`, `findings.json`, `report.md` and `manifest.json`;
- any change to a dataset, threshold or seed changes `data_version`, so a stored number can always
  be traced to the inputs that produced it;
- raw inputs are read-only and hashed; nothing is extracted, moved or rewritten;
- every code explanation is short and states **why** the conservative choice was made, not what
  the line does (§15 of `SPEC.md`).

## Consequences

- A new session can resume from `PROGRESS.md` alone, and the human can audit a phase without
  reading code.
- Costs one deliberate stop and one short document per phase. Accepted: the human asked for
  exactly that, and it is cheaper than a wrong assumption carried forward.
- Duplication is a real risk, so the split is strict: if a fact is a design decision it belongs in
  `SPEC.md`, if it is state it belongs in `PROGRESS.md`, and it should not be in both verbatim.
