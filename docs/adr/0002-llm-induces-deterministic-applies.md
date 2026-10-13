# 0002. The LLM induces vocabulary; deterministic code applies it at scale

**Status:** Accepted · **Date:** 2026-10-12

## Context

Two failure modes had to be avoided at once:

- **A hand-written lexicon is blind to this corpus.** Financial slang, ticker shorthand,
  sarcasm and promo cadence are corpus-specific; a general-purpose lexicon scores them badly.
- **An LLM scoring every row is neither reproducible nor affordable**, and it makes every
  number unverifiable — the opposite of what this demo is meant to show.

The original draft put the LLM only *above* the deterministic core (explore and narrate). That
leaves the feature layer unable to adapt to new vocabulary, which is a real problem for a
corpus that keeps arriving.

## Decision

The LLM sits **in the feature layer, once**, with a narrow job: read a capped stratified sample
and emit a `LexiconProposal` (terms with polarity/weight and evidence, plus relevance rules and
negation/sarcasm patterns). A **deterministic validator** then re-tests every proposed term on
a session-disjoint slice and accepts only those that pass a minimum support and an agreeing
sign. Accepted terms are frozen to a hashed lexicon `artifacts/lexicons/<lexicon_id>.json`.

The LLM never scores a row, never sees the full dataset, and never writes a number. If it is
missing, unavailable, or returns junk, the shipped `config/lexicons/default_v1.json` is used
and the run still completes.

## Consequences

- Reproducibility holds: the same data and lexicon give identical `result_id`s; only prose varies.
- A bad proposal is *rejected by measurement*, not trusted — the record of what was rejected and
  why is kept in `rejected_proposals.json`.
- Cost is one call per induction, cached by prompt hash.
- New data is scored with the **frozen** lexicon at zero LLM cost; re-induction is triggered by a
  token-coverage drift check, and the new lexicon is versioned alongside the old one.
- A consequence to respect: `lexicon_id` is part of `data_version`, so changing the vocabulary
  invalidates old result ids by design. Never edit a frozen lexicon in place.
