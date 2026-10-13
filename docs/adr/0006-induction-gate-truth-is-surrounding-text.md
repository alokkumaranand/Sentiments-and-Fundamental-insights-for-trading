# 0006. The induction gate measures agreement with surrounding text, not human labels

**Status:** Accepted · **Date:** 2026-10-12

## Context

Phase 1 had to make an LLM's proposed vocabulary safe to apply to 80,000 tweets. SPEC.md 4.3 requires
that every proposed term be re-tested on a session-disjoint slice and accepted only if its claimed
sign agrees with the validation data (`sign_agreement`, default 0.6) with a bootstrap CI that
excludes zero.

The corpus has **no sentiment labels**. There is no gold set of tweets marked positive or negative,
and this demo will not invent one — a randomly labelled sample would be worse than no check, because
it would look like validation.

## Decision

For a proposed term, the gate computes the polarity of the **text around it**: the surrounding words
scored with the shipped `default_v1.json` lexicon, with the candidate's own contribution removed so a
term already in the seed cannot validate itself. The candidate is accepted only when

1. it occurs at least `min_support` (10) times in the validation slice;
2. its claimed sign matches the surrounding polarity for at least `sign_agreement` (0.6) of the
   decisive occurrences — those whose surrounding polarity exceeds the ±0.05 label band;
3. a session-block bootstrap CI for that surrounding polarity excludes zero.

Marker families (`uncertainty`, `attribution`, `sarcasm`) are scored as shares, not polarity, so for
them the support test is the whole test and the accepted record says so (`ci_95: null`).

A CI that cannot be estimated — fewer than two session blocks — **rejects** the term. A gate that
passes when it cannot measure is not a gate.

## Consequences

- The gate is fully deterministic and needs no labels, no extra model call, and no human review.
- Accepted terms are **additive**: the induced lexicon is the shipped core plus the accepted terms,
  so a model's list of 40 words cannot silently delete 162 hand-checked ones.
- The gate's "truth" is the shipped seed lexicon. It therefore validates *agreement with a general
  sentiment prior*, not correctness on financial slang — a genuinely sarcastic term whose
  surrounding text is positive will be rejected. This is the price of having no labels, it is
  stated in every Phase 1 review and in the lexicon's own `notes`, and it is why the frozen file
  stays reviewable rather than authoritative.
- In practice most proposals fail on support: corpus slang is rare. On the demo's first live run,
  2 of 47 proposed terms were accepted. That ratio is the gate working, not the model failing.
- **Do not** replace the surrounding-text test with a model-judged one. A second model asked "is
  this term positive?" is a second chance to be wrong in the same way.
