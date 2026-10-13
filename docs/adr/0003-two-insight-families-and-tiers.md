# 0003. Two insight families (sentiment, attention) and three coverage tiers

**Status:** Accepted · **Date:** 2026-10-12

## Context

The human asked for a deliberately conservative scope: besides sentiment, **one** more
analysis, with attention named as it, each a top-layer deduction built on lower-layer features.
They also asked for everything to stay conservative and for the project to stay short.

Profiling then showed the corpus does not behave the way "80K tweets" suggests:

- 28.5% of rows are text cross-posted under several tickers;
- **five tickers (AMZN, MSFT, PG, F, NOC) are 100% cross-posted** — a per-ticker sentiment
  score for them measures the generic chatter pool, not the company;
- TSLA alone is 46.3% of the corpus, so a naive "pooled" number is nearly a TSLA number.

Shipping sentiment plus a raw tweet-count "attention" metric would therefore have produced
confident, wrong per-ticker results.

## Decision

**Exactly two families**, each a named set of seven measures:

- **Sentiment (7):** net polarity, positive/negative share, intensity, dispersion, finance
  directional, uncertainty/hedging, attribution — with negation and sarcasm as modifiers.
- **Attention (7):** raw rows, distinct texts, **own-text count**, **shared/market-wide count**,
  own-text share, velocity, past-only robust z-score.

**Three coverage tiers**, assigned from *measured* own-text share, written into every result:
A ≥ 0.80 (unrestricted per-ticker inference), B 0.10–0.80 (per-ticker only with own-text
restriction and the fan-out share printed), C < 0.10 (descriptive only, excluded from
inference, findings capped at `observed`).

Pooled statistics are always shown next to a per-ticker view **and** a TSLA-excluded pooled view.

## Consequences

- Per-ticker attention distinguishes *this company* from *the market talking*, which the raw
  count cannot. A3 vs A4 is the headline pair in every attention chart.
- Only 3 of 25 tickers qualify for unrestricted per-ticker inference. That is stated in the UI
  and the report rather than hidden.
- Additional families (novelty, coordination, narrative concentration) were considered and
  deferred with reasons recorded in `SPEC.md` §8.1 — novelty is the only one that needs no new
  data source, so it is the natural first extension after v0.
- Reversing this decision means re-deriving the whole feature layer, so it is recorded here.
