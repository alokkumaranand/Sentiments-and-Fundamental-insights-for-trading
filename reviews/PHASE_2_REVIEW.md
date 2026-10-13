# Phase 2 review — canonical tables, alignment, and the two feature families

**Gate: awaiting your review.** Phase 2 had to end with "alignment and leakage tests pass; row counts
reconcile; every feature has a definition and a caveat; determinism tests pass". The first three are
evidenced below; the fourth is a test.

---

## 1. What was built

| File | Purpose |
|---|---|
| `src/eda_demo/prep/clean.py` | `html.unescape`, whitespace collapse, URL → `__URL__` token, cashtags/hashtags, `promo_flag`, `reply_flag` — and a count of every transformation |
| `src/eda_demo/prep/align.py` | canonical `tweets` and `prices`, the session rule, fan-out flags, the split-continuity assertion, the discovery/holdout cut |
| `src/eda_demo/prep/scope.py` | the manual ticker subset, in one place for the profiler and every later stage |
| `src/eda_demo/features/sentiment.py` | the lexicon matcher, negation, VADER blend, per-tweet polarity |
| `src/eda_demo/features/attention.py` | A1–A7 on the full ticker × session rectangle |
| `src/eda_demo/features/price.py` | contemp/forward returns, realized vol, past-only volume z |
| `src/eda_demo/features/panel.py` | the panel: the 7 sentiment measures as per-cell deductions, the gates, the tier map, and `data_version` |
| `src/eda_demo/features/feature_book.py` | generates `docs/feature_book.md` from the panel and the definitions data |
| `run.py features` | build/reuse the cached panel and write the feature book |
| `tests/test_phase2.py` | 24 tests |

## 2. How to verify it yourself

```powershell
.\.venv\Scripts\python.exe run.py --dataset dev features            # fast
.\.venv\Scripts\python.exe run.py --dataset full features           # the corpus that matters
.\.venv\Scripts\python.exe run.py --dataset full features --no-cache # rebuild from raw
.\.venv\Scripts\python.exe -m pytest tests/test_phase2.py
```

Then read `docs/feature_book.md`: it is generated, so the caveat shown beside each measure is the one
the code enforces.

## 3. Evidence — the full corpus reconciles with Phase 0

| Fact | Phase 0 profiler | Phase 2 panel | Note |
|---|---|---|---|
| tweets read / in window | 80,793 | 80,793 / **80,758** | 35 rows past the last session — counted, flagged, excluded from the panel |
| promo-shaped rows | 45,069 (55.8%) | **45,068 (55.8%)** | identical rule, so identical number |
| ticker-sessions | — | **6,300** = 25 × 252 | complete rectangle, empty cells are real zeros |
| coverage tiers | A=3 B=17 C=5 | **A=3 B=17 C=5** | own-text share, computed from the data |
| split | 176 / 76, ends 2022-06-10 | **176 / 76, ends 2022-06-10** | chronological, by session fraction |
| forward returns at the tail | — | 25 nulls per horizon | null, not wrapped |
| realized vol / volume z | — | 6,175 units have a volume z | NaN until the baseline exists |

Two numbers differ from the profiler on purpose, and both are more correct:

- **distinct texts 64,213 vs 64,479** and **cross-posted texts 6,939 vs 6,948**: fan-out is now keyed
  on the *cleaned* text, so two tweets that differ only in their URL are recognised as the same text.
  Tier assignment is unaffected.
- **rolled share 51.7% vs the profiler's "38.7% + 22.1%"**: the profiler's two buckets overlap and
  count rows on a non-trading date *or* after the close; the canonical table counts rows that
  genuinely move to a later session. 51.7% is the share that actually rolls.

Feature-layer summary: mean polarity 0.1195, sarcasm-marker share 1.1%, hedging share 0.36%,
attribution share 0.28%, 215 spike cells, 41 sustained spikes.

## 4. The one number that should give you pause

**1,625 of 6,300 ticker-sessions (25.8%) get a sentiment score; 4,675 are excluded**, and the feature
book prints a second diagnostic beside it: **2,914 units** would clear `min_tweets_per_day` if the
own-text restriction were switched off. So the gates cost 1,289 usable units — 44% of the ones the
tweet minimum alone would allow — and the reason is structural: on tier B, own text is a minority of
rows, and one or two own tweets in a session is below a threshold of three.

That is the Phase 2 decision worth your attention (§6). Everything else follows the spec's defaults.

## 5. Leakage and alignment (tests, not assertions in prose)

- a tweet before the 16:00 ET close stays on its session; after the close it rolls; a Saturday tweet
  lands on Monday; a tweet past the last session is flagged, never placed;
- `attn_z` on day *d* is built from `shift(1)` plus a trailing window, so changing day *d*'s own count
  cannot change day *d*'s z (asserted directly);
- `fwd_ret_h` is null in the last *h* sessions, not wrapped;
- a flat baseline yields an undefined z rather than a division by zero;
- a share with no tweets is NaN, not 0.0 — "no tweets" is not "no promo";
- the cache round-trips **bit-identically** to a fresh build (this caught a real bug: pandas' default
  CSV parser is not correctly rounded, so a cached run and a fresh one disagreed in the fourth decimal
  of a rank correlation — see ADR-0007).

## 6. Decisions at this gate

| # | Item | Recommendation |
|---|---|---|
| 1 | **`own_text_only` for tier B/C** (spec default: on for B and C, off for A) | **Keep.** Its cost is visible and reported. Turning it off would let the generic pool into per-ticker numbers, which is the failure the tiers exist to prevent. |
| 2 | **`min_tweets_per_day: 3`** | Keep 3, but read it as "3 *own* tweets on tier B". If you want more usable units, 2 is the honest compromise; the feature book prints both counts so you can see what changes. |
| 3 | Tier assignment is per slice | Already recorded in ADR-0001: tiers are not comparable across slices. The panel stores the tier per row, so a number can never be read without it. |
| 4 | `sent_attribution` denominator | It is the share of **|polarity|** carried by reported speech. A cell with no polarity has NaN, not 0. |
| 5 | Sarcasm | Not modelled; `sent_sarcasm_share` (1.1% of tweets) is reported as a caveat metric. A finance-tuned transformer can sit behind the same interface later. |

## 7. What Phase 3 does with it

The panel is persisted as `artifacts/cache/panel_<data_version>.csv` and re-served only when a
fingerprint over the inputs, both configs, the lexicon id, the code version and the cache schema
matches. Every `ToolResult` carries the panel's `data_version`, so a chart or a finding can be traced
to the exact table that produced it.
