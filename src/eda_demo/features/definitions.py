"""Every panel column, with its definition and its caveat — as data, not as a docstring.

Design decision embodied here: **caveats live in code as data** (SPEC.md 15). The feature book,
the terminal overview and any report can therefore quote the same caveats the code enforces,
and a tool that returns a suspicious number can point at the entry rather than at prose nobody
reads at runtime.

The seven sentiment measures and the seven attention measures are marked, because those two
families are the demo's headline result; everything else supports them.
"""

from __future__ import annotations

from typing import Any

# family, definition, caveat. `headline` marks the 7 + 7 published measures.
FEATURES: dict[str, dict[str, Any]] = {
    # --- attention family (SPEC.md 4.6) --------------------------------------
    "n_tweets": {
        "family": "attention", "headline": "A1",
        "definition": "raw tweet rows for the ticker-session",
        "caveat": "structural, not attention: 5 tickers are 100% cross-posted, so this counts scraping, not interest",
    },
    "n_texts_unique": {
        "family": "attention", "headline": "A2",
        "definition": "distinct cleaned texts, so exact reposts count once",
        "caveat": "near-duplicates and templated promos still count as distinct",
    },
    "n_texts_own": {
        "family": "attention", "headline": "A3",
        "definition": "distinct texts that appear under no other ticker",
        "caveat": "the ticker-specific measure; it is near zero for tier-C tickers by construction",
    },
    "n_texts_shared": {
        "family": "attention", "headline": "A4",
        "definition": "distinct texts that also appear under another ticker",
        "caveat": "market-wide spillover, and a scraping artefact as much as trader behaviour",
    },
    "own_text_share": {
        "family": "attention", "headline": "A5",
        "definition": "A3 / A1 for the cell; the per-ticker tier detector over the whole window",
        "caveat": "window-dependent: NIO is tier A on the full corpus and tier B on the 26-session dev slice",
    },
    "attn_velocity": {
        "family": "attention", "headline": "A6",
        "definition": "session-over-session ratio change in A3",
        "caveat": "undefined (NaN) when the previous session had no own text, rather than infinite",
    },
    "attn_z": {
        "family": "attention", "headline": "A7",
        "definition": "robust z-score of A3 against a trailing 20-session median/MAD baseline",
        "caveat": "past-only via shift(1); NaN until 5 prior sessions exist and on a flat (MAD=0) baseline",
    },
    "spike": {
        "family": "attention", "headline": None,
        "definition": "|attn_z| >= attention.spike_z",
        "caveat": "a threshold on a robust z, not a significance test",
    },
    "spike_sustained": {
        "family": "attention", "headline": None,
        "definition": "a spike that held for attention.sustained_sessions consecutive sessions",
        "caveat": "distinguishes a pattern from a one-day burst, but a two-day burst is still short",
    },
    "promo_share": {
        "family": "attention", "headline": None,
        "definition": "share of the cell's rows flagged promo (URL or >=4 cashtags)",
        "caveat": "55.8% of the corpus is promo-shaped, so this column is a data-quality signal",
    },
    "reply_share": {
        "family": "attention", "headline": None,
        "definition": "share of rows starting with @",
        "caveat": "the dataset has no author field, so replies cannot be grouped into conversations",
    },
    "cashtag_share": {
        "family": "attention", "headline": None,
        "definition": "share of rows containing at least one cashtag",
        "caveat": "cashtag-dense rows (>=2) are also sampled as their own stratum in induction",
    },
    "share_of_all_tweets": {
        "family": "attention", "headline": None,
        "definition": "the cell's rows as a share of all rows in the panel",
        "caveat": "TSLA is 46.3% of the corpus, so a pooled statistic is nearly a TSLA statistic",
    },
    # --- sentiment family (SPEC.md 4.5) --------------------------------------
    "sent_net": {
        "family": "sentiment", "headline": "S1",
        "definition": "mean signed polarity over the cell's usable tweets; lexicon (tanh-squashed) blended with VADER",
        "caveat": "both components are offline and deterministic, but both are general-purpose and weak on finance slang and sarcasm",
    },
    "sent_pos_share": {
        "family": "sentiment", "headline": "S2",
        "definition": "share of usable tweets above the label band (+0.05)",
        "caveat": "the band is VADER's convention, not a calibrated cut for this corpus",
    },
    "sent_neg_share": {
        "family": "sentiment", "headline": "S2",
        "definition": "share of usable tweets below the label band (-0.05)",
        "caveat": "S2 is the difference of the two shares; both are reported so the difference can be read in context",
    },
    "sent_intensity": {
        "family": "sentiment", "headline": "S3",
        "definition": "mean absolute polarity in the cell",
        "caveat": "separates 'mildly positive, many tweets' from 'strongly positive, few'",
    },
    "sent_dispersion": {
        "family": "sentiment", "headline": "S4",
        "definition": "within-session standard deviation of polarity",
        "caveat": "a session can be net-neutral and deeply split; dispersion is not a magnitude of sentiment",
    },
    "sent_agreement": {
        "family": "sentiment", "headline": "S4",
        "definition": "majority side's share among labelled tweets only",
        "caveat": "NaN when no tweet in the cell is labelled, rather than an assumed 1.0",
    },
    "fin_directional": {
        "family": "sentiment", "headline": "S5",
        "definition": "(bullish - bearish term matches) / matched polarity terms, lexicon only",
        "caveat": "denominator is matched terms, so it is 0.0 when the cell uses no directional vocabulary at all",
    },
    "sent_uncertainty": {
        "family": "sentiment", "headline": "S6",
        "definition": "share of the cell's tokens in the uncertainty/hedging list",
        "caveat": "hedging is a distinct signal from polarity and is deliberately not folded into it",
    },
    "sent_attribution": {
        "family": "sentiment", "headline": "S7",
        "definition": "share of |polarity| carried by tweets with a reported-speech marker",
        "caveat": "marker matching is substring-based; a reported-cautious tweet is not a bullish author, which is the point",
    },
    "sent_sarcasm_share": {
        "family": "sentiment", "headline": None,
        "definition": "share of usable tweets containing a sarcasm marker",
        "caveat": "a caveat metric, not a correction: sarcasm is not modelled in the polarity score",
    },
    "n_sent_tweets": {
        "family": "sentiment", "headline": None,
        "definition": "tweets that passed the gates and were used for the sentiment measures",
        "caveat": "lower than n_tweets wherever own-text-only applies or promo rows are excluded",
    },
    # --- price features (SPEC.md 4.7) ---------------------------------------
    "ret_contemp": {
        "family": "price", "headline": None,
        "definition": "close(d-1) -> close(d) on adj_close",
        "caveat": "descriptive co-movement only; never labelled predictive",
    },
    "abs_ret": {
        "family": "price", "headline": None,
        "definition": "|ret_contemp|",
        "caveat": "the standard attention/volatility companion; it discards direction",
    },
    "fwd_ret_1": {
        "family": "price", "headline": None,
        "definition": "close(d) -> close(d+1), the h=1 horizon",
        "caveat": "null in the last session; overlapping at h>1, which is why CIs use a block bootstrap",
    },
    "fwd_ret_2": {
        "family": "price", "headline": None,
        "definition": "close(d) -> close(d+2), the h=2 horizon",
        "caveat": "null in the last two sessions; overlapping observations, so n_units is not n_obs",
    },
    "fwd_ret_5": {
        "family": "price", "headline": None,
        "definition": "close(d) -> close(d+5), the h=5 horizon",
        "caveat": "null in the last five sessions; forward returns are the only returns used in predictive-style analysis",
    },
    "rv_5": {
        "family": "price", "headline": None,
        "definition": "standard deviation of ret_contemp over the last 5 sessions",
        "caveat": "uses returns through day d; on a short slice it is noisy",
    },
    "rv_20": {
        "family": "price", "headline": None,
        "definition": "standard deviation of ret_contemp over the last 20 sessions",
        "caveat": "needs 20 sessions of history; the dev slice barely provides them",
    },
    "volume_z": {
        "family": "price", "headline": None,
        "definition": "log-volume z-score against a trailing 20-session baseline",
        "caveat": "past-only (shifted baseline); NaN while the baseline is flat or too short",
    },
    "close": {
        "family": "price", "headline": None,
        "definition": "daily close, already split-adjusted in this file",
        "caveat": "verified split-adjusted; the profiler asserts it on every run rather than assuming it",
    },
    "adj_close": {
        "family": "price", "headline": None,
        "definition": "daily adjusted close, the return basis",
        "caveat": "differs from close only by dividends here (2,761 rows)",
    },
    "volume": {
        "family": "price", "headline": None,
        "definition": "daily share volume",
        "caveat": "not comparable across tickers, which is why only the z-score is used",
    },
    # --- quality flags -------------------------------------------------------
    "tier": {
        "family": "quality", "headline": None,
        "definition": "coverage tier A/B/C from the ticker's own-text share",
        "caveat": "assigned per dataset slice; tiers must not be compared across slices",
    },
    "low_coverage": {
        "family": "quality", "headline": None,
        "definition": "fewer usable tweets in the cell than quality.min_tweets_per_day",
        "caveat": "the sentiment columns are left null for these cells rather than scored on one tweet",
    },
    "excluded_from_tests": {
        "family": "quality", "headline": None,
        "definition": "the cell may not be used in inference: tier C, or below the tweet minimum",
        "caveat": "exclusion is counted and named in every tool result's caveats, never silent",
    },
    "period": {
        "family": "quality", "headline": None,
        "definition": "discovery or holdout, by session date",
        "caveat": "the explorer may only read discovery; the coordinator runs the holdout check",
    },
}

HEADLINE_MEASURES = {key: value["headline"] for key, value in FEATURES.items() if value.get("headline")}
