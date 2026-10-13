"""Deterministic sentiment scoring: the frozen lexicon plus VADER.

Design decision embodied here: **no model scores a row**. A tweet's polarity is a pure function
of (its text, the frozen lexicon, VADER's fixed lexicon, the config weights). That is what makes
a number reproducible and what makes a vocabulary change attributable, because the lexicon id is
part of the data version every result carries.

Method (`vader_plus_lexicon`, SPEC.md 4.5 S1): the signed lexicon sum is squashed with tanh into
[-1, 1], VADER's compound is already in [-1, 1], and the two are blended with
`sentiment.lexicon_weight`. The lexicon gets real weight because it is the only part that knows
this corpus's finance vocabulary; VADER keeps the general-English core that the small hand list
cannot cover.

Negation: a negator within `negation_window` tokens before a matched term flips its sign and
damps it by `negation_damp`, because "not good" is weaker evidence than "bad". Sarcasm is not
modelled — its marker share is reported as a caveat instead, which is honest about the limit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .lexicon import FrozenLexicon, Term

# Punctuation stripped from token edges before lexicon lookup, so "beat," and "beat" match.
_EDGE = re.compile(r"^[^A-Za-z0-9$#]+|[^A-Za-z0-9%]+$")


@dataclass(frozen=True)
class Match:
    """One lexicon hit: which term, and at which token position (for negation scope)."""

    term: Term
    index: int


class LexiconMatcher:
    """Matches a frozen lexicon against text, deterministically and without any model.

    Single-word terms are matched token-exactly through a dict; multi-word terms are matched as
    substrings, because a phrase like "price target raised" must survive punctuation between its
    words. Token positions are recovered for phrases so negation scope still applies to them.
    """

    def __init__(self, lexicon: FrozenLexicon) -> None:
        self.lexicon = lexicon
        self.single: dict[str, Term] = {}
        self.phrases: list[tuple[str, Term]] = []
        for term in lexicon.terms:
            key = term.term.lower().strip()
            if " " in key:
                self.phrases.append((key, term))
            else:
                self.single[key] = term
        # Longest phrase first so "price target raised" wins over "price".
        self.phrases.sort(key=lambda pair: -len(pair[0]))
        self.negations = {n.lower().strip() for n in lexicon.negation_patterns}

    def tokens(self, text: str) -> list[str]:
        """Lowercased tokens with edge punctuation stripped. Also the drift-check tokenisation."""
        return [_EDGE.sub("", token.lower()) for token in str(text).split()]

    def match(self, text: str, tokens: list[str] | None = None) -> list[Match]:
        """Return every lexicon hit in `text`, with its token index."""
        tokens = self.tokens(text) if tokens is None else tokens
        hits: list[Match] = []
        for index, token in enumerate(tokens):
            term = self.single.get(token)
            if term is not None:
                hits.append(Match(term, index))
        lowered = str(text).lower()
        for phrase, term in self.phrases:
            start = lowered.find(phrase)
            if start >= 0:
                # Number of tokens before the phrase start gives negation scope for phrases.
                hits.append(Match(term, len(lowered[:start].split())))
        hits.sort(key=lambda hit: hit.index)
        return hits

    def polarity_of(self, text: str, tokens: list[str] | None = None) -> tuple[float, dict[str, float]]:
        """Return (signed lexicon score, family counts) for one text.

        The score is the sum of `polarity * weight` over hits, with negation applied, then
        squashed by tanh so a text stuffed with twenty bullish words cannot outweigh every other
        tweet. tanh rather than clipping because it is smooth: 1.0 is approached, never reached.
        """
        tokens = self.tokens(text) if tokens is None else tokens
        total = 0.0
        counts = {"bullish": 0.0, "bearish": 0.0, "polarity": 0.0, "uncertainty": 0.0,
                  "attribution": 0.0, "sarcasm": 0.0}
        for hit in self.match(text, tokens):
            family = hit.term.family
            counts[family] = counts.get(family, 0.0) + 1.0
            if family in ("uncertainty", "attribution", "sarcasm"):
                continue
            value = hit.term.polarity * hit.term.weight
            if self._negated(tokens, hit.index):
                value = -value * self.negation_damp
            total += value
        return math.tanh(total), counts

    def _negated(self, tokens: list[str], index: int) -> bool:
        """True when a negator appears within the configured window before token `index`."""
        window = self.negation_window
        if window <= 0 or index <= 0:
            return False
        return any(token in self.negations for token in tokens[max(0, index - window):index])

    # Set by `configure`; a property keeps the dataclass above small and immutable.
    negation_window: int = 3
    negation_damp: float = 0.75

    def configure(self, negation_window: int, negation_damp: float) -> None:
        """Apply the two negation constants from config (never hard-coded at a call site)."""
        self.negation_window = int(negation_window)
        self.negation_damp = float(negation_damp)


def _vader_scores(texts: list[str]) -> list[float]:
    """Return VADER compound scores, or zeros when the optional package is absent.

    Zeros, not an exception: `settings.sentiment.method` documents `vader_plus_lexicon` as the
    default, but a missing optional dependency must never fail a run. The caller notices the
    method downgrade from `vader_available` in the returned report.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:  # pragma: no cover - vaderSentiment is pinned in requirements.txt
        return [0.0] * len(texts)
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(text)["compound"] for text in texts]


def label_of(polarity: float, band: float) -> int:
    """Return +1 / 0 / -1 for a polarity against the label band (VADER's conventional 0.05)."""
    if polarity is None or (isinstance(polarity, float) and math.isnan(polarity)):
        return 0
    if polarity > band:
        return 1
    if polarity < -band:
        return -1
    return 0


def score_tweets(
    tweets: pd.DataFrame,
    lexicon: FrozenLexicon,
    method: str,
    label_band: float,
    lexicon_weight: float,
    negation_window: int,
    negation_damp: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add per-tweet sentiment columns and return (frame, report).

    Added columns: `pol_lex`, `pol_vader`, `polarity`, `label`, `abs_polarity`,
    `uncertainty_share`, `attribution_share`, `sarcasm_flag`, `fin_directional`.
    These are the lower-layer text features; the seven published sentiment measures are
    deductions computed per ticker-session in `features/panel.py`.
    """
    matcher = LexiconMatcher(lexicon)
    matcher.configure(negation_window, negation_damp)
    text = tweets["text_clean"].fillna("").astype(str).tolist()

    vader = _vader_scores(text) if method == "vader_plus_lexicon" else [0.0] * len(text)
    vader_available = method == "vader_plus_lexicon"
    blend = float(lexicon_weight) if vader_available else 1.0

    rows: list[dict[str, float]] = []
    for raw, vader_score in zip(text, vader):
        tokens = matcher.tokens(raw)
        lex_score, counts = matcher.polarity_of(raw, tokens)
        n_tokens = max(1, len(tokens))
        finance = counts["bullish"] - counts["bearish"]
        finance_total = counts["bullish"] + counts["bearish"] + counts["polarity"]
        rows.append(
            {
                "pol_lex": round(lex_score, 6),
                "pol_vader": round(float(vader_score), 6),
                "polarity": round((1.0 - blend) * float(vader_score) + blend * lex_score, 6),
                # Denominator is matched polarity terms, not all tokens; 0.0 when none matched,
                # which is honest ("no directional vocabulary here") rather than a divide by zero.
                "fin_directional": round(finance / finance_total, 6) if finance_total else 0.0,
                "uncertainty_share": round(counts["uncertainty"] / n_tokens, 6),
                "attribution_share": round(counts["attribution"] / n_tokens, 6),
                "attribution_flag": bool(counts["attribution"]),
                "sarcasm_flag": bool(counts["sarcasm"]),
            }
        )

    scored = pd.concat([tweets.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    scored["label"] = scored["polarity"].map(lambda value: label_of(value, label_band))
    scored["abs_polarity"] = scored["polarity"].abs()

    report = {
        "method": method,
        "vader_available": vader_available,
        "lexicon_id": lexicon.lexicon_id,
        "lexicon_source": lexicon.source,
        "label_band": label_band,
        "lexicon_weight": blend,
        "negation_window": negation_window,
        "negation_damp": negation_damp,
        "n_tweets_scored": int(len(scored)),
        "mean_polarity": round(float(scored["polarity"].mean()), 6),
        "sarcasm_share": round(float(scored["sarcasm_flag"].mean()), 6),
        "uncertainty_share": round(float(scored["uncertainty_share"].mean()), 6),
        "attribution_share": round(float(scored["attribution_share"].mean()), 6),
    }
    return scored, report
