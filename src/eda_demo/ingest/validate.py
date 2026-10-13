"""Blocking vs warning data checks.

The distinction is the whole point of this module and it is deliberately narrow:

  BLOCKING  conditions that make every downstream number meaningless. The run stops with
            a message naming the problem. There are five, from SPEC.md section 2.6.
  WARNING   everything else. Warnings never stop a run; they are carried into
            profile.json, the quality report and the run manifest, so a reader is told
            what is wrong with the data instead of being quietly handed a fragile result.

A check is only blocking if continuing would produce a *wrong* answer rather than an
*uncertain* one. A ragged final session, for example, is uncertain, not wrong, so it warns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class BlockingError(Exception):
    """A data problem severe enough that no result from this run could be trusted."""


# --- the five blocking checks -------------------------------------------------


def require_columns(frame: pd.DataFrame, required: dict[str, str], source: str) -> None:
    """Fail if any configured source column is absent.

    `required` maps a role ('date', 'ticker', ...) to the configured column name, so the
    error can say which role is missing rather than only the literal name.
    """
    missing = [f"{role}={name!r}" for role, name in required.items() if name not in frame.columns]
    if missing:
        raise BlockingError(
            f"{source}: missing required column(s) {missing}. "
            f"Present: {list(frame.columns)}. Fix config/settings.yaml -> columns."
        )


def parse_dates(frame: pd.DataFrame, column: str, source: str) -> pd.Series:
    """Parse a date column to UTC, failing on any unparseable value.

    Returns a tz-aware series. It does NOT drop bad rows: a silently dropped row would
    move every session boundary and change the panel without changing any reported count.
    """
    parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        examples = frame.loc[parsed.isna(), column].head(3).tolist()
        raise BlockingError(f"{source}: {bad} unparseable date(s) in '{column}', e.g. {examples}")
    return parsed


def require_ticker_overlap(tweet_tickers: set[str], price_tickers: set[str]) -> None:
    """Fail when the two files share no ticker, since no join is then possible."""
    shared = tweet_tickers & price_tickers
    if not shared:
        raise BlockingError(
            "tweets and prices share no ticker, so the two files cannot be joined. "
            f"tweets={len(tweet_tickers)} prices={len(price_tickers)}"
        )


def require_own_text(rows_with_own_text: int) -> None:
    """Fail when no tweet text is unique to a single ticker.

    Every per-ticker text measure would otherwise be measuring a cross-posted pool rather
    than the company, which is a wrong answer rather than an uncertain one.
    """
    if rows_with_own_text == 0:
        raise BlockingError(
            "no tweet text is unique to any single ticker: every row is cross-posted, "
            "so per-ticker sentiment/attention would measure the generic pool. See SPEC.md 2.4."
        )


def require_tier_a(min_tier_a: int = 2) -> Any:
    """Return a check for the number of tier-A tickers; fail below the minimum.

    Tier A needs at least two members so the demo can compare one ticker against another.
    With one or zero, per-ticker analysis is not a comparison and the design's pooling
    caveats cannot be demonstrated.
    """

    def _check(tier_a_count: int) -> None:
        if tier_a_count < min_tier_a:
            raise BlockingError(
                f"only {tier_a_count} ticker(s) reach coverage tier A (need >= {min_tier_a}). "
                "Lower tiers.specific_own_text_share in config/experiments.yaml only if the "
                "data genuinely changed."
            )

    return _check


# --- warnings -----------------------------------------------------------------


def collect_warnings(metrics: dict[str, Any]) -> list[str]:
    """Turn measured metrics into plain-language warnings, worst first.

    Takes a plain dict of already-computed numbers (no live dataframes) so the same
    function can run off a stored profile.json when re-checking an old run.
    """
    warnings: list[str] = []

    # Ragged panel: a session present for some tickers but not all. Seen in the 1:9 dev
    # slice, where the split cut mid-session and left 5 of 25 tickers on the final date.
    ragged = metrics.get("prices", {}).get("ragged_sessions", [])
    if ragged:
        warnings.append(
            f"{len(ragged)} price session(s) are incomplete (not all tickers present): {ragged[:5]}. "
            "Drop them before computing any cross-sectional statistic."
        )

    tiers = metrics.get("tiers", {}).get("by_ticker", {})
    tier_c = sorted(t for t, tier in tiers.items() if tier == "C")
    if tier_c:
        warnings.append(
            f"{len(tier_c)} ticker(s) are coverage tier C (100% cross-posted text): {tier_c}. "
            "They are excluded from sentiment and attention inference."
        )

    if metrics.get("sessions", 0) and metrics["sessions"] < 60:
        warnings.append(
            f"only {metrics['sessions']} trading session(s) in this dataset. Trailing-window "
            "features (20-session baselines) and any discovery/holdout split are not meaningful. "
            "Use --dataset full for real analysis."
        )

    entities = metrics.get("tweets", {}).get("html_entity_rows", 0)
    if entities:
        warnings.append(f"{entities:,} tweet(s) contain HTML entities and need html.unescape in cleaning.")

    promo = metrics.get("tweets", {}).get("promo_share")
    if promo is not None and promo > 0.20:
        warnings.append(f"{promo:.1%} of tweets are promo-shaped (URL or >=4 cashtags).")

    rolling = metrics.get("tweets", {}).get("non_trading_day_share")
    if rolling is not None and rolling > 0.10:
        warnings.append(
            f"{rolling:.1%} of tweets fall on a non-trading date and roll forward to the next session."
        )

    return warnings
