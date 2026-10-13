"""The manual ticker subset (`config/experiments.yaml -> scope`), in one place.

Why its own module: the profiler and every later stage must apply exactly the same filter
before computing anything, or a count from one stage would describe a different dataset from
the count of the same name in another. It lived inside `ingest/profile.py` in Phase 0 and is
re-exported from there, so `build_profile` behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_scope(
    tweets: pd.DataFrame,
    prices: pd.DataFrame,
    tcols: dict[str, str],
    pcols: dict[str, str],
    scope: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Restrict both files to the configured ticker subset.

    Returns (tweets, prices, record). `include: null` means every ticker; `exclude` always
    wins. The filter runs before any metric, so every reported count describes the data
    actually in use rather than what is on disk -- a profile of a subset must never look
    like a profile of the whole file.
    """
    include = scope.get("include") or None
    exclude = list(scope.get("exclude") or [])
    record: dict[str, Any] = {"include": include, "exclude": exclude, "applied": False}
    if include is None and not exclude:
        return tweets, prices, record

    def keep(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        mask = pd.Series(True, index=frame.index)
        if include is not None:
            mask &= frame[column].isin(include)
        if exclude:
            mask &= ~frame[column].isin(exclude)
        return frame[mask]

    kept_tweets = keep(tweets, tcols["ticker"])
    kept_prices = keep(prices, pcols["ticker"])
    record.update(
        applied=True,
        tweets_before=int(len(tweets)),
        tweets_after=int(len(kept_tweets)),
        tickers_after=int(kept_tweets[tcols["ticker"]].nunique()),
    )
    return kept_tweets, kept_prices, record
