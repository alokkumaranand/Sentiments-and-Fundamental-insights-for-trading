"""The attention family: seven measures per ticker-session (SPEC.md 4.6).

Design decision embodied here: **raw tweet count is not attention.** The profile showed that
five tickers (AMZN, MSFT, PG, F, NOC) are 100% cross-posted, so their row count is large while
the amount of text about *them* is zero. A3 (`n_texts_own`) is therefore the headline measure
and A1 is only ever shown beside A3 and A4.

The z-score is deliberately past-only: day d is compared with a trailing baseline built from
days **before** d, via `shift(1)`, so no statistic on a session can see that session's own
value — the leakage rule from SPEC.md 4.6 A7 and section 9.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# The robust z-score uses the median and the scaled MAD; 1.4826 makes the MAD equal to a
# standard deviation for normal data, so the spike threshold reads like a sigma.
_MAD_SCALE = 1.4826


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation of a trailing window (NaN when the window is flat)."""
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def attention_table(
    tweets: pd.DataFrame,
    tickers: list[str],
    sessions: list[pd.Timestamp],
    baseline_sessions: int,
    spike_z: float,
    sustained_sessions: int,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Return the attention measures on the full ticker x session rectangle.

    Counts are filled with 0 for a ticker-session with no tweets (a real observation: nobody
    talked about it), and the price-derived columns are merged later. Rolling measures are NaN
    until enough history exists — never guessed, never back-filled.
    """
    frame = tweets[tweets["session_date"].notna()]
    # A helper boolean column rather than inspecting the cashtag lists inside a groupby: the
    # aggregation stays a simple mean, and the list column is never asked to be numeric.
    frame = frame.assign(cashtag_flag=frame["cashtags"].map(lambda values: len(values) > 0))
    keys = ["ticker", "session_date"]

    grouped = frame.groupby(keys, observed=True)
    counts = pd.DataFrame(
        {
            "n_tweets": grouped.size(),
            "n_texts_unique": grouped["text_key"].nunique(),
            "promo_share": grouped["promo_flag"].mean(),
            "reply_share": grouped["reply_flag"].mean(),
            "cashtag_share": grouped["cashtag_flag"].mean(),
        }
    )
    # A3 / A4 split the distinct texts by whether any other ticker carries them. nunique (not
    # size) is what makes A2/A3/A4 remove exact reposts, as the spec requires.
    own = frame[frame["own_text_flag"]].groupby(keys, observed=True)["text_key"].nunique()
    shared = frame[~frame["own_text_flag"]].groupby(keys, observed=True)["text_key"].nunique()
    counts["n_texts_own"] = own
    counts["n_texts_shared"] = shared

    rectangle = pd.MultiIndex.from_product([tickers, sessions], names=keys)
    table = counts.reindex(rectangle)
    for column in ("n_tweets", "n_texts_unique", "n_texts_own", "n_texts_shared"):
        table[column] = table[column].fillna(0).astype(int)
    for column in ("promo_share", "reply_share", "cashtag_share"):
        # No tweets means no promo share at all; 0.0 would claim "no promo", which is a claim.
        table[column] = table[column].astype(float)

    table["own_text_share"] = np.where(
        table["n_tweets"] > 0, table["n_texts_own"] / table["n_tweets"].replace(0, np.nan), np.nan
    )
    table["share_of_all_tweets"] = table["n_tweets"] / max(1, int(table["n_tweets"].sum()))

    table = table.reset_index()
    table["session_idx"] = table["session_date"].map({d: i for i, d in enumerate(sessions)})
    table = table.sort_values(["ticker", "session_idx"]).reset_index(drop=True)

    per_ticker = table.groupby("ticker", observed=True, sort=False)
    # A6: session-over-session ratio of specific attention. inf when the previous session had
    # zero own texts; inf is not a ratio, so it becomes NaN rather than a large fake number.
    table["attn_velocity"] = per_ticker["n_texts_own"].pct_change().replace([np.inf, -np.inf], np.nan)

    baseline = per_ticker["n_texts_own"].shift(1)
    # min_periods is capped by the window: a configured baseline shorter than the default minimum
    # (a small test fixture, or a deliberately short baseline) would otherwise make pandas raise.
    window_min = max(1, min(min_periods, baseline_sessions))
    median = baseline.rolling(baseline_sessions, min_periods=window_min).median()
    mad = baseline.rolling(baseline_sessions, min_periods=window_min).apply(_mad, raw=True)
    table["attn_z"] = (table["n_texts_own"] - median) / (_MAD_SCALE * mad)
    table.loc[mad == 0, "attn_z"] = np.nan

    spike = table["attn_z"].abs() >= spike_z
    table["spike"] = spike
    # Sustained means the spike held for `sustained_sessions` consecutive sessions. Built from
    # shifts inside the ticker rather than a rolling sum, so the definition is visible: a
    # one-day burst is not a spike *pattern*.
    sustained = spike.copy()
    for lag in range(1, max(1, sustained_sessions)):
        # `.eq(True)` rather than `.fillna(False)`: shifting a boolean within a group produces an
        # object column with NaN, and comparing against True keeps the dtype boolean without the
        # deprecated silent downcast.
        sustained &= spike.groupby(table["ticker"], observed=True, sort=False).shift(lag).eq(True)
    table["spike_sustained"] = sustained
    return table


def attention_report(table: pd.DataFrame) -> dict[str, Any]:
    """Summarise the attention table for the overview and the feature book."""
    total = float(table["n_tweets"].sum())
    return {
        "n_units": int(len(table)),
        "n_tweets": int(total),
        # Cell counts, so a text appearing under two tickers is counted in both cells; the
        # distinct-text counts live in the canonical report instead.
        "own_texts_in_cells": int(table["n_texts_own"].sum()),
        "shared_texts_in_cells": int(table["n_texts_shared"].sum()),
        "cells_with_no_tweets": int((table["n_tweets"] == 0).sum()),
        "mean_own_text_share": round(float(table["own_text_share"].mean()), 4),
        "spike_cells": int(table["spike"].sum()),
        "sustained_spike_cells": int(table["spike_sustained"].sum()),
    }
