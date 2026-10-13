"""Text cleaning for the canonical tweet table.

Design decision embodied here: cleaning never removes a row. URLs become a `__URL__` token
rather than disappearing, so promo detection still works after cleaning, and HTML entities are
unescaped rather than dropped, because the profile measured 7,745 such rows and a silently
smaller corpus would change every count. Each transformation records how many rows it touched.

Cleaning is vectorised with pandas string ops: 80,793 Python-level regex passes are slow enough
to be felt on every run, and the rows are small strings where pandas is the right tool.
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

# A URL in this corpus is always a t.co link or a full link; the trailing-token match is
# deliberately greedy on non-space, which is what a URL looks like in a tweet.
URL = re.compile(r"https?://\S+|www\.\S+")
# Matches $TSLA and $BRK.B — the same shape the profiler counts, kept identical on purpose so
# the promo numbers in the quality report and in this table agree.
CASHTAG = re.compile(r"\$[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?")
HASHTAG = re.compile(r"#\w+")
WHITESPACE = re.compile(r"\s+")


def clean_text(raw: pd.Series) -> pd.Series:
    """Return the cleaned text series: entities unescaped, URLs tokenised, whitespace collapsed.

    Order matters. Unescaping first means `&amp;` becomes `&` before tokenising, so a link
    written as `...&amp;id=1` still matches the URL pattern as one token.
    """
    text = raw.fillna("").astype(str)
    needs_unescape = text.str.contains("&", regex=False)
    if bool(needs_unescape.any()):
        text = text.where(~needs_unescape, text.map(html.unescape))
    text = text.str.replace(URL, "__URL__", regex=True)
    return text.str.replace(WHITESPACE, " ", regex=True).str.strip()


def marker_lists(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (cashtags, hashtags) as lists per row, preserving order and duplicates.

    Lists rather than counts because later stages want both `len()` and membership, and
    re-extracting from the cleaned text at each call site would be duplication.
    """
    cashtags = text.map(lambda value: CASHTAG.findall(value))
    hashtags = text.map(lambda value: HASHTAG.findall(value))
    return cashtags, hashtags


def clean_tweets(tweets: pd.DataFrame, text_col: str, cashtag_min: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add the cleaning columns to a tweet frame and return (frame, report).

    Added columns: `text_clean`, `cashtags`, `hashtags`, `url_flag`, `promo_flag`, `reply_flag`.
    `promo_flag` is URL present **or** >= `cashtag_min` cashtags, exactly as the profiler
    defines it, so the two stages cannot disagree about what a promo is.
    """
    raw = tweets[text_col].fillna("").astype(str)
    cleaned = clean_text(raw)
    cashtags, hashtags = marker_lists(cleaned)
    url_flag = cleaned.str.contains("__URL__", regex=False)
    cashtag_count = cashtags.map(len)

    frame = tweets.copy()
    frame["text_raw"] = raw
    frame["text_clean"] = cleaned
    frame["cashtags"] = cashtags
    frame["hashtags"] = hashtags
    frame["url_flag"] = url_flag
    frame["promo_flag"] = url_flag | (cashtag_count >= cashtag_min)
    frame["reply_flag"] = cleaned.str.startswith("@")

    report = {
        "rows": int(len(frame)),
        # "rows" is also the "after" count: this stage never drops, it only rewrites.
        "rows_with_html_entities": int(raw.str.contains("&", regex=False).sum()),
        "rows_with_url": int(url_flag.sum()),
        "rows_emptied_by_cleaning": int((cleaned.str.len() == 0).sum()),
        "promo_rows": int(frame["promo_flag"].sum()),
        "promo_min_cashtags": cashtag_min,
        "cashtag_rows": int((cashtag_count > 0).sum()),
        "hashtag_rows": int((hashtags.map(len) > 0).sum()),
        "reply_rows": int(frame["reply_flag"].sum()),
    }
    return frame, report
