"""Canonical tables and session alignment.

Three decisions are embodied here, all conservative by design:

1. **The price file is the trading calendar.** It is a complete 25 x 252 rectangle, so the
   session dates come from the data. No exchange-calendar dependency, nothing to drift.
2. **A tweet belongs to the next session that can still trade on it.** Posted before 16:00 ET
   on a trading day -> that session; at/after the close, or on a non-trading date -> the next
   session. This can discard timing information but can never let a later tweet influence an
   earlier session, which is the failure that would silently invalidate every result.
3. **Nothing is dropped for being low quality.** Tweets that roll past the last session in the
   window get `session_date = NaT` and a `drop_reason`; they are counted and reported, and only
   the panel build excludes them.

Fan-out (`own_text_flag`, `shared_with_n`) is computed on the cleaned text, because
`28.5%` of rows in this corpus are text posted under more than one ticker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..ingest import validate
from ..ingest.loaders import read_raw_csv
from .clean import clean_tweets
from .scope import apply_scope


@dataclass
class Canonical:
    """The two canonical tables plus the calendar and a report of every decision made."""

    tweets: pd.DataFrame
    prices: pd.DataFrame
    calendar: list[pd.Timestamp]
    report: dict[str, Any]


def _column_map(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    """Return (tweet columns, price columns) from settings.yaml; fail if a section is missing."""
    cfg = settings.raw.get("columns", {})
    return cfg.get("tweets", {}), cfg.get("prices", {})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return `column` as float, or an all-NaN column when it is not present at all."""
    if column not in frame.columns:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def load_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Read both raw files read-only and apply the manual ticker scope.

    Returns (tweets, prices, scope_record, hashes). Used by both `features` and `induce`, so
    the subset rule cannot differ between them.
    """
    tcols, pcols = _column_map(settings)
    tweets, tweets_hashes = read_raw_csv(settings.tweets_path, settings.tweets_member)
    prices, prices_hashes = read_raw_csv(settings.prices_path, None)
    validate.require_columns(tweets, tcols, settings.tweets_path.name)
    validate.require_columns(prices, pcols, settings.prices_path.name)
    tweets, prices, scope_record = apply_scope(
        tweets, prices, tcols, pcols, settings.experiments.get("scope", {}) or {}
    )
    return tweets, prices, scope_record, {"tweets": tweets_hashes, "prices": prices_hashes}


def build_calendar(prices: pd.DataFrame, date_col: str) -> list[pd.Timestamp]:
    """Return the sorted, tz-naive session dates taken from the price file.

    Deliberately naive `Timestamp` objects, not `datetime.date`: `Timestamp == date` is False in
    Python, so a `date`-keyed set would silently make every tweet look like a non-trading day.
    """
    parsed = validate.parse_dates(prices, date_col, "prices").dt.normalize().dt.tz_localize(None)
    return sorted(set(parsed))


def align_sessions(
    ts_utc: pd.Series, calendar: list[pd.Timestamp], market_timezone: str, close_hour: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map each tweet timestamp to the session that can act on it.

    Returns (session_date, rolled_flag, drop_reason). `rolled_flag` marks the tweets that move
    to a later session (measured: ~49% of the corpus), and `drop_reason` is set only for tweets
    that roll past the end of the calendar, which cannot belong to any session in this window.
    """
    ts_et = ts_utc.dt.tz_convert(market_timezone)
    et_day = ts_et.dt.normalize().dt.tz_localize(None)
    before_close = ts_et.dt.hour < close_hour

    ordered = np.array(calendar, dtype="datetime64[ns]")
    days = et_day.to_numpy(dtype="datetime64[ns]")
    if len(ordered) == 0:
        # No calendar at all: every tweet is out of window. Returned rather than raised so the
        # caller reports it as a data problem with counts instead of a stack trace.
        session = pd.Series(pd.NaT, index=ts_utc.index)
        rolled = pd.Series(True, index=ts_utc.index)
        return session, rolled, pd.Series(["no trading calendar"], index=ts_utc.index, dtype=object)
    # side="right" => the first session strictly after this calendar day. A tweet on a trading
    # day before the close is handled by the `same_day` branch instead, so the two never mix.
    next_idx = np.searchsorted(ordered, days, side="right")
    same_day = et_day.isin(set(calendar)) & before_close
    has_next = next_idx < len(ordered)
    safe_idx = np.where(has_next, next_idx, 0)
    next_session = pd.Series(
        np.where(has_next, ordered[safe_idx], np.datetime64("NaT")), index=ts_utc.index
    )

    session = next_session.where(~same_day, et_day)
    rolled = ~same_day
    drop_reason = pd.Series([None] * len(ts_utc), index=ts_utc.index, dtype=object)
    drop_reason[~has_next & ~same_day] = "rolls past the last session in the window"
    return session, rolled, drop_reason


def fanout_flags(text_key: pd.Series, ticker: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (own_text_flag, shared_with_n) for each row.

    `shared_with_n` is the number of distinct tickers that carry this exact text. Own text means
    `shared_with_n == 1` — the only rows on which a per-ticker text measure is about that ticker
    rather than about the shared chatter pool.
    """
    per_text = pd.DataFrame({"key": text_key, "ticker": ticker}).groupby("key")["ticker"].nunique()
    shared_with_n = text_key.map(per_text).astype(int)
    return shared_with_n.eq(1), shared_with_n


def check_split_continuity(prices: pd.DataFrame, ticker_col: str, close_col: str, limit: float) -> dict[str, Any]:
    """Assert that no 1-day move looks like an unadjusted split, and return the measurement.

    Verified once in Step 0 and re-checked on every run because a single unadjusted split would
    corrupt every return, volatility and bucket statistic downstream — a wrong answer rather
    than an uncertain one, so it stops the run (SPEC.md 2.3, 2.6).
    """
    close = pd.to_numeric(prices[close_col], errors="coerce")
    ordered = prices.assign(_c=close).sort_values([ticker_col, "date"])
    returns = ordered.groupby(ticker_col)["_c"].pct_change().abs()
    worst = float(returns.max()) if len(returns) else 0.0
    worst_ticker = str(ordered.loc[returns.idxmax(), ticker_col]) if returns.notna().any() else None
    result = {
        "max_abs_1d_return": round(worst, 6),
        "worst_ticker": worst_ticker,
        "limit": limit,
        "passed": worst < limit,
    }
    if not result["passed"]:
        raise validate.BlockingError(
            f"price split check failed: |1-day return| = {worst:.3f} for {worst_ticker} >= {limit}. "
            "This looks like an unadjusted split; returns cannot be trusted."
        )
    return result


def session_split(calendar: list[pd.Timestamp], fraction: float) -> dict[str, Any]:
    """Cut the session calendar into a chronological discovery part and a holdout part.

    A fraction rather than a date literal, so the split stays correct if the window changes.
    The boundary is stated in sessions and in dates, because "70/30" alone hides the sample.
    """
    cut = max(1, min(len(calendar) - 1, int(round(len(calendar) * fraction)))) if len(calendar) > 1 else len(calendar)
    discovery = calendar[:cut]
    holdout = calendar[cut:]
    return {
        "fraction": fraction,
        "discovery_sessions": len(discovery),
        "holdout_sessions": len(holdout),
        "discovery_start": str(discovery[0].date()) if discovery else None,
        "discovery_end": str(discovery[-1].date()) if discovery else None,
        "holdout_start": str(holdout[0].date()) if holdout else None,
        "holdout_end": str(holdout[-1].date()) if holdout else None,
    }


def _canonical_warnings(report: dict[str, Any]) -> list[str]:
    """Turn the measured canonical counts into plain warnings, worst first.

    These are the same facts as the profiler's warnings, re-measured after cleaning so the
    inductor and the manifest are told about the text that is actually in use. A check is only
    blocking when continuing would produce a *wrong* answer; all of these are uncertain, so they
    travel as warnings and are never silent drops.
    """
    warnings: list[str] = []
    rows = max(1, report["rows_in_window"])
    promo_share = report["clean"]["promo_rows"] / max(1, report["clean"]["rows"])
    shared_rows = rows - report["own_text_rows"]
    if promo_share > 0.20:
        warnings.append(f"{promo_share:.1%} of tweets are promo-shaped (URL or >=4 cashtags).")
    if report["rolled_share"] > 0.10:
        warnings.append(
            f"{report['rolled_share']:.1%} of tweets roll forward to the next session "
            "(posted at/after the close or on a non-trading date)."
        )
    if shared_rows / rows > 0.20:
        warnings.append(
            f"{shared_rows / rows:.1%} of in-window rows carry text that also appears under "
            "another ticker; per-ticker text measures are only about the ticker on own text."
        )
    if report["rows_out_of_window"]:
        warnings.append(
            f"{report['rows_out_of_window']:,} rows fall outside the last session of the window "
            "and are excluded from the panel (counted, never dropped silently)."
        )
    if report["sessions"] < 60:
        warnings.append(
            f"only {report['sessions']} trading session(s): trailing-window features and the "
            "discovery/holdout split are not meaningful. Use --dataset full."
        )
    return warnings


def build_canonical(settings: Settings) -> Canonical:
    """Load, clean and align the inputs into the two canonical tables.

    Returns a `Canonical` whose `report` records every row-count and decision, so the run
    manifest and the terminal overview can state what was kept, flagged and excluded.
    """
    tcols, pcols = _column_map(settings)
    tweets_raw, prices_raw, scope_record, hashes = load_inputs(settings)

    cashtag_min = int(settings.experiments.get("quality", {}).get("promo_min_cashtags", 4))
    tweets, clean_report = clean_tweets(tweets_raw, tcols["text"], cashtag_min)

    ts_utc = validate.parse_dates(tweets, tcols["date"], "tweets")
    tweets["ts_utc"] = ts_utc
    tweets["ts_original"] = tweets[tcols["date"]]

    prices = pd.DataFrame(
        {
            "ticker": prices_raw[pcols["ticker"]],
            "date": validate.parse_dates(prices_raw, pcols["date"], "prices")
            .dt.normalize()
            .dt.tz_localize(None),
            # OHLC is optional: settings.yaml maps only what the pipeline needs, and a missing
            # OHLC column becomes NaN rather than an error, because no feature depends on it.
            "open": _numeric(prices_raw, pcols.get("open", "Open")),
            "high": _numeric(prices_raw, pcols.get("high", "High")),
            "low": _numeric(prices_raw, pcols.get("low", "Low")),
            "close": pd.to_numeric(prices_raw[pcols["close"]], errors="coerce"),
            "adj_close": pd.to_numeric(prices_raw[pcols["adj_close"]], errors="coerce"),
            "volume": pd.to_numeric(prices_raw[pcols["volume"]], errors="coerce"),
        }
    ).sort_values(["ticker", "date"]).reset_index(drop=True)

    calendar = build_calendar(prices, "date")
    limit = float(settings.experiments.get("quality", {}).get("split_return_limit", 0.50))
    split_check = check_split_continuity(prices, "ticker", "close", limit)

    session, rolled, drop_reason = align_sessions(
        tweets["ts_utc"], calendar, settings.market_timezone, settings.close_hour
    )
    tweets["session_date"] = session
    tweets["rolled_flag"] = rolled
    tweets["drop_reason"] = drop_reason

    tweets["ticker"] = tweets[tcols["ticker"]]
    tweets["text_key"] = tweets["text_clean"]
    own_text_flag, shared_with_n = fanout_flags(tweets["text_key"], tweets["ticker"])
    tweets["own_text_flag"] = own_text_flag
    tweets["shared_with_n"] = shared_with_n

    in_window = tweets["session_date"].notna()
    report: dict[str, Any] = {
        "scope": scope_record,
        "hashes": hashes,
        "clean": clean_report,
        "rows_in": int(len(tweets)),
        "rows_in_window": int(in_window.sum()),
        "rows_out_of_window": int((~in_window).sum()),
        "out_of_window_reasons": {
            str(reason): int(count)
            for reason, count in tweets.loc[~in_window, "drop_reason"].value_counts().items()
        },
        "rolled_rows": int(rolled.sum()),
        "rolled_share": round(float(rolled.mean()), 4),
        "sessions": len(calendar),
        "first_session": str(calendar[0].date()),
        "last_session": str(calendar[-1].date()),
        "distinct_texts": int(tweets["text_key"].nunique()),
        "shared_texts": int((tweets.groupby("text_key")["ticker"].nunique() > 1).sum()),
        "own_text_rows": int(own_text_flag.sum()),
        "own_text_share_per_ticker": {
            str(k): round(float(v), 4)
            for k, v in tweets.groupby("ticker")["own_text_flag"].mean().items()
        },
        "split": session_split(calendar, float(settings.experiments.get("split", {}).get("discovery_fraction", 0.70))),
        "split_check": split_check,
    }
    report["warnings"] = _canonical_warnings(report)
    return Canonical(tweets=tweets, prices=prices, calendar=calendar, report=report)
