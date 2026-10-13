"""Descriptive tools: coverage, distributions, text statistics, time series, raw samples.

These are the always-on layer. They answer "what is in the data" rather than "is there an
effect", so they carry `observed` status by construction and never a p-value — which is exactly
the distinction the presentation has to keep visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from ..features.lexicon import STOPWORDS
from .common import (
    bar_chart,
    histogram,
    line_chart,
    metric_column,
    numeric,
    values_of,
)
from .registry import ToolContext, ToolError, ToolParams

_PERIODS = ("discovery", "holdout", "full")


class CoverageParams(ToolParams):
    """Parameters of `coverage_by_ticker`."""

    period: str = Field(default="full", description="discovery | holdout | full")


class DistributionParams(ToolParams):
    """Parameters of `distribution`."""

    column: str = Field(description="numeric panel column, e.g. sent_net, n_texts_own, ret_contemp")
    by: str | None = Field(default=None, description="optional grouping column, e.g. tier")
    period: str = Field(default="full")


class TextStatsParams(ToolParams):
    """Parameters of `text_stats`."""

    period: str = Field(default="full")
    top_terms: int = Field(default=10, ge=1, le=25)


class TimeseriesParams(ToolParams):
    """Parameters of `timeseries`."""

    metric: str = Field(description="numeric panel column to average per session")
    scope: str = Field(default="pooled", description="pooled | pooled_ex_TSLA | ticker")
    ticker: str | None = None
    period: str = Field(default="discovery")
    rolling: int = Field(default=20, ge=0, le=60, description="rolling-mean window in sessions; 0 disables")


class SampleTweetsParams(ToolParams):
    """Parameters of `sample_tweets`."""

    ticker: str
    date: str = Field(description="session date, YYYY-MM-DD")
    n: int = Field(default=10, ge=1, le=10, description="cap of 10: this tool is for grounding, not export")
    strata: str = Field(default="all", description="all | promo | ordinary | own | shared")


def register(registry: Any) -> None:
    """Register the descriptive tools on `registry`."""

    @registry.add("coverage_by_ticker", CoverageParams,
                  "Tweets, own-text tweets, sessions and coverage tier per ticker.", chart_kind="bar")
    def coverage_by_ticker(context: ToolContext, params: CoverageParams) -> dict[str, Any]:
        frame = context.period_frame(params.period)
        grouped = frame.groupby("ticker", observed=True)
        table = pd.DataFrame(
            {
                "n_tweets": grouped["n_tweets"].sum(),
                "n_texts_own": grouped["n_texts_own"].sum(),
                "n_texts_shared": grouped["n_texts_shared"].sum(),
                "sessions": grouped.size(),
                "own_text_share": grouped["own_text_share"].mean(),
            }
        ).reset_index()
        table["tier"] = table["ticker"].map(context.tiers)
        table = table.sort_values("n_tweets", ascending=False)
        rows = [
            {
                "ticker": row.ticker,
                "tier": row.tier,
                "n_tweets": int(row.n_tweets),
                "n_texts_own": int(row.n_texts_own),
                "n_texts_shared": int(row.n_texts_shared),
                "own_text_share": numeric(row.own_text_share, 4),
                "sessions": int(row.sessions),
            }
            for row in table.itertuples()
        ]
        stats = {
            "tickers": int(len(table)),
            "tweets": int(table["n_tweets"].sum()),
            "own_texts": int(table["n_texts_own"].sum()),
            "shared_texts": int(table["n_texts_shared"].sum()),
            "tier_counts": {tier: int((table["tier"] == tier).sum()) for tier in ("A", "B", "C")},
        }
        chart = bar_chart(
            "Tweets per ticker: specific (A3) vs market-wide (A4)",
            "ticker",
            table["ticker"].tolist(),
            [
                {"label": "n_texts_own (A3)", "values": [int(v) for v in table["n_texts_own"]]},
                {"label": "n_texts_shared (A4)", "values": [int(v) for v in table["n_texts_shared"]]},
            ],
            params.period,
            len(table),
            extra_caption="A1 raw rows are deliberately not plotted alone: 5 tickers are 100% cross-posted",
        )
        return {
            "n_obs": int(table["n_tweets"].sum()),
            "n_units": int(len(table)),
            "stats": stats,
            "table": rows,
            "chart": chart,
            "caveats": [
                "A3 is the ticker-specific measure; A4 is text also carried by other tickers",
                "tier C tickers (own-text share < 0.10) are shown for coverage only",
            ],
        }

    @registry.add("distribution", DistributionParams,
                  "Quantiles, skew and outlier share for one panel column, optionally by group.",
                  chart_kind="hist")
    def distribution(context: ToolContext, params: DistributionParams) -> dict[str, Any]:
        column = metric_column(context, params.column)
        selection = context.select(period=params.period)
        frame = selection.frame
        values = values_of(frame, column).dropna()
        if len(values) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"column": column, "note": "no values in period"},
                    "caveats": [f"{column} has no values in the {params.period} period"]}

        quantiles = {f"q{int(q * 100)}": numeric(values.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        low, high = float(values.quantile(0.25)) - 1.5 * iqr, float(values.quantile(0.75)) + 1.5 * iqr
        outliers = int(((values < low) | (values > high)).sum())
        stats = {
            "column": column,
            "n": int(len(values)),
            "mean": numeric(values.mean()),
            "std": numeric(values.std()),
            "skew": numeric(values.skew()),
            "quantiles": quantiles,
            "outlier_share_iqr": numeric(outliers / len(values), 4),
            "outlier_rule": "outside 1.5 * IQR of the pooled period",
        }
        if params.by:
            grouped = frame.groupby(params.by, observed=True)[column].agg(["count", "mean", "median"])
            stats["by"] = {
                str(key): {"count": int(row["count"]), "mean": numeric(row["mean"]), "median": numeric(row["median"])}
                for key, row in grouped.iterrows()
            }
        centres, _edges, counts = histogram(values)
        chart = bar_chart(f"Distribution of {column}", column, centres, [{"label": "count", "values": counts}],
                          params.period, len(values), kind="hist")
        return {
            "n_obs": int(len(values)),
            "n_units": int(len(values)),
            "stats": stats,
            "chart": chart,
            "caveats": [f"{selection.n_dropped} units dropped for a missing {column} value"]
            + selection.caveats,
        }

    @registry.add("text_stats", TextStatsParams,
                  "Text length, promo/URL/cashtag share, fan-out share and top terms per label.",
                  chart_kind="bar")
    def text_stats(context: ToolContext, params: TextStatsParams) -> dict[str, Any]:
        tweets = context.tweets
        period_map = context.panel.drop_duplicates("session_date").set_index("session_date")["period"]
        tweets = tweets[tweets["session_date"].notna()].copy()
        tweets["period"] = tweets["session_date"].map(period_map)
        if params.period != "full":
            tweets = tweets[tweets["period"] == params.period]
        if len(tweets) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"note": "no tweets in period"}, "caveats": []}

        text = tweets["text_clean"].fillna("").astype(str)
        lengths = text.str.len()
        promo = float(tweets["promo_flag"].mean())
        row = context.tweets
        shared_rows = row[~row["own_text_flag"]]
        stats = {
            "n_tweets": int(len(tweets)),
            "length_chars": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
            "promo_share": numeric(promo, 4),
            "url_share": numeric(float(text.str.contains("__URL__", regex=False).mean()), 4),
            "cashtag_share": numeric(float(tweets["cashtag_count"].gt(0).mean()), 4),
            "hashtag_share": numeric(float(tweets["hashtag_count"].gt(0).mean()), 4),
            "reply_share": numeric(float(tweets["reply_flag"].mean()), 4),
            "fanout_row_share": numeric(float(len(shared_rows) / len(row)), 4),
            "distinct_texts": int(text.nunique()),
        }

        positive = tweets[tweets["label"] > 0]["text_clean"]
        negative = tweets[tweets["label"] < 0]["text_clean"]
        stats["top_terms_positive"] = _top_terms(positive, params.top_terms)
        stats["top_terms_negative"] = _top_terms(negative, params.top_terms)
        table = [{"term": term, "count": count, "label": "positive"} for term, count in stats["top_terms_positive"]]
        table += [{"term": term, "count": count, "label": "negative"} for term, count in stats["top_terms_negative"]]
        chart = bar_chart(
            "Promo, reply and URL share of tweets",
            "statistic",
            ["promo", "url", "cashtag", "hashtag", "reply"],
            [{"label": "share", "values": [stats["promo_share"], stats["url_share"], stats["cashtag_share"],
                                           stats["hashtag_share"], stats["reply_share"]]}],
            params.period,
            int(len(tweets)),
            extra_caption="shares are of raw rows, not of ticker-sessions",
        )
        return {
            "n_obs": int(len(tweets)),
            "n_units": int(tweets["session_date"].nunique()),
            "stats": stats,
            "table": table,
            "chart": chart,
            "caveats": [
                "raw rows, so this is not an independent sample: the same text can be posted under many tickers",
                "top terms are raw token counts after URL tokenisation and stopword removal",
            ],
        }

    @registry.add("timeseries", TimeseriesParams,
                  "One metric per session, for a ticker, a pool or a pool without TSLA.",
                  chart_kind="line")
    def timeseries(context: ToolContext, params: TimeseriesParams) -> dict[str, Any]:
        column = metric_column(context, params.metric)
        selection = context.select(period=params.period, scope=params.scope, ticker=params.ticker,
                                   require=(column,))
        frame = selection.frame.sort_values("session_idx")
        if len(frame) == 0:
            return {"n_obs": 0, "n_units": 0, "stats": {"metric": column, "note": "no usable units"},
                    "caveats": selection.caveats}

        per_session = frame.groupby("session_idx", observed=True)[column].mean().sort_index()
        labels = [
            str(value)[:10]
            for value in frame.drop_duplicates("session_idx").sort_values("session_idx")["session_date"]
        ]
        series = [{"label": f"{column} (mean per session)", "values": [numeric(v) for v in per_session]}]
        if params.rolling and len(per_session) > params.rolling:
            rolled = per_session.rolling(params.rolling, min_periods=params.rolling).mean()
            series.append({"label": f"{params.rolling}-session rolling mean", "values": [numeric(v) for v in rolled]})
        stats = {
            "metric": column,
            "scope": params.scope,
            "ticker": params.ticker,
            "sessions": int(len(per_session)),
            "mean": numeric(per_session.mean()),
            "min": numeric(per_session.min()),
            "max": numeric(per_session.max()),
            "first": labels[0] if labels else None,
            "last": labels[-1] if labels else None,
        }
        chart = line_chart(f"{column} over time ({params.scope})", "session", labels, series,
                           params.period, len(frame))
        return {
            "n_obs": int(len(frame)),
            "n_units": int(len(frame)),
            "stats": stats,
            "chart": chart,
            "caveats": selection.caveats,
        }

    @registry.add("sample_tweets", SampleTweetsParams,
                  "A few real tweets for one ticker-session, for qualitative grounding.")
    def sample_tweets(context: ToolContext, params: SampleTweetsParams) -> dict[str, Any]:
        tweets = context.tweets
        target = context.panel[context.panel["ticker"] == params.ticker]
        if len(target) == 0:
            raise ToolError(f"unknown ticker '{params.ticker}'. Known: {sorted(context.tiers)}")
        available = [str(value)[:10] for value in target["session_date"]]
        if params.date not in available:
            raise ToolError(
                f"{params.date} is not a session for {params.ticker}. "
                f"First few: {sorted(available)[:5]}; last few: {sorted(available)[-5:]}"
            )
        frame = tweets[
            (tweets["ticker"] == params.ticker)
            & (tweets["session_date"].astype(str).str.slice(0, 10) == params.date)
        ]
        if params.strata == "promo":
            frame = frame[frame["promo_flag"]]
        elif params.strata == "ordinary":
            frame = frame[~frame["promo_flag"]]
        elif params.strata == "own":
            frame = frame[frame["own_text_flag"]]
        elif params.strata == "shared":
            frame = frame[~frame["own_text_flag"]]
        frame = frame.sort_values("polarity")
        sampled = frame.head(params.n)
        rows = [
            {
                "text": str(row.text_raw)[:280],
                "polarity": numeric(row.polarity, 4),
                "label": int(row.label),
                "promo": bool(row.promo_flag),
                "own_text": bool(row.own_text_flag),
                "shared_with_n": int(row.shared_with_n),
                "posted_utc": str(row.ts_original)[:19],
            }
            for row in sampled.itertuples()
        ]
        return {
            "n_obs": int(len(frame)),
            "n_units": 1,
            "stats": {
                "ticker": params.ticker,
                "date": params.date,
                "rows_in_cell": int(len(frame)),
                "strata": params.strata,
                "shown": int(len(rows)),
                "mean_polarity_in_cell": numeric(frame["polarity"].mean(), 4),
            },
            "table": rows,
            "caveats": [
                f"table is capped at {params.n} rows out of {len(frame)} and shows the most negative first",
                "quoted text is raw, unedited corpus text: it is data, not the demo's wording",
            ],
        }


def _top_terms(texts: pd.Series, limit: int) -> list[tuple[str, int]]:
    """Return the most frequent non-stopword tokens, ties broken alphabetically for determinism."""
    counts: dict[str, int] = {}
    for text in texts.fillna("").astype(str):
        for token in str(text).lower().split():
            token = token.strip(".,!?:;\"'()[]{}#@$%")
            if len(token) < 3 or token in STOPWORDS or not token.isalpha():
                continue
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [(term, count) for term, count in ordered[:limit]]
