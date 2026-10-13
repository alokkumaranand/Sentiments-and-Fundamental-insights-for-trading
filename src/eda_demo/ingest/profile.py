"""Dataset profiling — the Step 0 deliverable.

Produces `profile.json` (machine-readable) and `docs/quality_report.md` (human-readable)
and applies the blocking/warning split from `validate.py`.

Design decision embodied here: the profile is computed from the RAW string columns with no
implicit coercion, and it measures the things that actually change the design rather than a
generic column summary. On this dataset that means cross-ticker text fan-out (the same text
posted under several tickers), promo-shaped text, tweets that roll to the next session, and
whether the price series contain an unadjusted split. Those four facts decide whether any
downstream number means anything, so they are measured first and stated plainly.

Every value written to profile.json is a plain Python primitive, so the file is stable
across pandas/numpy versions and can be diffed between runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..prep.scope import apply_scope
from . import validate
from .loaders import read_raw_csv

# Matches a cashtag such as $TSLA or $BRK.B. Used only to count promo-shaped tweets.
_CASHTAG = re.compile(r"\$[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?")
# Matches the HTML entities actually present in this corpus (&amp; &gt; &lt; &#39; ...).
_ENTITY = re.compile(r"&(?:amp|gt|lt|quot|#\d+|#x[0-9A-Fa-f]+);")


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain Python, leaving strings alone.

    Without this, json.dump raises on numpy.int64 and the profile silently becomes
    unpublishable, or worse, a float that only round-trips in one numpy version.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def assign_tiers(own_text_share: dict[str, float], settings: Settings) -> dict[str, str]:
    """Map each ticker to coverage tier A, B or C from its own-text share.

    Tier A: the ticker's text is overwhelmingly its own, so per-ticker inference is allowed.
    Tier B: a mix; per-ticker numbers must restrict to own text and print the fan-out share.
    Tier C: every row is cross-posted, so the ticker has no text of its own to measure.
    Thresholds come from config/experiments.yaml, never from constants here.
    """
    tiers_cfg = settings.experiments.get("tiers", {})
    specific = float(tiers_cfg.get("specific_own_text_share", 0.80))
    mixed = float(tiers_cfg.get("mixed_own_text_share", 0.10))
    return {t: ("A" if s >= specific else "B" if s >= mixed else "C") for t, s in own_text_share.items()}


def _tweet_metrics(
    tweets: pd.DataFrame, cols: dict[str, str], settings: Settings, calendar: set[Any]
) -> dict[str, Any]:
    """Measure the tweet file: hygiene, text quality, and cross-ticker fan-out.

    `calendar` is the set of trading dates taken from the price file, so the
    non-trading-day share is measured against the real sessions rather than a guess.
    """
    date_col, text_col, ticker_col = cols["date"], cols["text"], cols["ticker"]
    text = tweets[text_col].fillna("")
    ticker = tweets[ticker_col]

    # Timestamps arrive with an explicit +00:00 offset (verified), so utc=True cannot
    # mis-read them. ET is used only for the session decision, never for storage.
    ts_utc = validate.parse_dates(tweets, date_col, "tweets")
    ts_et = ts_utc.dt.tz_convert(settings.market_timezone)
    et_date = ts_et.dt.normalize().dt.tz_localize(None)

    # A text posted under several tickers: the giveaway that a ticker's corpus is generic.
    per_text_tickers = tweets.assign(_t=text).groupby("_t")[ticker_col].nunique()
    shared_texts = per_text_tickers[per_text_tickers > 1]
    rows_shared = int(text.isin(set(shared_texts.index)).sum())

    grouped = tweets.assign(_t=text).groupby(ticker_col)
    own_counts = grouped.apply(lambda g: int((~g["_t"].isin(set(shared_texts.index))).sum()), include_groups=False)
    total_counts = grouped.size()
    own_share = (own_counts / total_counts).round(4)

    promo = text.str.contains("http", regex=False) | (text.str.count(_CASHTAG) >= 4)

    return {
        "rows": int(len(tweets)),
        "columns": list(tweets.columns),
        "null_cells": int(tweets.isna().sum().sum()),
        "date_min": ts_utc.min().isoformat(),
        "date_max": ts_utc.max().isoformat(),
        "distinct_tickers": int(ticker.nunique()),
        "tweets_per_ticker": {str(k): int(v) for k, v in total_counts.sort_values(ascending=False).items()},
        "duplicate_rows": int(tweets.duplicated().sum()),
        "duplicate_text_same_ticker": int(tweets.duplicated(subset=[ticker_col, text_col]).sum()),
        "distinct_texts": int(per_text_tickers.shape[0]),
        "shared_texts": int(shared_texts.shape[0]),
        "rows_in_shared_text": rows_shared,
        "rows_in_shared_text_share": round(rows_shared / len(tweets), 4),
        "own_text_count_per_ticker": {str(k): int(v) for k, v in own_counts.items()},
        "own_text_share_per_ticker": {str(k): float(v) for k, v in own_share.items()},
        "promo_rows": int(promo.sum()),
        "promo_share": round(float(promo.mean()), 4),
        "html_entity_rows": int(text.str.contains(_ENTITY).sum()),
        "url_rows": int(text.str.contains("http", regex=False).sum()),
        "cashtag_rows": int(text.str.contains(_CASHTAG).sum()),
        "hashtag_rows": int(text.str.contains("#", regex=False).sum()),
        "reply_rows": int(text.str.startswith("@").sum()),
        "non_trading_day_rows": int((~et_date.isin(calendar)).sum()),
        "non_trading_day_share": round(float((~et_date.isin(calendar)).mean()), 4),
        "at_or_after_close_rows": int((ts_et.dt.hour >= settings.close_hour).sum()),
        "at_or_after_close_share": round(float((ts_et.dt.hour >= settings.close_hour).mean()), 4),
        "length_chars": {
            "min": int(text.str.len().min()),
            "median": float(text.str.len().median()),
            "max": int(text.str.len().max()),
        },
    }


def _price_metrics(prices: pd.DataFrame, cols: dict[str, str], split_limit: float) -> dict[str, Any]:
    """Measure the price file, including the split-continuity check.

    Split continuity is checked here rather than in a test because it is a property of the
    *data*, not of the code: a future refresh could ship unadjusted prices, and a single
    unadjusted 20:1 split would corrupt every return, volatility and bucket statistic
    downstream. The check is cheap, so it runs on every profile.
    """
    date_col, ticker_col = cols["date"], cols["ticker"]
    dates = validate.parse_dates(prices, date_col, "prices")
    close = pd.to_numeric(prices[cols["close"]], errors="coerce")
    adj = pd.to_numeric(prices[cols["adj_close"]], errors="coerce")

    per_ticker_rows = prices.groupby(ticker_col).size()
    sessions = int(dates.nunique())
    tickers = int(prices[ticker_col].nunique())

    # A session is ragged when fewer than all tickers report it. Seen in the 1:9 dev
    # slice, where the split cut mid-session.
    per_session_tickers = prices.assign(_d=dates.dt.date).groupby("_d")[ticker_col].nunique()
    ragged = [str(d) for d, n in per_session_tickers.items() if n != tickers]

    # Largest single-day close move per ticker. A split left unadjusted would appear here
    # as roughly -0.5 (2:1), -0.67 (3:1) or -0.95 (20:1) and is impossible to miss.
    ordered = prices.assign(_d=dates, _c=close).sort_values([ticker_col, "_d"])
    returns = ordered.groupby(ticker_col)["_c"].pct_change()
    max_abs = returns.abs().groupby(ordered[ticker_col]).max()
    worst_ticker = str(max_abs.idxmax()) if len(max_abs) else None
    worst_move = float(max_abs.max()) if len(max_abs) else 0.0

    return {
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "null_cells": int(prices.isna().sum().sum()),
        "date_min": dates.min().isoformat(),
        "date_max": dates.max().isoformat(),
        "sessions": sessions,
        "distinct_tickers": tickers,
        "duplicate_ticker_date": int(prices.duplicated(subset=[ticker_col, date_col]).sum()),
        "weekend_rows": int((dates.dt.dayofweek >= 5).sum()),
        "rows_per_ticker": {
            "min": int(per_ticker_rows.min()),
            "max": int(per_ticker_rows.max()),
            "distinct_values": sorted({int(v) for v in per_ticker_rows.unique()}),
        },
        "ragged_sessions": ragged,
        "adj_equals_close_rows": int((adj == close).sum()),
        "max_abs_adj_minus_close": round(float((adj - close).abs().max()), 6),
        "split_check": {
            "max_abs_1d_return": round(worst_move, 6),
            "worst_ticker": worst_ticker,
            "limit": split_limit,
            "passed": worst_move < split_limit,
        },
        "close_parse_failures": int(close.isna().sum()),
    }


def _apply_scope(
    tweets: pd.DataFrame,
    prices: pd.DataFrame,
    tcols: dict[str, str],
    pcols: dict[str, str],
    scope: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Delegate to the single implementation in `prep/scope.py`.

    Kept as a one-line wrapper because Phase 0's reviewed behaviour must not change, while the
    rule itself must exist only once: if the profiler and the feature layer disagreed about the
    subset, the same count would mean two different things in two reports.
    """
    return apply_scope(tweets, prices, tcols, pcols, scope)


def build_profile(settings: Settings, enforce_split_check: bool = True) -> dict[str, Any]:
    """Load both inputs read-only and return the full profile as a JSON-safe dict.

    Blocking problems raise BlockingError before any metrics are computed where possible,
    so a broken input never produces a half-trustworthy profile. `enforce_split_check=False`
    downgrades the split test to a recorded warning, which is only useful when inspecting a
    dataset that is already known to be adjusted differently.
    """
    cfg_cols = settings.raw.get("columns", {})
    tcols, pcols = cfg_cols.get("tweets", {}), cfg_cols.get("prices", {})

    tweets, tweets_hashes = read_raw_csv(settings.tweets_path, settings.tweets_member)
    prices, prices_hashes = read_raw_csv(settings.prices_path, None)

    # Cheap, decisive checks first: a wrong column name or a bad date makes everything
    # downstream meaningless, so stop before spending time on metrics.
    validate.require_columns(tweets, tcols, settings.tweets_path.name)
    validate.require_columns(prices, pcols, settings.prices_path.name)

    # Manual subset, from config/experiments.yaml -> scope. Applied before any metric.
    tweets, prices, scope_record = _apply_scope(
        tweets, prices, tcols, pcols, settings.experiments.get("scope", {}) or {}
    )

    tweet_tickers = set(tweets[tcols["ticker"]].unique())
    price_tickers = set(prices[pcols["ticker"]].unique())
    validate.require_ticker_overlap(tweet_tickers, price_tickers)

    # Trading dates as tz-naive midnight Timestamps. Deliberately NOT datetime.date objects:
    # a pandas Timestamp is a datetime subclass, and `Timestamp == datetime.date` is False in
    # Python, so a date-keyed set would silently make every tweet look like a non-trading day.
    calendar = set(
        validate.parse_dates(prices, pcols["date"], "prices").dt.normalize().dt.tz_localize(None)
    )
    limit = float(settings.experiments.get("quality", {}).get("split_return_limit", 0.50))

    tmetrics = _tweet_metrics(tweets, tcols, settings, calendar)
    pmetrics = _price_metrics(prices, pcols, limit)

    own_share = tmetrics["own_text_share_per_ticker"]
    validate.require_own_text(sum(tmetrics["own_text_count_per_ticker"].values()))
    tiers = assign_tiers(own_share, settings)
    tier_counts = {t: sum(1 for v in tiers.values() if v == t) for t in ("A", "B", "C")}
    validate.require_tier_a(2)(tier_counts["A"])

    # Split continuity is a property of the DATA, not the code, so it is checked on every run
    # rather than only in a test: an unadjusted split would corrupt every downstream return.
    split = pmetrics["split_check"]
    if not split["passed"]:
        message = (
            f"price split check failed: |1-day return| = {split['max_abs_1d_return']:.3f} for "
            f"{split['worst_ticker']} >= {split['limit']}. This looks like an unadjusted split; "
            "returns cannot be trusted."
        )
        if enforce_split_check:
            raise validate.BlockingError(message)
        pmetrics.setdefault("notes", []).append(message)

    metrics: dict[str, Any] = {
        "dataset": settings.dataset_name,
        "scope": scope_record,
        "tweets_path": str(settings.tweets_path),
        "prices_path": str(settings.prices_path),
        "sessions": pmetrics["sessions"],
        "tweets": tmetrics,
        "prices": pmetrics,
        "tiers": {
            "thresholds": settings.experiments.get("tiers", {}),
            "by_ticker": tiers,
            "counts": tier_counts,
        },
        "reconcile": {
            "tickers_shared": len(tweet_tickers & price_tickers),
            "tickers_only_in_tweets": sorted(tweet_tickers - price_tickers),
            "tickers_only_in_prices": sorted(price_tickers - tweet_tickers),
        },
        "hashes": {"tweets": tweets_hashes, "prices": prices_hashes},
    }
    metrics["warnings"] = validate.collect_warnings(metrics)
    metrics["blocking"] = []
    metrics["data_version"] = _data_version(metrics, settings)
    return _jsonable(metrics)


def _data_version(metrics: dict[str, Any], settings: Settings) -> str:
    """Return a short id over the inputs and the experiment config.

    Any tool result in later phases carries this id, so a number can always be traced to
    the exact data and thresholds that produced it.
    """
    payload = json.dumps(
        {
            "hashes": metrics.get("hashes", {}),
            "experiments": settings.experiments,
            "dataset": settings.dataset_name,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def new_run_id(dataset: str) -> str:
    """Return a sortable run id: UTC timestamp plus the dataset name."""
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{dataset}"


def write_profile(profile: dict[str, Any], path: Path) -> None:
    """Write profile.json, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=False), encoding="utf-8")


def write_quality_report(profile: dict[str, Any], path: Path) -> None:
    """Write the human-readable Markdown quality report.

    Reads only from the profile dict, so it can regenerate an old run's report without
    re-reading the dataset.
    """
    t, p, tiers = profile["tweets"], profile["prices"], profile["tiers"]
    lines: list[str] = []
    add = lines.append

    add("# Data quality report")
    add("")
    add(f"- Dataset slice: **{profile['dataset']}**")
    scope = profile.get("scope") or {}
    if scope.get("applied"):
        add(
            f"- **Manual scope applied:** include={scope.get('include')} exclude={scope.get('exclude')} "
            f"-> {scope.get('tickers_after')} tickers, {scope.get('tweets_after'):,} of "
            f"{scope.get('tweets_before'):,} tweet rows kept. Every count below describes this subset."
        )
    add(f"- Tweets: `{profile['tweets_path']}`")
    add(f"- Prices: `{profile['prices_path']}`")
    add(f"- `data_version`: `{profile['data_version']}`")
    add("")
    add("## Inputs and hashes")
    add("")
    add("| File | SHA-256 |")
    add("|---|---|")
    for label, key in (("tweets", "tweets"), ("prices", "prices")):
        hashes = profile["hashes"][key]
        for hkey, hval in hashes.items():
            add(f"| {label} ({hkey}) | `{hval}` |")
    add("")

    add("## Tweets")
    add("")
    add(f"- {t['rows']:,} rows x {len(t['columns'])} cols; {t['null_cells']} null cells")
    add(f"- Window: {t['date_min']} -> {t['date_max']}")
    add(f"- {t['distinct_tickers']} tickers; {t['distinct_texts']:,} distinct texts")
    add(f"- Duplicate rows: {t['duplicate_rows']}; duplicate (ticker, text): {t['duplicate_text_same_ticker']}")
    add(
        f"- **Cross-ticker fan-out: {t['shared_texts']:,} texts appear under >1 ticker, "
        f"covering {t['rows_in_shared_text']:,} rows ({t['rows_in_shared_text_share']:.1%})**"
    )
    add(f"- Promo-shaped (URL or >=4 cashtags): {t['promo_rows']:,} ({t['promo_share']:.1%})")
    add(f"- HTML entities present in {t['html_entity_rows']:,} rows")
    add(f"- Replies (start with @): {t['reply_rows']:,}; cashtags: {t['cashtag_rows']:,}; hashtags: {t['hashtag_rows']:,}")
    add(f"- On a non-trading date: {t['non_trading_day_rows']:,} ({t['non_trading_day_share']:.1%}) -> rolls to the next session")
    add(f"- At/after the close hour: {t['at_or_after_close_rows']:,} ({t['at_or_after_close_share']:.1%}) -> rolls to the next session")
    add(f"- Text length (chars): min {t['length_chars']['min']}, median {t['length_chars']['median']}, max {t['length_chars']['max']}")
    add("")
    add("### Tweets per ticker, own-text share, tier")
    add("")
    add("| Ticker | Tweets | Own-text share | Tier |")
    add("|---|---:|---:|:--:|")
    for ticker, count in t["tweets_per_ticker"].items():
        share = t["own_text_share_per_ticker"][ticker]
        add(f"| {ticker} | {count:,} | {share:.2f} | {tiers['by_ticker'][ticker]} |")
    add("")

    add("## Prices")
    add("")
    add(f"- {p['rows']:,} rows x {len(p['columns'])} cols; {p['null_cells']} null cells")
    add(f"- Window: {p['date_min']} -> {p['date_max']} ({p['sessions']} sessions, {p['distinct_tickers']} tickers)")
    add(f"- Duplicate (ticker, date): {p['duplicate_ticker_date']}; weekend rows: {p['weekend_rows']}")
    add(f"- Rows per ticker: min {p['rows_per_ticker']['min']}, max {p['rows_per_ticker']['max']}, values {p['rows_per_ticker']['distinct_values']}")
    add(f"- Ragged sessions (not all tickers present): {len(p['ragged_sessions'])} {p['ragged_sessions'][:6]}")
    add(f"- `Adj Close` equals `Close` on {p['adj_equals_close_rows']:,} rows; max difference {p['max_abs_adj_minus_close']:.4f}")
    split = p["split_check"]
    verdict = "passed" if split["passed"] else "FAILED"
    add(
        f"- **Split continuity: {verdict}** — largest |1-day return| {split['max_abs_1d_return']:.4f} "
        f"({split['worst_ticker']}), limit {split['limit']}"
    )
    add("")

    add("## Reconciliation")
    add("")
    r = profile["reconcile"]
    add(f"- Tickers shared by both files: {r['tickers_shared']}")
    add(f"- Only in tweets: {r['tickers_only_in_tweets'] or 'none'}")
    add(f"- Only in prices: {r['tickers_only_in_prices'] or 'none'}")
    add("")

    add("## Tier summary")
    add("")
    add(f"- Thresholds: {tiers['thresholds']}")
    add(f"- Tier A (specific): {tiers['counts']['A']} | Tier B (mixed): {tiers['counts']['B']} | Tier C (generic): {tiers['counts']['C']}")
    add("")

    add("## Warnings")
    add("")
    if profile["warnings"]:
        for warning in profile["warnings"]:
            add(f"- {warning}")
    else:
        add("- none")
    add("")
    add("## Blocking errors")
    add("")
    add("- none" if not profile["blocking"] else "\n".join(f"- {b}" for b in profile["blocking"]))
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
