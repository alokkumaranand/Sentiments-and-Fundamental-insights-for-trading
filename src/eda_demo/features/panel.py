"""The daily panel: one row per ticker-session, the single input every tool reads.

Design decision embodied here: **the unit of analysis is the ticker-day, not the tweet.** This
corpus has 80,793 tweets but they are not 80,793 independent observations — the same text is
posted under many tickers, days are autocorrelated, and half of all ticker-days carry fewer than
five tweets. Rolling everything up to the ticker-session and reporting `n_units` beside every
statistic is the honest version of that (SPEC.md 2.4).

The seven sentiment measures are **deductions on top of the per-tweet scores**, not the raw
scores themselves: net polarity, the positive/negative share pair, intensity, dispersion plus
agreement, the finance directional ratio, hedging share and attribution share. Quality gates
(`min_tweets_per_day`, own-text-only per tier) are applied here, once, and the counts they
exclude travel with the panel so no tool has to re-derive or silently drop anything.

This module also owns the panel `data_version`, which is what every `ToolResult` carries:
data hash + experiment config + lexicon id + code version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, Settings
from ..ingest.profile import assign_tiers
from ..version import CODE_VERSION
from . import price as price_features_module
from .attention import attention_report, attention_table
from .lexicon import FrozenLexicon, resolve_lexicon
from .sentiment import score_tweets

# Bump when the set of columns written to (or read from) the panel/tweet cache changes, so an old
# cache file can never be served to code that expects a column it does not contain. Also bumped for
# a change in how the cache is written (v2 added columns, v3 writes floats at full precision).
CACHE_SCHEMA = 3


@dataclass
class Panel:
    """Everything downstream stages read: the panel, the scored tweets, tiers and metadata."""

    panel: pd.DataFrame
    tweets: pd.DataFrame
    prices: pd.DataFrame
    tiers: dict[str, str]
    calendar: list[pd.Timestamp]
    data_version: str
    meta: dict[str, Any]

    def tickers(self, tier: str | None = None) -> list[str]:
        """Return tickers, optionally only those in one coverage tier (sorted for determinism)."""
        names = [t for t in self.panel["ticker"].unique() if tier is None or self.tiers.get(t) == tier]
        return sorted(names)


def _aggregate_sentiment(scored: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 sentiment measures per ticker-session on the gate-approved rows.

    `gates` carries, per ticker-session, whether own-text-only applies and how many tweets are
    needed. Rows below the minimum still appear in the panel with `low_coverage=True` and NaN
    sentiment, so a reader sees the excluded day rather than a gap.
    """
    keys = ["ticker", "session_date"]
    frame = scored.merge(gates, on=keys, how="inner")
    frame["used"] = np.where(
        frame["own_text_only"], frame["own_text_flag"], True
    )
    used = frame[frame["used"]]
    grouped = used.groupby(keys, observed=True)

    labels = used.assign(
        is_pos=(used["label"] > 0).astype(float),
        is_neg=(used["label"] < 0).astype(float),
    )
    label_groups = labels.groupby(keys, observed=True)

    table = pd.DataFrame(
        {
            "n_sent_tweets": grouped.size(),
            "sent_net": grouped["polarity"].mean(),
            "sent_intensity": grouped["abs_polarity"].mean(),
            "sent_dispersion": grouped["polarity"].std(ddof=0),
            "fin_directional": grouped["fin_directional"].mean(),
            "sent_uncertainty": grouped["uncertainty_share"].mean(),
            "sent_sarcasm_share": grouped["sarcasm_flag"].mean(),
            "sent_pos_share": label_groups["is_pos"].mean(),
            "sent_neg_share": label_groups["is_neg"].mean(),
        }
    )
    # S7: the share of *polarity* carried by reported speech. Weighted by |polarity| so a
    # hedged tweet with no sentiment at all cannot move the measure.
    weight = used.assign(_w=used["abs_polarity"] * used["attribution_flag"].astype(float))
    attributed = weight.groupby(keys, observed=True)["_w"].sum()
    total = used.groupby(keys, observed=True)["abs_polarity"].sum()
    table["sent_attribution"] = (attributed / total.replace(0.0, np.nan)).reindex(table.index)

    # S4's companion: agreement is the majority side's share *among labelled tweets*. A day with
    # no labelled tweet has no agreement to report, hence NaN rather than 1.0.
    labelled = table["sent_pos_share"] + table["sent_neg_share"]
    table["sent_agreement"] = np.where(
        labelled > 0, table[["sent_pos_share", "sent_neg_share"]].max(axis=1) / labelled.replace(0, np.nan), np.nan
    )
    return table


def _gates(
    settings: Settings, canonical_tweets: pd.DataFrame, tiers: dict[str, str], rectangle: pd.MultiIndex
) -> pd.DataFrame:
    """Return per ticker-session gate flags: own-text-only (by tier) and the tweet minimum."""
    cfg = settings.experiments.get("sentiment", {})
    own_by_tier = cfg.get("own_text_only", {}) or {}
    min_tweets = int(settings.experiments.get("quality", {}).get("min_tweets_per_day", 3))

    gates = rectangle.to_frame(index=False)
    gates["tier"] = gates["ticker"].map(tiers).fillna("C")
    gates["own_text_only"] = gates["tier"].map(
        lambda tier: bool(own_by_tier.get(f"tier_{tier.lower()}", True))
    )
    gates["min_tweets_per_day"] = min_tweets
    return gates


def cache_fingerprint(settings: Settings, lexicon_id: str) -> str:
    """Return a cheap fingerprint of everything a cached panel depends on.

    Recomputing the full `data_version` would mean re-reading and cleaning the corpus, which is the
    work the cache exists to avoid. The fingerprint hashes only the *container* files (a few tens of
    milliseconds: any change to the data changes the container, member included), plus both config
    files, the lexicon id, the code version and the cache schema. A cache whose fingerprint does not
    match is treated as absent, so a stale panel can never be served — the failure this prevents is
    a stage reading a table that belongs to different inputs.
    """
    from ..ingest.loaders import sha256_file

    payload = json.dumps(
        {
            "containers": {
                "tweets": sha256_file(settings.tweets_path) if settings.tweets_path.exists() else None,
                "prices": sha256_file(settings.prices_path) if settings.prices_path.exists() else None,
            },
            "settings": settings.raw,
            "experiments": settings.experiments,
            "dataset": settings.dataset_name,
            "lexicon_id": lexicon_id,
            "code_version": CODE_VERSION,
            "cache_schema": CACHE_SCHEMA,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def panel_data_version(
    settings: Settings, hashes: dict[str, Any], lexicon_id: str
) -> str:
    """Return the 12-char id every `ToolResult` carries.

    SPEC.md 4.8: data hash + config hash + lexicon_id + code version. Sorted keys and a fixed
    payload mean two runs over the same inputs and config produce the same id — which is the
    reproducibility contract, and the reason a reviewer can diff two runs at all.

    `cache_schema` is included so that changing which columns the cache holds invalidates old cache
    files instead of letting a stage read a table missing a column it now needs.
    """
    payload = json.dumps(
        {
            "hashes": hashes,
            "experiments": settings.experiments,
            "settings": settings.raw,
            "dataset": settings.dataset_name,
            "lexicon_id": lexicon_id,
            "code_version": CODE_VERSION,
            "cache_schema": CACHE_SCHEMA,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_panel(
    settings: Settings, canonical: Any, lexicon: FrozenLexicon | None = None
) -> Panel:
    """Build the ticker-session panel from the canonical tables.

    Returns a `Panel`; the `meta` dict records every gate, count and id needed by the overview,
    the manifest and the feature book.
    """
    lexicon = lexicon or resolve_lexicon(settings)[0]
    sentiment_cfg = settings.raw.get("sentiment", {}) or {}
    method = str(sentiment_cfg.get("method", "vader_plus_lexicon"))
    experiment_sentiment = settings.experiments.get("sentiment", {}) or {}

    scored, scoring_report = score_tweets(
        canonical.tweets,
        lexicon,
        method=method,
        label_band=float(sentiment_cfg.get("label_band", 0.05)),
        lexicon_weight=float(experiment_sentiment.get("lexicon_weight", 0.5)),
        negation_window=int(experiment_sentiment.get("negation_window", 3)),
        negation_damp=float(experiment_sentiment.get("negation_damp", 0.75)),
    )

    in_window = scored[scored["session_date"].notna()]
    # Counts rather than the raw lists: the list columns are convenient in memory but do not
    # survive the CSV cache, and a stage that only runs from the cache must see the same columns.
    scored["cashtag_count"] = scored["cashtags"].map(len)
    scored["hashtag_count"] = scored["hashtags"].map(len)
    tiers = assign_tiers(canonical.report["own_text_share_per_ticker"], settings)
    tickers = sorted(scored["ticker"].unique())
    sessions = canonical.calendar
    # Session ordinal on the tweets too: the induction sampler and `sample_tweets` group by
    # session, and recomputing the mapping in each caller is how two stages drift apart.
    session_index = {session: index for index, session in enumerate(sessions)}
    scored["session_idx"] = scored["session_date"].map(session_index)

    attention_cfg = settings.experiments.get("attention", {})
    attention = attention_table(
        in_window,
        tickers=tickers,
        sessions=sessions,
        baseline_sessions=int(attention_cfg.get("baseline_sessions", 20)),
        spike_z=float(attention_cfg.get("spike_z", 3.0)),
        sustained_sessions=int(attention_cfg.get("sustained_sessions", 2)),
    )

    rectangle = pd.MultiIndex.from_product([tickers, sessions], names=["ticker", "session_date"])
    gates = _gates(settings, in_window, tiers, rectangle)
    sentiment = _aggregate_sentiment(in_window, gates).reset_index()

    panel = attention.merge(sentiment, on=["ticker", "session_date"], how="left")
    panel["tier"] = panel["ticker"].map(tiers)

    price_cfg = settings.experiments.get("price", {})
    horizons = [int(h) for h in price_cfg.get("horizons", [1, 2, 5])]
    vol_windows = [int(w) for w in price_cfg.get("realized_vol_windows", [5, 20])]
    # Column names come from the config, never from a literal: a run with different windows or
    # horizons must produce a panel with those columns rather than silently losing them.
    horizon_columns = [f"fwd_ret_{h}" for h in horizons]
    vol_columns = [f"rv_{w}" for w in vol_windows]
    priced = price_features_module.price_features(
        canonical.prices, horizons=horizons, vol_windows=vol_windows
    ).rename(columns={"date": "session_date"})
    panel = panel.merge(
        priced[["ticker", "session_date", "close", "adj_close", "volume", "ret_contemp", "abs_ret",
                "volume_z"] + vol_columns + horizon_columns],
        on=["ticker", "session_date"],
        how="left",
    )

    min_tweets = int(settings.experiments.get("quality", {}).get("min_tweets_per_day", 3))
    panel["min_tweets_per_day"] = min_tweets
    panel["low_coverage"] = panel["n_sent_tweets"].fillna(0).astype(int) < min_tweets
    # Diagnostic, reported but never used for a result: how many units would clear the tweet
    # minimum if own-text restriction were switched off. Publishing both numbers is what makes
    # the cost of that gate visible instead of hidden inside a single "excluded" count.
    panel["meets_min_on_all_text"] = panel["n_tweets"] >= min_tweets
    # Sentiment is not computed where it cannot mean anything; leaving the score in place would
    # invite a tool to average it. The day is kept, flagged and counted out loud.
    sentiment_columns = [
        "sent_net", "sent_intensity", "sent_dispersion", "sent_agreement", "fin_directional",
        "sent_uncertainty", "sent_attribution", "sent_pos_share", "sent_neg_share",
        "sent_sarcasm_share",
    ]
    panel.loc[panel["low_coverage"], sentiment_columns] = np.nan

    panel["excluded_from_tests"] = panel["low_coverage"] | panel["tier"].eq("C")
    panel["exclusion_reason"] = np.where(
        panel["tier"].eq("C"),
        "coverage tier C: all text is cross-posted, so there is nothing ticker-specific to test",
        np.where(panel["low_coverage"], f"fewer than {min_tweets} usable tweets in this session", ""),
    )

    split = canonical.report["split"]
    discovery_end = split["discovery_end"]
    panel["period"] = np.where(
        panel["session_date"].astype(str).str.slice(0, 10) <= str(discovery_end), "discovery", "holdout"
    )

    ordered_columns = [
        "ticker", "session_date", "session_idx", "period", "tier",
        "n_tweets", "n_texts_unique", "n_texts_own", "n_texts_shared", "own_text_share",
        "attn_velocity", "attn_z", "spike", "spike_sustained",
        "promo_share", "reply_share", "cashtag_share", "share_of_all_tweets",
        "n_sent_tweets", "sent_net", "sent_pos_share", "sent_neg_share", "sent_intensity",
        "sent_dispersion", "sent_agreement", "fin_directional", "sent_uncertainty",
        "sent_attribution", "sent_sarcasm_share",
        "close", "adj_close", "volume", "ret_contemp", "abs_ret", *vol_columns, "volume_z",
    ] + horizon_columns + [
        "low_coverage", "min_tweets_per_day", "meets_min_on_all_text", "excluded_from_tests", "exclusion_reason",
    ]
    panel = panel[ordered_columns].sort_values(["ticker", "session_idx"]).reset_index(drop=True)

    data_version = panel_data_version(settings, canonical.report["hashes"], lexicon.lexicon_id)
    meta: dict[str, Any] = {
        "data_version": data_version,
        "code_version": CODE_VERSION,
        "dataset": settings.dataset_name,
        "lexicon": {
            "lexicon_id": lexicon.lexicon_id,
            "source": lexicon.source,
            "terms": len(lexicon.terms),
        },
        "scoring": scoring_report,
        "attention": attention_report(panel),
        "price": price_features_module.price_report(priced, [int(h) for h in price_cfg.get("horizons", [1, 2, 5])]),
        "tiers": {"by_ticker": tiers, "counts": {t: sum(1 for v in tiers.values() if v == t) for t in "ABC"}},
        "sessions": len(sessions),
        "n_units": int(len(panel)),
        "units_with_sentiment": int(panel["sent_net"].notna().sum()),
        "units_low_coverage": int(panel["low_coverage"].sum()),
        "units_meeting_min_on_all_text": int(panel["meets_min_on_all_text"].sum()),
        "units_excluded_from_tests": int(panel["excluded_from_tests"].sum()),
        "split": split,
        "canonical_report": canonical.report,
    }
    return Panel(
        panel=panel,
        tweets=scored,
        prices=canonical.prices,
        tiers=tiers,
        calendar=sessions,
        data_version=data_version,
        meta=meta,
    )


# --- persistence: stages communicate through files (SPEC.md 3) ---------------


def cache_dir(settings: Settings) -> Path:
    """Directory holding the cached canonical tables, keyed by data version."""
    return REPO_ROOT / "artifacts" / "cache"


def save_panel(panel: Panel, settings: Settings) -> dict[str, str]:
    """Write the panel and the scored tweets to the cache; return the two paths.

    CSV rather than Parquet: pyarrow is not a dependency of this demo, and the panel is 6,300
    rows. The scored tweet table is larger (~80k rows) and exists so `sample_tweets` can quote
    real text without re-running the whole preparation stage.

    `float_format="%.17g"` is deliberate: 17 significant digits is what makes a float64 round-trip
    exactly through text. Without it a cached run and a freshly built run disagreed in the fourth
    decimal of a rank correlation — small, but the reproducibility contract says the same inputs
    yield identical stats, and "nearly identical" is not that.
    """
    directory = cache_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    panel_path = directory / f"panel_{panel.data_version}.csv"
    tweets_path = directory / f"tweets_{panel.data_version}.csv"
    panel.panel.to_csv(panel_path, index=False, float_format="%.17g")
    columns = [
        "ticker", "session_date", "ts_utc", "ts_original", "text_raw", "text_clean", "polarity",
        "label", "promo_flag", "reply_flag", "own_text_flag", "shared_with_n", "session_idx",
        "cashtag_count", "hashtag_count",
    ]
    panel.tweets[columns].to_csv(tweets_path, index=False, float_format="%.17g")
    # The metadata is cached too: the overview and the manifest quote the lexicon, the gate
    # counts and the tier map, and re-deriving them would mean rebuilding the whole panel.
    meta_path = directory / f"panel_meta_{panel.data_version}.json"
    meta_path.write_text(json.dumps(panel.meta, indent=2, default=str), encoding="utf-8")
    # One pointer per dataset: a `full` run and a `dev` run must never serve each other's panel,
    # which is exactly what a single global "latest" pointer would have done. The pointer records
    # the fingerprint as well as the version, so it can be checked without rebuilding anything.
    pointer = directory / f"panel_latest_{settings.dataset_name}.json"
    pointer.write_text(
        json.dumps(
            {
                "data_version": panel.data_version,
                "fingerprint": cache_fingerprint(settings, str(panel.meta.get("lexicon", {}).get("lexicon_id", ""))),
                "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"panel": str(panel_path), "tweets": str(tweets_path), "meta": str(meta_path)}


def load_cached_panel(settings: Settings, expected_fingerprint: str | None = None) -> Panel | None:
    """Load the cached panel for this dataset, or None when it is absent, stale or unreadable.

    Returns None rather than raising: the cache is an optimisation, and a stale or partial cache
    must send the caller back to rebuilding rather than fail the run. A pointer written by an older
    code version (plain text rather than JSON) is treated as stale, which is the conservative
    reading of "I cannot prove this cache is current".
    """
    pointer = cache_dir(settings) / f"panel_latest_{settings.dataset_name}.json"
    if not pointer.exists():
        return None
    try:
        record = json.loads(pointer.read_text(encoding="utf-8"))
        data_version = str(record["data_version"])
        fingerprint = str(record.get("fingerprint", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        return None
    panel_path = cache_dir(settings) / f"panel_{data_version}.csv"
    tweets_path = cache_dir(settings) / f"tweets_{data_version}.csv"
    if not panel_path.exists() or not tweets_path.exists():
        return None
    # `float_precision="round_trip"` is not optional: pandas' default C parser is not correctly
    # rounded, so it turned a 17-digit value back into a neighbouring float64 and a cached run
    # disagreed with a fresh one in the fourth decimal of a rank correlation. The round-trip parser
    # uses Python's own conversion and reproduces the value exactly.
    panel_frame = pd.read_csv(panel_path, parse_dates=["session_date"], float_precision="round_trip")
    tweets = pd.read_csv(tweets_path, parse_dates=["session_date", "ts_utc"], float_precision="round_trip")
    tiers = (
        panel_frame.drop_duplicates("ticker").set_index("ticker")["tier"].to_dict()
    )
    sessions = sorted(panel_frame["session_date"].unique())
    meta_path = cache_dir(settings) / f"panel_meta_{data_version}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Panel(
        panel=panel_frame,
        tweets=tweets,
        prices=pd.DataFrame(),
        tiers=tiers,
        calendar=list(pd.to_datetime(sessions)),
        data_version=data_version,
        meta=meta,
    )
