"""New tweets arriving: append-only ingest, deterministic re-scoring, drift check (SPEC.md 7.3).

Design decision embodied here: **new text is scored with the frozen lexicon, and the LLM is not
called.** The common case for a fresh file is "more of the same", and re-inducing a vocabulary
because a file arrived would make every number in the run depend on a model call that changes
nothing. Re-induction is proposed only when a measured drift check says the frozen vocabulary no
longer covers the text — and even then it is a proposal recorded in a report, not an automatic
rewrite of previously published numbers.

Schema drift is blocking, not silent: if the arriving file's columns differ from what Phase 0
profiled, the run stops with the diff rather than guessing which column is the text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPO_ROOT, Settings
from .features.lexicon import FrozenLexicon, resolve_lexicon, uncovered_token_share
from .ingest.loaders import read_raw_csv
from .ingest.profile import new_run_id
from .ingest.validate import BlockingError
from .prep.clean import clean_tweets
from .prep.align import align_sessions, build_calendar


@dataclass
class AppendResult:
    """What the append path did, and what it recommends."""

    run_id: str
    report: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


def append_tweets(settings: Settings, new_file: Path, write_report: bool = True) -> AppendResult:
    """Ingest a new tweets file, re-score it with the frozen lexicon and check for drift.

    Steps implemented (SPEC.md 7.3): append-only read, deterministic re-scoring with the active
    lexicon, coverage-drift measurement, and a logged recommendation. Existing rows are never
    rewritten: nothing in this function writes to the raw inputs or to any earlier run.
    """
    tcols = (settings.raw.get("columns", {}) or {}).get("tweets", {})
    pcols = (settings.raw.get("columns", {}) or {}).get("prices", {})

    if not new_file.exists():
        raise BlockingError(f"new tweets file not found: {new_file}")

    arriving, arriving_hashes = read_raw_csv(new_file, None)
    # Schema drift is blocking: a renamed or missing column means the file is not the same shape as
    # the one Phase 0 profiled, and guessing which column is the text would corrupt every number.
    missing = [f"{role}={name!r}" for role, name in tcols.items() if name not in arriving.columns]
    if missing:
        raise BlockingError(
            f"schema drift in {new_file.name}: missing {missing}. "
            f"Present: {list(arriving.columns)}. The pipeline does not guess column mappings."
        )

    existing_raw, _ = read_raw_csv(settings.tweets_path, settings.tweets_member)
    existing_keys = set(
        zip(existing_raw[tcols["ticker"]], existing_raw[tcols["date"]], existing_raw[tcols["text"]])
    )
    arriving = arriving[
        ~pd.Series(
            list(zip(arriving[tcols["ticker"]], arriving[tcols["date"]], arriving[tcols["text"]]))
        ).isin(existing_keys).to_numpy()
    ]

    lexicon, provenance = resolve_lexicon(settings)
    cashtag_min = int(settings.experiments.get("quality", {}).get("promo_min_cashtags", 4))
    cleaned, clean_report = clean_tweets(arriving, tcols["text"], cashtag_min)

    from .features.sentiment import score_tweets

    sentiment_cfg = settings.raw.get("sentiment", {}) or {}
    experiment_sentiment = settings.experiments.get("sentiment", {}) or {}
    scored, scoring_report = score_tweets(
        cleaned,
        lexicon,
        method=str(sentiment_cfg.get("method", "vader_plus_lexicon")),
        label_band=float(sentiment_cfg.get("label_band", 0.05)),
        lexicon_weight=float(experiment_sentiment.get("lexicon_weight", 0.5)),
        negation_window=int(experiment_sentiment.get("negation_window", 3)),
        negation_damp=float(experiment_sentiment.get("negation_damp", 0.75)),
    )

    # Alignment reuses the existing price calendar: appended tweets are placed on the sessions the
    # current price file defines, and anything past its last session is flagged rather than guessed.
    prices_raw, _ = read_raw_csv(settings.prices_path, None)
    calendar = build_calendar(prices_raw, pcols["date"])
    session, rolled, drop_reason = align_sessions(
        pd.to_datetime(scored[tcols["date"]], utc=True), calendar, settings.market_timezone, settings.close_hour
    )
    scored["ticker"] = scored[tcols["ticker"]]
    scored["ts_original"] = scored[tcols["date"]]
    scored["session_date"] = session
    scored["rolls_forward"] = rolled
    scored["drop_reason"] = drop_reason

    drift = uncovered_token_share(scored["text_clean"].fillna("").astype(str).tolist(), lexicon)
    threshold = float(settings.experiments.get("quality", {}).get("max_uncovered_token_share", 0.15))
    min_volume = 500
    reinduction = bool(drift["uncovered_token_share"] > threshold and len(scored) >= min_volume)

    run_id = new_run_id(settings.dataset_name)
    report: dict[str, Any] = {
        "run_id": run_id,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(new_file),
        "source_hashes": arriving_hashes,
        "appended_rows": int(len(scored)),
        "duplicate_rows_skipped": int(len(arriving) - len(scored)),
        "tickers": sorted(scored[tcols["ticker"]].unique().tolist()),
        "date_min": str(pd.to_datetime(scored[tcols["date"]], utc=True).min()),
        "date_max": str(pd.to_datetime(scored[tcols["date"]], utc=True).max()),
        "in_window_rows": int(scored["session_date"].notna().sum()),
        "past_last_session_rows": int(scored["session_date"].isna().sum()),
        "rolled_forward_rows": int(scored["rolls_forward"].sum()),
        "clean": clean_report,
        "lexicon": {"lexicon_id": lexicon.lexicon_id, "source": lexicon.source, "path": provenance.get("path")},
        "scoring": scoring_report,
        "llm_calls": 0,
        "drift": drift,
        "drift_threshold": threshold,
        "drift_min_volume": min_volume,
        "reinduction_recommended": reinduction,
        "recommendation": (
            f"coverage drift {drift['uncovered_token_share']:.1%} exceeds {threshold:.0%} on "
            f"{len(scored)} new rows: propose a re-induction (run `induce`)"
            if reinduction
            else f"coverage drift {drift['uncovered_token_share']:.1%} within the {threshold:.0%} budget: "
                 "keep scoring with the frozen lexicon, no LLM call"
        ),
        "history_preserved": "existing rows and earlier runs were not rewritten; this file is a new record",
    }

    result = AppendResult(run_id=run_id, report=report)
    if write_report:
        run_dir = settings.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        append_path = run_dir / "append_report.json"
        append_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        scored_path = run_dir / "appended_tweets.csv"
        scored[
            ["ticker", "session_date", "ts_original", "text_clean", "polarity", "label",
             "promo_flag", "rolls_forward", "drop_reason"]
        ].to_csv(scored_path, index=False)
        docs_path = Path(
            str(settings.raw.get("paths", {}).get("append_report", "docs/append_report.md"))
        )
        docs_path = docs_path if docs_path.is_absolute() else REPO_ROOT / docs_path
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(render_append_report(report), encoding="utf-8")
        result.paths = {"report": str(append_path), "tweets": str(scored_path), "summary": str(docs_path)}
    return result


def render_append_report(report: dict[str, Any]) -> str:
    """Render the append report as Markdown for `docs/append_report.md`."""
    lines = [
        "# New data arrived (append-only)",
        "",
        f"- Run: `{report['run_id']}` at {report['at']}",
        f"- Source: `{report['source_file']}`",
        f"- Appended rows: {report['appended_rows']:,} (duplicates of existing rows skipped: {report['duplicate_rows_skipped']})",
        f"- Tickers: {', '.join(report['tickers']) or 'none'}",
        f"- Window: {report['date_min'][:19]} -> {report['date_max'][:19]}",
        f"- Placed on a session: {report['in_window_rows']:,}; past the last session in the price "
        f"calendar: {report['past_last_session_rows']:,}; rolled forward: {report['rolled_forward_rows']:,}",
        f"- Lexicon used: `{report['lexicon']['lexicon_id']}` ({report['lexicon']['source']})",
        f"- LLM calls: {report['llm_calls']} — new text is scored deterministically with the frozen lexicon",
        "",
        "## Coverage drift",
        "",
        f"- Content tokens seen: {report['drift']['tokens_seen']:,}",
        f"- Not covered by the frozen lexicon (or VADER): {report['drift']['tokens_uncovered']:,} "
        f"({report['drift']['uncovered_token_share']:.1%})",
        f"- Threshold: {report['drift_threshold']:.0%} on at least {report['drift_min_volume']:,} new rows",
        f"- Examples: {', '.join(report['drift']['examples'][:12]) or 'none'}",
        "",
        f"**Recommendation:** {report['recommendation']}",
        "",
        f"**History:** {report['history_preserved']}",
        "",
    ]
    return "\n".join(lines)
