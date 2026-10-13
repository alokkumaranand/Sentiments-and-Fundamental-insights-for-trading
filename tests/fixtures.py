"""Shared synthetic fixtures for the phase 1-4 tests.

Everything here is tiny and hand-computable, so the suite stays fast and stays meaningful if the
real dataset is swapped. The fixture is built to exercise the facts the design rests on: three
tickers with different own-text shares, a deliberately shared text, an after-close tweet that must
roll forward, and a weekend tweet.

`scratch()` deliberately does NOT use `tempfile` — see the note in `test_phase0.py`: this
environment allows creating files in the platform temp directory but denies the scandir/chmod that
teardown performs, so the failure would appear during cleanup and mask the real result. A fixed
name under `.tmp-tests/` is removable with `shutil.rmtree`.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from eda_demo.config import Settings  # noqa: E402

_SCRATCH_ROOT = REPO_ROOT / ".tmp-tests"
_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

TWEET_COLS = ["Date", "Tweet", "Stock Name", "Company Name"]
PRICE_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Stock Name"]

TWEET_COLUMN_MAP = {"date": "Date", "text": "Tweet", "ticker": "Stock Name", "company": "Company Name"}
PRICE_COLUMN_MAP = {"date": "Date", "ticker": "Stock Name", "close": "Close",
                    "adj_close": "Adj Close", "volume": "Volume"}

EXPERIMENTS: dict = {
    "scope": {"include": None, "exclude": []},
    # min_tweets_per_day is 1 rather than the shipped 3 so the tiny fixture has usable units at all.
    # A ticker-session with no tweets is still below this minimum, so the low-coverage path is still
    # exercised rather than skipped.
    "quality": {"min_tweets_per_day": 1, "promo_min_cashtags": 4, "split_return_limit": 0.50,
                "max_uncovered_token_share": 0.15},
    "tiers": {"specific_own_text_share": 0.80, "mixed_own_text_share": 0.10},
    "split": {"discovery_fraction": 0.70},
    "attention": {"baseline_sessions": 3, "spike_z": 3.0, "sustained_sessions": 2},
    "price": {"realized_vol_windows": [2, 3], "horizons": [1, 2], "return_basis": "adj_close"},
    "sentiment": {"negation_window": 3, "negation_damp": 0.75, "lexicon_weight": 0.5,
                  "exclude_promo": False,
                  "own_text_only": {"tier_a": False, "tier_b": True, "tier_c": True}},
    "induction": {"validation_parity": 1, "strata_caps": {"promo": 0.2, "reply": 0.1, "cashtag_dense": 0.2},
                  "bootstrap_resamples": 200, "bootstrap_block_sessions": 2},
    "bootstrap": {"block_sessions": 2, "resamples": 200, "seed": 7},
    "multiple_testing": {"alpha": 0.05, "method": "benjamini_hochberg"},
    "tools": {"table_row_cap": 50},
    "charts": {"default_backend": "ascii"},
}


def scratch(name: str) -> Path:
    """Return a clean workspace-local scratch directory for one test (best-effort cleanup)."""
    path = _SCRATCH_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_settings(tmp: Path, tweets: Path, prices: Path, dataset: str = "synthetic") -> Settings:
    """Build a Settings object pointed at synthetic files, with no `.env` and no network."""
    return Settings(
        repo_root=tmp,
        raw={"columns": {"tweets": dict(TWEET_COLUMN_MAP), "prices": dict(PRICE_COLUMN_MAP)},
             # Generated convenience copies go to the scratch directory, never into the repo's docs/.
             "paths": {"quality_report": str(tmp / "quality.md"), "append_report": str(tmp / "append_report.md")},
             "sentiment": {"method": "lexicon_only", "label_band": 0.05,
                           "lexicon": str(REPO_ROOT / "config" / "lexicons" / "default_v1.json")},
             "llm": {"tool_calling": "json_prompt", "temperature": 0, "max_tokens": 500, "cache": False,
                     "cache_dir": str(tmp / "llm_cache")},
             "agents": {"induction": {"per_ticker": 3, "max_total": 9, "min_support": 2, "sign_agreement": 0.6},
                        "explorer": {"max_tool_calls": 2, "retries_on_schema_error": 0},
                        "presenter": {"max_findings": 4}},
             "market": {"timezone": "America/New_York", "close_hour": 16}},
        experiments=EXPERIMENTS,
        dataset_name=dataset,
        tweets_path=tweets,
        prices_path=prices,
        tweets_member=None,
        artifacts_dir=tmp / "artifacts" / "runs",
        quality_report_path=tmp / "quality.md",
        market_timezone="America/New_York",
        close_hour=16,
    )


def write_fixture(tmp: Path, sessions: list[str] | None = None, dataset: str = "synthetic") -> Settings:
    """Write a 3-ticker, 6-session fixture and return settings for it.

    Layout (all timestamps UTC; the ET close at 16:00 is 20:00Z in October, so 16:0xZ is midday ET):
      AAA - 6 own texts, one of them at/after the close, plus one weekend tweet (both roll forward)
      BBB - 5 own texts plus one row carrying the shared text
      CCC - 1 own text plus four rows carrying the shared text, so it is the most cross-posted ticker
    The shared text is own to nobody, and it appears under all three tickers, which is what makes
    `shared_with_n` 3 and every one of those rows non-own. Six sessions give a discovery period long
    enough (12 usable ticker-sessions) for the tools that need a minimum sample.
    """
    sessions = sessions or ["2021-10-01", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07", "2021-10-08"]
    shared = "MY TOP PICKS $AAA $BBB $CCC http://t.co/x"
    rows = [
        ("2021-10-01 14:00:00+00:00", "AAA strong earnings beat", "AAA", "Alpha Co"),
        ("2021-10-04 14:00:00+00:00", "AAA raises guidance", "AAA", "Alpha Co"),
        ("2021-10-05 14:00:00+00:00", "AAA dips on weak volume", "AAA", "Alpha Co"),
        ("2021-10-06 14:00:00+00:00", "AAA announces a buyback", "AAA", "Alpha Co"),
        ("2021-10-06 21:00:00+00:00", "AAA after the close thoughts", "AAA", "Alpha Co"),
        ("2021-10-07 14:00:00+00:00", "AAA is bullish into the print", "AAA", "Alpha Co"),
        ("2021-10-02 18:00:00+00:00", "AAA weekend chatter", "AAA", "Alpha Co"),
        ("2021-10-01 15:00:00+00:00", "BBB beats estimates", "BBB", "Beta Co"),
        ("2021-10-04 15:00:00+00:00", "BBB hires a new CFO", "BBB", "Beta Co"),
        ("2021-10-05 15:00:00+00:00", "BBB flat on the day", "BBB", "Beta Co"),
        ("2021-10-06 15:00:00+00:00", "BBB announces a buyback", "BBB", "Beta Co"),
        ("2021-10-07 15:00:00+00:00", "BBB misses on weak guidance", "BBB", "Beta Co"),
        ("2021-10-01 15:30:00+00:00", "CCC wins a contract", "CCC", "Gamma Co"),
        ("2021-10-04 16:00:00+00:00", shared, "AAA", "Alpha Co"),
        ("2021-10-04 16:01:00+00:00", shared, "BBB", "Beta Co"),
        ("2021-10-05 16:02:00+00:00", shared, "CCC", "Gamma Co"),
        ("2021-10-06 16:03:00+00:00", shared, "CCC", "Gamma Co"),
        ("2021-10-07 16:04:00+00:00", shared, "CCC", "Gamma Co"),
        ("2021-10-07 16:05:00+00:00", shared, "CCC", "Gamma Co"),
    ]
    tweets = pd.DataFrame([dict(zip(TWEET_COLS, row)) for row in rows])

    price_rows = []
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0), ("CCC", 25.0)):
        for index, session in enumerate(sessions):
            close = base * (1 + 0.02 * index)
            price_rows.append([session, close - 0.5, close + 0.5, close - 1.0, close, close, 1000 + 10 * index, ticker])
    prices = pd.DataFrame(price_rows, columns=PRICE_COLS)

    tweets_path = tmp / "tweets.csv"
    prices_path = tmp / "prices.csv"
    tweets.to_csv(tweets_path, index=False)
    prices.to_csv(prices_path, index=False)
    return make_settings(tmp, tweets_path, prices_path, dataset=dataset)


def build_synthetic_panel(settings: Settings):
    """Build the canonical tables and the panel for a synthetic fixture.

    Returns the `Panel`. Used by the tool, findings and pipeline tests so they all read the same
    table the CLI would.
    """
    from eda_demo.features.panel import build_panel
    from eda_demo.features.lexicon import load_lexicon, default_lexicon_path
    from eda_demo.prep.align import build_canonical

    canonical = build_canonical(settings)
    lexicon = load_lexicon(default_lexicon_path(settings))
    return build_panel(settings, canonical, lexicon)
