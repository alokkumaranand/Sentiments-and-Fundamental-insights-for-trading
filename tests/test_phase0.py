"""Phase 0 tests: read-only loading, the blocking checks, and the profiler's key metrics.

Run either way:
    .venv\\Scripts\\python.exe -m pytest
    .venv\\Scripts\\python.exe -m unittest discover -s tests -t .

Everything here uses TINY synthetic fixtures, so the suite is fast and stays meaningful if
the real dataset is swapped out.

Note on scratch directories: these tests deliberately do NOT use `tempfile` at all. In this
environment the platform temp directory allows creating files but denies the scandir/chmod that
teardown performs, and `mkdtemp`-style directories created inside the workspace cannot be removed
either -- so every test would fail for an environment reason, or leave undeletable litter behind.
`scratch()` below returns a fixed-named, git-ignored workspace directory that `shutil.rmtree`
does handle.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

# The package is not installed, so make src/ importable exactly as run.py does.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_SCRATCH_ROOT = REPO_ROOT / ".tmp-tests"
_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

import pandas as pd  # noqa: E402

from eda_demo.config import Settings  # noqa: E402
from eda_demo.ingest import validate  # noqa: E402
from eda_demo.ingest.loaders import read_raw_csv  # noqa: E402
from eda_demo.ingest.profile import assign_tiers, build_profile  # noqa: E402

TWEET_COLS = ["Date", "Tweet", "Stock Name", "Company Name"]
PRICE_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Stock Name"]


def scratch(name: str) -> Path:
    """Return a clean workspace-local scratch directory for one test.

    Best-effort cleanup only: if a leftover can be removed it is, otherwise the directory is
    reused. Never raises for an environment reason.
    """
    path = _SCRATCH_ROOT / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_settings(tmp: Path, tweets: Path, prices: Path) -> Settings:
    """Build a Settings object pointed at synthetic files.

    Constructed directly instead of by editing YAML: the dataclass is frozen and has no
    required behaviour, which keeps the test independent of config file contents.
    """
    return Settings(
        repo_root=tmp,
        raw={"columns": {"tweets": {"date": "Date", "text": "Tweet", "ticker": "Stock Name",
                                    "company": "Company Name"},
                         "prices": {"date": "Date", "ticker": "Stock Name", "close": "Close",
                                    "adj_close": "Adj Close", "volume": "Volume"}}},
        experiments={"tiers": {"specific_own_text_share": 0.80, "mixed_own_text_share": 0.10},
                     "quality": {"split_return_limit": 0.50}},
        dataset_name="synthetic",
        tweets_path=tweets,
        prices_path=prices,
        tweets_member=None,
        artifacts_dir=tmp / "artifacts",
        quality_report_path=tmp / "quality.md",
        market_timezone="America/New_York",
        close_hour=16,
    )


def write_synthetic(tmp: Path) -> Settings:
    """Write a 3-session, 3-ticker fixture containing one deliberately shared tweet.

    The shared text appears under both AAA and BBB, so a correct profiler must count it
    once in `distinct_texts`, twice in the rows, and zero times in either ticker's own-text
    count.

    Timestamps are UTC, so "after the close" means >= 20:00Z -- 16:00 US/Eastern in EDT.
    Writing 16:30Z here would be 12:30 ET, before the close, and the test would pass for the
    wrong reason. Three tickers reach tier A so that later tests can damage one ticker
    without tripping the "at least two tier-A tickers" gate.
    """
    sessions = ["2021-10-01", "2021-10-04", "2021-10-05"]
    rows = [
        # AAA: 4 own texts
        ("2021-10-01 14:00:00+00:00", "AAA looking strong today", "AAA", "Alpha Co"),
        ("2021-10-04 14:00:00+00:00", "AAA guidance raised", "AAA", "Alpha Co"),
        ("2021-10-05 14:00:00+00:00", "AAA dips on volume", "AAA", "Alpha Co"),
        ("2021-10-02 18:00:00+00:00", "weekend thoughts on AAA", "AAA", "Alpha Co"),  # Saturday
        # BBB: 4 own texts
        ("2021-10-01 15:00:00+00:00", "BBB beats estimates", "BBB", "Beta Co"),
        ("2021-10-04 15:00:00+00:00", "BBB hires a new CFO", "BBB", "Beta Co"),
        ("2021-10-05 15:00:00+00:00", "BBB flat", "BBB", "Beta Co"),
        ("2021-10-05 15:30:00+00:00", "BBB announces a buyback", "BBB", "Beta Co"),
        # CCC: 3 own texts, fully specific
        ("2021-10-01 15:30:00+00:00", "CCC wins a contract", "CCC", "Gamma Co"),
        ("2021-10-04 15:30:00+00:00", "CCC raises guidance", "CCC", "Gamma Co"),
        ("2021-10-05 15:30:00+00:00", "CCC insider selling", "CCC", "Gamma Co"),
        # the SAME text under two tickers: own text for neither
        ("2021-10-04 20:30:00+00:00", "MY TOP PICKS $AAA $BBB $CCC", "AAA", "Alpha Co"),
        ("2021-10-04 20:31:00+00:00", "MY TOP PICKS $AAA $BBB $CCC", "BBB", "Beta Co"),
    ]
    tweets = pd.DataFrame([dict(zip(TWEET_COLS, r)) for r in rows])
    price_rows = []
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0), ("CCC", 25.0)):
        for i, session in enumerate(sessions):
            close = base * (1 + 0.01 * i)
            price_rows.append([session, close, close, close, close, close, 1000, ticker])
    prices = pd.DataFrame(price_rows, columns=PRICE_COLS)

    tweets_path = tmp / "tweets.csv"
    prices_path = tmp / "prices.csv"
    tweets.to_csv(tweets_path, index=False)
    prices.to_csv(prices_path, index=False)
    return make_settings(tmp, tweets_path, prices_path)


class TestLoaders(unittest.TestCase):
    def test_reads_plain_csv_with_bom(self) -> None:
        """A BOM must not survive into the first column name."""
        tmp = scratch("loader_bom")
        path = tmp / "t.csv"
        path.write_text("Date,Tweet\n2021-10-01,hi\n", encoding="utf-8-sig")
        frame, hashes = read_raw_csv(path)
        self.assertEqual(list(frame.columns), ["Date", "Tweet"])
        self.assertIn("container", hashes)
        self.assertNotIn("member", hashes)

    def test_reads_zip_member_without_extracting(self) -> None:
        """The CSV is read from inside the archive and nothing is written beside it."""
        tmp = scratch("loader_zip")
        archive = tmp / "t.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("inner.csv", "Date,Tweet\n2021-10-01,hi\n")
        before = {p.name for p in tmp.iterdir()}
        frame, hashes = read_raw_csv(archive, "inner.csv")
        self.assertEqual(len(frame), 1)
        self.assertEqual(hashes["member_name"], "inner.csv")
        self.assertIn("member", hashes)
        self.assertEqual({p.name for p in tmp.iterdir()}, before, "extracted a file")

    def test_missing_member_is_an_error(self) -> None:
        tmp = scratch("loader_missing_member")
        archive = tmp / "t.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("a.csv", "x\n1\n")
        with self.assertRaises(SystemExit):
            read_raw_csv(archive, "nope.csv")

    def test_missing_file_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            read_raw_csv(Path("does-not-exist.csv"))


class TestBlockingChecks(unittest.TestCase):
    def test_missing_column_is_blocking(self) -> None:
        frame = pd.DataFrame({"Date": ["2021-10-01"], "Tweet": ["x"]})
        with self.assertRaises(validate.BlockingError):
            validate.require_columns(frame, {"date": "Date", "ticker": "Stock Name"}, "tweets")

    def test_unparseable_date_is_blocking(self) -> None:
        frame = pd.DataFrame({"Date": ["2021-10-01", "not-a-date"]})
        with self.assertRaises(validate.BlockingError):
            validate.parse_dates(frame, "Date", "tweets")

    def test_disjoint_tickers_are_blocking(self) -> None:
        with self.assertRaises(validate.BlockingError):
            validate.require_ticker_overlap({"AAA"}, {"BBB"})

    def test_no_own_text_is_blocking(self) -> None:
        with self.assertRaises(validate.BlockingError):
            validate.require_own_text(0)

    def test_too_few_tier_a_is_blocking(self) -> None:
        with self.assertRaises(validate.BlockingError):
            validate.require_tier_a(2)(1)


class TestProfile(unittest.TestCase):
    def test_fanout_own_text_and_calendar(self) -> None:
        """The three facts that decide downstream design, on a fixture where they are known."""
        settings = write_synthetic(scratch("profile_fanout"))
        profile = build_profile(settings)
        t, p = profile["tweets"], profile["prices"]

        # the shared text is one distinct text but two rows
        self.assertEqual(t["distinct_texts"], 12)
        self.assertEqual(t["rows"], 13)
        self.assertEqual(t["shared_texts"], 1)
        self.assertEqual(t["rows_in_shared_text"], 2)

        # AAA has 5 rows of which 1 is shared -> 4/5 own text
        self.assertAlmostEqual(t["own_text_share_per_ticker"]["AAA"], 0.80, places=4)
        self.assertEqual(t["own_text_count_per_ticker"]["AAA"], 4)

        # the Saturday tweet is on a non-trading date; the 20:30/20:31Z pair is after the close
        self.assertEqual(t["non_trading_day_rows"], 1)
        self.assertEqual(t["at_or_after_close_rows"], 2)

        self.assertEqual(p["sessions"], 3)
        self.assertEqual(p["ragged_sessions"], [])
        self.assertTrue(p["split_check"]["passed"])

        # AAA and BBB sit exactly on the 0.80 threshold; CCC is fully specific
        self.assertEqual(profile["tiers"]["by_ticker"], {"AAA": "A", "BBB": "A", "CCC": "A"})
        self.assertEqual(profile["tiers"]["counts"], {"A": 3, "B": 0, "C": 0})

    def test_ragged_session_is_warned_not_blocking(self) -> None:
        """An incomplete session is uncertain, not wrong, so it must not stop the run."""
        settings = write_synthetic(scratch("profile_ragged"))
        prices = pd.read_csv(settings.prices_path)
        # drop BBB's last row so 2021-10-05 exists for only one ticker
        prices = prices[~((prices["Stock Name"] == "BBB") & (prices["Date"] == "2021-10-05"))]
        prices.to_csv(settings.prices_path, index=False)

        profile = build_profile(settings)
        self.assertEqual(profile["prices"]["ragged_sessions"], ["2021-10-05"])
        self.assertTrue(any("incomplete" in w for w in profile["warnings"]))

    def test_unadjusted_split_is_blocking(self) -> None:
        """A 20:1 jump must stop the run: otherwise every downstream return is corrupt."""
        settings = write_synthetic(scratch("profile_split"))
        prices = pd.read_csv(settings.prices_path)
        mask = (prices["Stock Name"] == "AAA") & (prices["Date"] == "2021-10-05")
        prices.loc[mask, "Close"] = prices.loc[mask, "Close"] / 20.0
        prices.to_csv(settings.prices_path, index=False)

        with self.assertRaises(validate.BlockingError):
            build_profile(settings)
        # ...and downgradeable to a note when explicitly asked
        profile = build_profile(settings, enforce_split_check=False)
        self.assertFalse(profile["prices"]["split_check"]["passed"])

    def test_profile_is_json_serialisable_and_reproducible(self) -> None:
        """No numpy scalars may leak, and the same inputs must give the same data_version."""
        settings = write_synthetic(scratch("profile_repro"))
        first = build_profile(settings)
        second = build_profile(settings)
        json.dumps(first)  # raises if any numpy type survived
        self.assertEqual(first["data_version"], second["data_version"])
        self.assertEqual(first["tweets"], second["tweets"])

    def test_tier_thresholds_come_from_config(self) -> None:
        """Thresholds are configuration, never constants buried in code."""
        settings = make_settings(Path("."), Path("a"), Path("b"))
        self.assertEqual(assign_tiers({"X": 0.9, "Y": 0.5, "Z": 0.05}, settings),
                         {"X": "A", "Y": "B", "Z": "C"})

    def test_scope_include_and_exclude(self) -> None:
        """The manual ticker subset must drop rows from BOTH files before any metric."""
        settings = write_synthetic(scratch("profile_scope"))
        # write_synthetic returns a frozen dataclass, so rebuild it with a scope configured
        scoped = replace(settings, experiments={**settings.experiments,
                                                "scope": {"include": None, "exclude": ["AAA"]}})
        profile = build_profile(scoped)

        self.assertTrue(profile["scope"]["applied"])
        self.assertEqual(profile["scope"]["tickers_after"], 2)
        self.assertEqual(profile["scope"]["tweets_after"], 8)   # 13 - 5 AAA rows
        # AAA's own text must be gone from every metric, including the fan-out bookkeeping
        self.assertNotIn("AAA", profile["tweets"]["tweets_per_ticker"])
        self.assertEqual(set(profile["prices"]["ragged_sessions"]), set())
        # its partner in the shared text is no longer shared with anything
        self.assertEqual(profile["tweets"]["shared_texts"], 0)

    def test_scope_include_restricts_to_named_tickers(self) -> None:
        """`include` narrows the run to the named tickers."""
        settings = write_synthetic(scratch("profile_scope_include"))
        scoped = replace(settings, experiments={**settings.experiments,
                                                "scope": {"include": ["BBB", "CCC"], "exclude": []}})
        profile = build_profile(scoped)
        self.assertEqual(set(profile["tweets"]["tweets_per_ticker"]), {"BBB", "CCC"})

    def test_scope_down_to_one_ticker_is_blocking(self) -> None:
        """A one-ticker profile is not a comparison, so the run must refuse rather than mislead."""
        settings = write_synthetic(scratch("profile_scope_one"))
        scoped = replace(settings, experiments={**settings.experiments,
                                                "scope": {"include": ["CCC"], "exclude": []}})
        with self.assertRaises(validate.BlockingError):
            build_profile(scoped)


if __name__ == "__main__":
    unittest.main()
