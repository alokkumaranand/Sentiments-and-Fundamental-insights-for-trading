"""Phase 2 tests: alignment, leakage, fan-out, and the two feature families.

The alignment tests matter most: every number downstream depends on a tweet landing in the session
that could actually act on it, and on `attn_z` never seeing its own day. Both are asserted on a
fixture whose expected answers can be written by hand.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.fixtures import build_synthetic_panel, scratch, write_fixture

from eda_demo.features.attention import attention_table
from eda_demo.features.definitions import FEATURES, HEADLINE_MEASURES
from eda_demo.features.lexicon import default_lexicon_path, load_lexicon
from eda_demo.features.panel import build_panel
from eda_demo.features.price import price_features
from eda_demo.prep.align import align_sessions, build_canonical, session_split
from eda_demo.prep.clean import clean_text


class TestCleaning(unittest.TestCase):
    def test_entities_urls_and_whitespace(self) -> None:
        cleaned = clean_text(pd.Series(["RT &amp;  $AAA   great!   https://t.co/abc"]))
        self.assertEqual(cleaned.iloc[0], "RT & $AAA great! __URL__")

    def test_url_token_survives_so_promo_detection_still_works(self) -> None:
        cleaned = clean_text(pd.Series(["check this http://t.co/x"]))
        self.assertIn("__URL__", cleaned.iloc[0])


class TestAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = [pd.Timestamp(day) for day in ("2021-10-01", "2021-10-04", "2021-10-05")]

    def test_before_close_stays_on_the_same_session(self) -> None:
        ts = pd.Series(pd.to_datetime(["2021-10-04T14:00:00Z"]))  # 10:00 ET
        session, rolled, reason = align_sessions(ts, self.calendar, "America/New_York", 16)
        self.assertEqual(session.iloc[0], pd.Timestamp("2021-10-04"))
        self.assertFalse(bool(rolled.iloc[0]))
        self.assertIsNone(reason.iloc[0])

    def test_after_close_rolls_to_the_next_session(self) -> None:
        ts = pd.Series(pd.to_datetime(["2021-10-04T21:00:00Z"]))  # 17:00 ET
        session, rolled, _ = align_sessions(ts, self.calendar, "America/New_York", 16)
        self.assertEqual(session.iloc[0], pd.Timestamp("2021-10-05"))
        self.assertTrue(bool(rolled.iloc[0]))

    def test_weekend_rolls_to_monday(self) -> None:
        ts = pd.Series(pd.to_datetime(["2021-10-02T18:00:00Z"]))  # Saturday
        session, _, _ = align_sessions(ts, self.calendar, "America/New_York", 16)
        self.assertEqual(session.iloc[0], pd.Timestamp("2021-10-04"))

    def test_tweet_past_the_last_session_is_flagged_not_placed(self) -> None:
        ts = pd.Series(pd.to_datetime(["2021-10-08T14:00:00Z"]))
        session, _, reason = align_sessions(ts, self.calendar, "America/New_York", 16)
        self.assertTrue(pd.isna(session.iloc[0]))
        self.assertIsNotNone(reason.iloc[0])

    def test_session_split_is_chronological_and_covers_every_session(self) -> None:
        split = session_split(self.calendar, 0.70)
        self.assertEqual(split["discovery_sessions"] + split["holdout_sessions"], len(self.calendar))
        self.assertEqual(split["discovery_start"], "2021-10-01")
        self.assertEqual(split["holdout_end"], "2021-10-05")


class TestCanonicalTables(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = write_fixture(scratch("canonical"))
        self.canonical = build_canonical(self.settings)

    def test_fanout_marks_the_shared_text_for_every_ticker(self) -> None:
        tweets = self.canonical.tweets
        shared = tweets[tweets["text_clean"].str.startswith("MY TOP PICKS")]
        self.assertEqual(len(shared), 6)
        self.assertEqual(set(shared["shared_with_n"]), {3})
        self.assertFalse(bool(shared["own_text_flag"].any()))

    def test_tickers_get_distinct_own_text_shares(self) -> None:
        shares = self.canonical.report["own_text_share_per_ticker"]
        self.assertGreater(shares["AAA"], shares["CCC"])

    def test_rows_out_of_window_are_counted_not_dropped(self) -> None:
        report = self.canonical.report
        self.assertEqual(report["rows_in"], report["rows_in_window"] + report["rows_out_of_window"])
        self.assertEqual(report["rows_in"], 19)

    def test_rolled_share_matches_the_known_fixture(self) -> None:
        """Exactly two rows move to a later session: the after-close tweet and the weekend one."""
        self.assertEqual(self.canonical.report["rolled_rows"], 2)
        self.assertAlmostEqual(self.canonical.report["rolled_share"], 2 / 19, places=4)
        self.assertEqual(self.canonical.report["rows_out_of_window"], 0)


class TestPriceFeatures(unittest.TestCase):
    def test_forward_returns_are_forward_and_null_at_the_tail(self) -> None:
        prices = pd.DataFrame({
            "ticker": ["AAA"] * 5,
            "date": pd.to_datetime(["2021-10-01", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07"]),
            "close": [100.0, 110.0, 121.0, 133.1, 146.41],
            "adj_close": [100.0, 110.0, 121.0, 133.1, 146.41],
            "volume": [1000, 1100, 1200, 1300, 1400],
        })
        features = price_features(prices, horizons=[1, 2], vol_windows=[2])
        self.assertAlmostEqual(features.loc[0, "fwd_ret_1"], 0.10, places=6)
        self.assertTrue(pd.isna(features.loc[4, "fwd_ret_1"]), "last session must have no forward return")
        self.assertTrue(pd.isna(features.loc[3, "fwd_ret_2"]))
        self.assertAlmostEqual(features.loc[1, "ret_contemp"], 0.10, places=6)

    def test_volume_z_uses_a_shifted_baseline(self) -> None:
        """A volume spike must not be part of its own baseline, so its own z is the outlier."""
        prices = pd.DataFrame({
            "ticker": ["AAA"] * 8,
            "date": pd.date_range("2021-10-01", periods=8, freq="D"),
            "close": [100.0] * 8,
            "adj_close": [100.0] * 8,
            "volume": [1000, 1010, 990, 1005, 995, 1002, 900_000, 1001],
        })
        features = price_features(prices, horizons=[1], vol_windows=[2])
        self.assertGreater(features.loc[6, "volume_z"], 1.0)
        # The day after is compared with a baseline that now contains the spike, so its own z is far
        # smaller: that is the whole point of shifting the baseline.
        self.assertLess(features.loc[7, "volume_z"], features.loc[6, "volume_z"])
        self.assertLess(abs(features.loc[7, "volume_z"]), 1.5)

    def test_flat_baseline_gives_an_undefined_z_not_an_infinite_one(self) -> None:
        prices = pd.DataFrame({
            "ticker": ["AAA"] * 7,
            "date": pd.date_range("2021-10-01", periods=7, freq="D"),
            "close": [100.0] * 7,
            "adj_close": [100.0] * 7,
            "volume": [1000] * 6 + [900_000],
        })
        features = price_features(prices, horizons=[1], vol_windows=[2])
        self.assertTrue(pd.isna(features.loc[6, "volume_z"]), "a zero-variance baseline must not divide")


class TestAttentionFamily(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = [pd.Timestamp(day) for day in
                         ("2021-10-01", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07")]
        # One ticker, own texts 1,1,1,1 then a spike of 10 on the last session.
        rows = []
        counts = [1, 1, 1, 1, 10]
        for session, count in zip(self.sessions, counts):
            for index in range(count):
                rows.append({
                    "ticker": "AAA", "session_date": session, "text_key": f"{session}-{index}",
                    "own_text_flag": True, "promo_flag": False, "reply_flag": False, "cashtags": [],
                })
        self.tweets = pd.DataFrame(rows)

    def test_counts_and_own_share(self) -> None:
        table = attention_table(self.tweets, ["AAA"], self.sessions, 3, 3.0, 2)
        self.assertEqual(list(table["n_tweets"]), [1, 1, 1, 1, 10])
        self.assertEqual(list(table["n_texts_own"]), [1, 1, 1, 1, 10])
        self.assertAlmostEqual(table.loc[4, "own_text_share"], 1.0)

    def test_attn_z_is_past_only(self) -> None:
        """Changing the last session's count must not change any earlier z, and vice versa."""
        table = attention_table(self.tweets, ["AAA"], self.sessions, 3, 3.0, 2)
        # The baseline for session 4 is sessions 1-3 (three own texts, MAD 0), so the z is undefined
        # rather than infinite, and the spike is visible once a spread exists in the baseline.
        self.assertTrue(pd.isna(table.loc[0, "attn_z"]))
        modified = self.tweets.copy()
        modified.loc[modified["session_date"] == self.sessions[-1], "text_key"] = "changed"
        modified_table = attention_table(modified, ["AAA"], self.sessions, 3, 3.0, 2)
        for index in range(3):
            self.assertEqual(
                pd.isna(table.loc[index, "attn_z"]), pd.isna(modified_table.loc[index, "attn_z"])
            )

    def test_rectangle_is_complete_and_empty_cells_are_zero(self) -> None:
        table = attention_table(self.tweets, ["AAA", "BBB"], self.sessions, 3, 3.0, 2)
        self.assertEqual(len(table), 10)
        bbb = table[table["ticker"] == "BBB"]
        self.assertEqual(int(bbb["n_tweets"].sum()), 0)
        self.assertTrue(bbb["own_text_share"].isna().all(), "no tweets means no share, not 0.0")


class TestPanel(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = write_fixture(scratch("panel"))
        self.panel = build_synthetic_panel(self.settings)

    def test_panel_is_a_full_rectangle_with_the_expected_columns(self) -> None:
        frame = self.panel.panel
        self.assertEqual(len(frame), 3 * 6, "one row per ticker-session, including empty cells")
        for column in ("sent_net", "sent_pos_share", "sent_neg_share", "sent_intensity",
                       "sent_dispersion", "sent_agreement", "fin_directional", "sent_uncertainty",
                       "sent_attribution", "n_tweets", "n_texts_unique", "n_texts_own",
                       "n_texts_shared", "own_text_share", "attn_velocity", "attn_z",
                       "fwd_ret_1", "fwd_ret_2", "rv_2", "volume_z", "tier", "low_coverage"):
            self.assertIn(column, frame.columns, column)

    def test_tier_assignment_follows_the_configured_thresholds(self) -> None:
        """Tiers are assigned from the measured own-text share, not chosen by hand."""
        shares = self.panel.meta["canonical_report"]["own_text_share_per_ticker"]
        for ticker, share in shares.items():
            expected = "A" if share >= 0.80 else "B" if share >= 0.10 else "C"
            self.assertEqual(self.panel.tiers[ticker], expected, f"{ticker} share={share}")
        self.assertEqual(self.panel.tiers["AAA"], "A")
        self.assertGreater(shares["AAA"], shares["CCC"])

    def test_low_coverage_units_have_no_sentiment_score(self) -> None:
        frame = self.panel.panel
        low = frame[frame["low_coverage"]]
        self.assertTrue(low["sent_net"].isna().all())
        self.assertTrue(frame["excluded_from_tests"][frame["tier"] == "C"].all())

    def test_gate_diagnostic_is_reported(self) -> None:
        meta = self.panel.meta
        self.assertIn("units_meeting_min_on_all_text", meta)
        self.assertLessEqual(meta["units_with_sentiment"], meta["units_meeting_min_on_all_text"])

    def test_data_version_is_stable_for_the_same_inputs(self) -> None:
        again = build_synthetic_panel(self.settings)
        self.assertEqual(self.panel.data_version, again.data_version)

    def test_the_cache_round_trips_exactly(self) -> None:
        """A cached panel must be bit-identical to a fresh build.

        This is a regression guard: pandas' default CSV parser is not correctly rounded, so a cached
        panel used to differ from a fresh one in the last bits of some float columns, which showed up
        as a rank correlation differing in the fourth decimal between two runs.
        """
        from eda_demo.features.panel import cache_fingerprint, load_cached_panel, save_panel

        save_panel(self.panel, self.settings)
        reloaded = load_cached_panel(
            self.settings, cache_fingerprint(self.settings, self.panel.meta["lexicon"]["lexicon_id"])
        )
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        columns = [column for column in self.panel.panel.columns
                   if pd.api.types.is_numeric_dtype(self.panel.panel[column])]
        left = self.panel.panel.sort_values(["ticker", "session_idx"]).reset_index(drop=True)[columns]
        right = reloaded.panel.sort_values(["ticker", "session_idx"]).reset_index(drop=True)[columns]
        for column in columns:
            pd.testing.assert_series_equal(left[column], right[column], check_names=False,
                                           check_dtype=False, rtol=0, atol=0, obj=column)

    def test_every_headline_feature_has_a_definition_and_a_caveat(self) -> None:
        """Phase 2's exit condition, asserted: 7 sentiment and 7 attention measures, all documented.

        Some measures are two columns (S2 is the positive/negative share pair, S4 is dispersion plus
        agreement), so the test checks the measure *labels*, not the column count.
        """
        labels = set(HEADLINE_MEASURES.values())
        self.assertEqual({label for label in labels if label.startswith("S")},
                         {f"S{index}" for index in range(1, 8)})
        self.assertEqual({label for label in labels if label.startswith("A")},
                         {f"A{index}" for index in range(1, 8)})
        for name in HEADLINE_MEASURES:
            self.assertTrue(FEATURES[name]["definition"].strip(), name)
            self.assertTrue(FEATURES[name]["caveat"].strip(), name)


if __name__ == "__main__":
    unittest.main()
