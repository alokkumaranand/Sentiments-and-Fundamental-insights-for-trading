"""Phase 1 tests: the LLM client, the deterministic sampler, and the induction validation gate.

The gate is the part of the system that most needs a test, because it is what makes an LLM's
output safe to use: a term with the right spelling but the wrong sign must be rejected on the
validation slice, and a term that barely occurs must be rejected for support. Both are asserted
here against a hand-built slice, with no network involved.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from tests.fixtures import REPO_ROOT, build_synthetic_panel, scratch, write_fixture

from eda_demo.agents.induction import gate_terms
from eda_demo.agents.llm import LLMClient, LLMConfig, LLMError, LLMResponse, extract_json
from eda_demo.agents.sampler import sample_for_induction
from eda_demo.features.lexicon import (
    FrozenLexicon,
    LexiconProposal,
    Term,
    default_lexicon_path,
    lexicon_id_for,
    load_lexicon,
    uncovered_token_share,
)
from eda_demo.features.sentiment import LexiconMatcher, score_tweets


class TestLexiconContract(unittest.TestCase):
    def setUp(self) -> None:
        self.lexicon = load_lexicon(REPO_ROOT / "config" / "lexicons" / "default_v1.json")

    def test_default_lexicon_loads_and_has_all_families(self) -> None:
        families = {term.family for term in self.lexicon.terms}
        self.assertEqual(families, {"polarity", "bullish", "bearish", "uncertainty", "attribution", "sarcasm"})
        self.assertGreater(len(self.lexicon.terms), 100)

    def test_extra_key_is_rejected(self) -> None:
        """An unexpected key from a model is a loud failure, not a silently ignored field."""
        with self.assertRaises(Exception):
            LexiconProposal.model_validate({"terms": [{"term": "x", "family": "bullish",
                                                       "polarity": 1, "weight": 0.5, "surprise": 1}]})

    def test_lexicon_id_is_order_independent(self) -> None:
        terms = [Term(term="beat", family="bullish", polarity=1.0, weight=0.7),
                 Term(term="miss", family="bearish", polarity=-1.0, weight=0.7)]
        self.assertEqual(
            lexicon_id_for(terms, {"a": 1}, "model", 7),
            lexicon_id_for(list(reversed(terms)), {"a": 1}, "model", 7),
        )

    def test_drift_coverage_counts_content_tokens(self) -> None:
        report = uncovered_token_share(["zibberish flurble beat the estimates"], self.lexicon)
        self.assertGreaterEqual(report["tokens_uncovered"], 2)
        self.assertLessEqual(report["uncovered_token_share"], 1.0)


class TestSentimentScoring(unittest.TestCase):
    def test_lexicon_and_vader_blend(self) -> None:
        lexicon = load_lexicon(REPO_ROOT / "config" / "lexicons" / "default_v1.json")
        frame = pd.DataFrame({
            "text_clean": ["AAA beats estimates and is bullish", "AAA misses and is bearish"],
            "promo_flag": [False, False], "reply_flag": [False, False],
            "cashtags": [[], []], "hashtags": [[], []], "own_text_flag": [True, True],
            "shared_with_n": [1, 1], "ticker": ["AAA", "AAA"],
            "session_date": pd.to_datetime(["2021-10-01", "2021-10-01"]),
            "ts_utc": pd.to_datetime(["2021-10-01T14:00:00Z", "2021-10-01T15:00:00Z"]),
        })
        scored, report = score_tweets(frame, lexicon, "lexicon_only", 0.05, 0.5, 3, 0.75)
        self.assertGreater(scored.loc[0, "polarity"], 0)
        self.assertLess(scored.loc[1, "polarity"], 0)
        self.assertEqual(report["method"], "lexicon_only")

    def test_negation_flips_a_term(self) -> None:
        lexicon = load_lexicon(REPO_ROOT / "config" / "lexicons" / "default_v1.json")
        matcher = LexiconMatcher(lexicon)
        matcher.configure(3, 0.75)
        positive, _ = matcher.polarity_of("this is good")
        negated, _ = matcher.polarity_of("this is not good")
        self.assertGreater(positive, 0)
        self.assertLess(negated, 0)
        self.assertLess(abs(negated), abs(positive))  # damped, not merely flipped


class TestSampler(unittest.TestCase):
    def test_sample_and_validation_are_session_disjoint(self) -> None:
        settings = write_fixture(scratch("sampler_disjoint"))
        from eda_demo.prep.align import build_canonical

        canonical = build_canonical(settings)
        result = sample_for_induction(
            canonical.tweets, per_ticker=3, max_total=9, validation_parity=1,
            strata_caps={"promo": 0.2, "reply": 0.1, "cashtag_dense": 0.2},
        )
        sample_sessions = set(result.sample["session_date"])
        validation_sessions = set(result.validation["session_date"])
        self.assertEqual(sample_sessions & validation_sessions, set(), "slices overlap")
        self.assertLessEqual(len(result.sample), 9)
        self.assertGreater(result.report["validation_rows"], 0)

    def test_sample_is_deterministic(self) -> None:
        settings = write_fixture(scratch("sampler_deterministic"))
        from eda_demo.prep.align import build_canonical

        canonical = build_canonical(settings)
        first = sample_for_induction(canonical.tweets, 3, 9, 1, {})
        second = sample_for_induction(canonical.tweets, 3, 9, 1, {})
        self.assertEqual(list(first.sample["sample_hash"]), list(second.sample["sample_hash"]))


def _gate_validation() -> pd.DataFrame:
    """A hand-built validation slice: 16 tweets over 4 sessions.

    'sparkle' appears in all of them: 12 sit in clearly positive context and 4 in neutral context
    (attribution words only, which carry no polarity), so the gate has support, a decisive context,
    and a known sign to compare against. Nothing here comes from the real corpus, so the expectation
    is arithmetic rather than empirical.
    """
    positive = ["ACME sparkle beat estimates", "ACME sparkle strong results",
                "ACME sparkle rally continues", "ACME sparkle gains again",
                "ACME sparkle upgraded today", "ACME sparkle record high",
                "ACME sparkle bullish setup", "ACME sparkle more upside",
                "ACME sparkle beats again", "ACME sparkle strong quarter",
                "ACME sparkle surges higher", "ACME sparkle optimistic outlook"]
    neutral = ["ACME sparkle announced a schedule", "ACME sparkle published a notice",
               "ACME sparkle posted a list", "ACME sparkle updated the page"]
    rows = []
    for index, text in enumerate(positive + neutral):
        rows.append({
            "ticker": "ACME",
            "session_date": pd.Timestamp("2021-10-0%d" % (1 + index % 4)),
            "text_clean": text,
        })
    return pd.DataFrame(rows)


class TestInductionGate(unittest.TestCase):
    """The gate is what makes an LLM's vocabulary safe to use, so it is tested directly."""

    def setUp(self) -> None:
        self.settings = write_fixture(scratch("gate_fixture"))
        self.lexicon = load_lexicon(default_lexicon_path(self.settings))
        self.validation = _gate_validation()

    def _gate(self, terms: list[Term], min_support: int = 2, sign_agreement: float = 0.6):
        return gate_terms(
            terms, self.validation, self.lexicon,
            min_support=min_support, sign_agreement=sign_agreement,
            resamples=100, block_sessions=2, seed=7,
        )

    def test_wrong_sign_term_is_rejected(self) -> None:
        """'sparkle' sits in positive context here; claiming a negative polarity must fail."""
        accepted, rejected = self._gate([Term(term="sparkle", family="polarity", polarity=-1.0, weight=0.8)])
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("sign agreement", rejected[0]["rule"])

    def test_term_below_min_support_is_rejected(self) -> None:
        accepted, rejected = self._gate([Term(term="zibberish", family="polarity", polarity=1.0, weight=0.5)])
        self.assertEqual(accepted, [])
        self.assertIn("support", rejected[0]["rule"])

    def test_positive_context_term_is_accepted_with_evidence(self) -> None:
        accepted, rejected = self._gate([Term(term="sparkle", family="polarity", polarity=1.0, weight=0.8)],
                                        min_support=2)
        self.assertEqual(len(accepted), 1, f"rejected: {rejected}")
        self.assertGreaterEqual(accepted[0].support, 2)
        self.assertIsNotNone(accepted[0].ci_95)
        self.assertGreaterEqual(accepted[0].sign_agreement, 0.6)

    def test_unevaluable_ci_rejects_rather_than_passes(self) -> None:
        """One occurrence cannot produce a bootstrap CI, so the term must not be accepted."""
        accepted, rejected = self._gate([Term(term="beat", family="bullish", polarity=1.0, weight=0.7)],
                                        min_support=1)
        self.assertEqual(accepted, [])
        self.assertIn("could not be estimated", rejected[0]["rule"])

    def test_marker_family_only_needs_support(self) -> None:
        term = Term(term="sparkle", family="uncertainty", polarity=0.0, weight=0.0)
        accepted, rejected = self._gate([term], min_support=2)
        self.assertEqual(len(accepted), 1, f"rejected: {rejected}")
        self.assertIsNone(accepted[0].ci_95)

    def test_gate_output_is_deterministic(self) -> None:
        terms = [Term(term="sparkle", family="polarity", polarity=1.0, weight=0.8)]
        first, _ = self._gate(terms)
        second, _ = self._gate(terms)
        self.assertEqual([term.ci_95 for term in first], [term.ci_95 for term in second])


class TestLLMClientWithoutEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = write_fixture(scratch("llm_off"))
        self.client = LLMClient(self.settings, LLMConfig(enabled=False, base_url="", model="", api_key=""))

    def test_unavailable_client_raises_instead_of_guessing(self) -> None:
        self.assertFalse(self.client.available)
        with self.assertRaises(LLMError):
            self.client.complete([{"role": "user", "content": "hi"}])

    def test_ping_reports_why_it_did_not_run(self) -> None:
        result = self.client.ping()
        self.assertFalse(result["ok"])
        self.assertFalse(result["attempted"])
        self.assertIn("EDA_LLM_ENABLED", result["reason"])

    def test_describe_never_contains_the_key(self) -> None:
        client = LLMClient(self.settings, LLMConfig(enabled=True, base_url="https://example.test/v1",
                                                    model="m", api_key="secret-key"))
        self.assertNotIn("secret-key", json.dumps(client.describe()))

    def test_extract_json_handles_fences_and_preamble(self) -> None:
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Here you go: {"a": [1, 2]}  thanks'), {"a": [1, 2]})
        with self.assertRaises(LLMError):
            extract_json("no json at all")

    def test_extract_json_salvages_a_truncated_answer(self) -> None:
        """A cut-off answer keeps its complete leading elements; the partial tail is dropped."""
        from eda_demo.agents.llm import _looks_truncated

        truncated = '{"terms":[{"term":"a","family":"polarity"},{"term":"b","family":"polarity"},{"term":"par'
        payload = extract_json(truncated)
        self.assertEqual([term["term"] for term in payload["terms"]], ["a", "b"])
        self.assertTrue(_looks_truncated(truncated))
        self.assertFalse(_looks_truncated('{"terms":[{"term":"a"}]}'))

    def test_extract_json_salvages_a_truncated_nested_value(self) -> None:
        payload = extract_json('{"findings":[{"title":"x"},{"title":"y","limitations":[1,2')
        self.assertEqual(len(payload["findings"]), 1)

    def test_extract_json_refuses_a_truncated_scalar(self) -> None:
        with self.assertRaises(LLMError):
            extract_json('{"summary":"abc')


if __name__ == "__main__":
    unittest.main()
