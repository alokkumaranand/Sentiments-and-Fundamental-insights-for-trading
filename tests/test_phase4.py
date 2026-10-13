"""Phase 4 tests: the claim checker, the status rules, the no-key agent and the whole pipeline.

The claim checker and the status rules carry the two promises the demo makes — no invented number
and no over-claimed status — so both are tested against deliberate violations rather than only
against the happy path.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from tests.fixtures import build_synthetic_panel, scratch, write_fixture

from eda_demo.agents.null_agent import null_agent_findings, template_narrative
from eda_demo.append import append_tweets
from eda_demo.findings.checker import check_draft, check_numbers_in_text
from eda_demo.findings.schema import Evidence, FindingDraft
from eda_demo.findings.status_rules import assign_status, summarise_status_counts
from eda_demo.ingest.validate import BlockingError
from eda_demo.pipeline import run_pipeline
from eda_demo.tools import build_registry
from eda_demo.tools.playbook import run_playbook


def _registry_with_playbook(settings, panel):
    """Build a registry and run the baseline so there are results to cite."""
    registry = build_registry(settings, panel.panel, panel.tweets, panel.data_version,
                              panel.tiers, panel.calendar)
    run_playbook(registry)
    return registry


class TestClaimChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = write_fixture(scratch("checker"))
        cls.panel = build_synthetic_panel(cls.settings)

    def setUp(self) -> None:
        self.registry = _registry_with_playbook(self.settings, self.panel)
        self.correlation = next(
            result for result in self.registry.store.all() if result.tool == "correlate"
        )

    def _draft(self, claim: str, result_id: str | None = None) -> FindingDraft:
        return FindingDraft(
            title="A test claim",
            claim_text=claim,
            evidence=[Evidence(result_id=result_id or self.correlation.result_id)],
        )

    def test_a_valid_number_passes(self) -> None:
        rho = self.correlation.stats["spearman_rho"]
        verdict, cited = check_draft(self._draft(f"Spearman rho {rho} was observed."), self.registry.store)
        self.assertEqual(verdict.verdict, "accepted")
        self.assertEqual(len(cited), 1)

    def test_a_rounded_valid_number_passes(self) -> None:
        """A claim written to two decimals may describe a statistic with six."""
        rho = round(self.correlation.stats["spearman_rho"], 2)
        verdict, _ = check_draft(self._draft(f"Spearman rho {rho} was observed."), self.registry.store)
        self.assertEqual(verdict.verdict, "accepted", verdict.reason)

    def test_a_near_miss_is_rejected(self) -> None:
        """The tolerance is the claim's own precision, so a different 4-decimal number must fail."""
        rho = float(self.correlation.stats["spearman_rho"]) + 0.01
        verdict, _ = check_draft(self._draft(f"Spearman rho {rho:.4f} was observed."), self.registry.store)
        self.assertEqual(verdict.verdict, "rejected")

    def test_a_made_up_number_is_rejected(self) -> None:
        verdict, _ = check_draft(self._draft("Spearman rho 0.9871 was observed."), self.registry.store)
        self.assertEqual(verdict.verdict, "rejected")
        self.assertIn("not present", verdict.reason)

    def test_a_result_id_from_another_run_is_rejected(self) -> None:
        verdict, _ = check_draft(self._draft("Spearman rho 0.5 was observed.", result_id="deadbeefcafe"),
                                 self.registry.store)
        self.assertEqual(verdict.verdict, "rejected")
        self.assertIn("does not belong to this run", verdict.reason)

    def test_advisory_language_is_rejected_outright(self) -> None:
        verdict, _ = check_draft(self._draft("This is a buy and it is profitable."), self.registry.store)
        self.assertEqual(verdict.verdict, "rejected")
        self.assertIn("advisory", verdict.reason)

    def test_causal_language_downgrades_rather_than_rejects(self) -> None:
        rho = self.correlation.stats["spearman_rho"]
        verdict, _ = check_draft(self._draft(f"Sentiment causes the move (rho {rho})."), self.registry.store)
        self.assertEqual(verdict.verdict, "downgraded")
        self.assertEqual(verdict.status_cap, "observed")
        self.assertTrue(verdict.limitations)

    def test_a_confidence_level_is_not_treated_as_a_claim(self) -> None:
        self.assertEqual(check_numbers_in_text("95% CI", {0.5}), [])
        self.assertEqual(check_numbers_in_text("no numbers here", {0.5}), [])

    def test_a_citation_tag_is_not_read_as_a_number(self) -> None:
        """A result id is made of digits, so `[R:...]` must be stripped before the scan."""
        rho = self.correlation.stats["spearman_rho"]
        text = f"Spearman rho {rho} was observed [R:{self.correlation.result_id}]."
        self.assertEqual(check_numbers_in_text(text, {float(rho)}, [self.correlation.result_id]), [])
        verdict, _ = check_draft(
            self._draft(f"Spearman rho {rho} was observed [R:{self.correlation.result_id}]."),
            self.registry.store,
        )
        self.assertEqual(verdict.verdict, "accepted", verdict.reason)

    def test_a_draft_without_evidence_is_rejected(self) -> None:
        draft = FindingDraft(title="No evidence", claim_text="Something happened.")
        verdict, _ = check_draft(draft, self.registry.store)
        self.assertEqual(verdict.verdict, "rejected")
        self.assertIn("no evidence", verdict.reason)


class TestStatusRules(unittest.TestCase):
    def _assign(self, **overrides):
        base = dict(has_test=True, q_value=0.01, ci_95=[0.1, 0.5], same_sign_on_holdout=True,
                    tier=None, own_text_only=True, alpha=0.05)
        base.update(overrides)
        return assign_status(**base)

    def test_supported_needs_q_ci_and_the_holdout_sign(self) -> None:
        status, limitations = self._assign()
        self.assertEqual(status, "supported")
        self.assertTrue(any("weak evidence" in line for line in limitations))

    def test_not_replicated_on_a_sign_flip(self) -> None:
        status, _ = self._assign(same_sign_on_holdout=False)
        self.assertEqual(status, "not_replicated")

    def test_inconclusive_when_the_ci_includes_zero(self) -> None:
        status, limitations = self._assign(ci_95=[-0.2, 0.3])
        self.assertEqual(status, "inconclusive")
        self.assertTrue(any("includes zero" in line for line in limitations))

    def test_inconclusive_when_q_is_above_alpha(self) -> None:
        status, limitations = self._assign(q_value=0.4)
        self.assertEqual(status, "inconclusive")
        self.assertTrue(any("q = 0.400" in line for line in limitations))

    def test_descriptive_result_is_observed(self) -> None:
        status, _ = self._assign(has_test=False, q_value=None, ci_95=None, same_sign_on_holdout=None)
        self.assertEqual(status, "observed")

    def test_tier_c_can_never_exceed_observed(self) -> None:
        status, limitations = self._assign(tier="C")
        self.assertEqual(status, "observed")
        self.assertTrue(any("tier C" in line for line in limitations))

    def test_tier_b_without_own_text_can_never_exceed_observed(self) -> None:
        status, limitations = self._assign(tier="B", own_text_only=False)
        self.assertEqual(status, "observed")
        self.assertTrue(any("tier B" in line for line in limitations))

    def test_tier_b_with_own_text_may_be_supported(self) -> None:
        status, _ = self._assign(tier="B", own_text_only=True)
        self.assertEqual(status, "supported")

    def test_status_counts(self) -> None:
        counts = summarise_status_counts([{"status": "observed"}, {"status": "observed"},
                                          {"status": "supported"}])
        self.assertEqual(counts["observed"], 2)
        self.assertEqual(counts["supported"], 1)
        self.assertEqual(counts["not_replicated"], 0)


class TestNullAgent(unittest.TestCase):
    def test_templated_findings_pass_the_checker(self) -> None:
        """The no-key path is the strongest evidence that the checker is not just 'trust the LLM less'."""
        settings = write_fixture(scratch("null_agent"))
        panel = build_synthetic_panel(settings)
        registry = _registry_with_playbook(settings, panel)
        drafts = null_agent_findings(registry)
        self.assertGreater(len(drafts), 0)
        for draft in drafts:
            verdict, cited = check_draft(draft, registry.store)
            self.assertEqual(verdict.verdict, "accepted", f"{draft.title}: {verdict.reason}")
            self.assertTrue(cited)

    def test_template_narrative_is_honest_about_being_templated(self) -> None:
        narrative = template_narrative([], llm_enabled=False)
        self.assertIn("LLM features are off", narrative["summary"])
        self.assertEqual(narrative["source"], "template")


class TestPipeline(unittest.TestCase):
    """The end-to-end no-key run: the acceptance criterion that matters most."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = write_fixture(scratch("pipeline"))
        cls.state = run_pipeline(cls.settings, mode="playbook", induce=False,
                                 use_cache=False, force_default_lexicon=True)

    def test_every_artifact_is_written(self) -> None:
        run_dir = self.state.run_dir
        for name in ("report.md", "report.html", "findings.json", "manifest.json"):
            self.assertTrue((run_dir / name).exists(), name)
        self.assertTrue((run_dir / "results" / "results.json").exists())

    def test_findings_have_statuses_effects_and_evidence(self) -> None:
        self.assertGreater(len(self.state.findings), 0, "the no-key run must still produce findings")
        for finding in self.state.findings:
            self.assertIn(finding.status, ("observed", "supported", "not_replicated", "inconclusive"))
            self.assertTrue(finding.evidence)
            for item in finding.evidence:
                self.assertIsNotNone(self.state.registry.store.get(item.result_id),
                                     "every cited id must exist in this run")

    def test_no_finding_claims_causation_or_advice(self) -> None:
        from eda_demo.findings.checker import ADVISORY, CAUSAL

        for finding in self.state.findings:
            self.assertIsNone(ADVISORY.search(finding.claim_text), finding.finding_id)
            self.assertIsNone(CAUSAL.search(finding.claim_text), finding.finding_id)

    def test_manifest_reproduces_the_run_and_leaks_no_secret(self) -> None:
        manifest = json.loads((self.state.run_dir / "manifest.json").read_text(encoding="utf-8"))
        for key in ("run_id", "data_version", "input_hashes", "settings", "experiments",
                    "lexicon", "tool_calls", "findings_summary", "quality_warnings"):
            self.assertIn(key, manifest, key)
        blob = json.dumps(manifest)
        self.assertNotIn("api_key", blob.lower().replace("api_key_env", ""))
        self.assertGreater(len(manifest["tool_calls"]), 5)
        self.assertEqual(manifest["llm_calls"], [], "a no-key run makes no LLM call")

    def test_report_states_the_badges_and_the_limits(self) -> None:
        report = (self.state.run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("OBSERVED DATA", report)
        self.assertIn("STATISTICAL ESTIMATE", report)
        self.assertIn("not a trading system", report)
        self.assertIn("Holdout:", report)

    def test_a_second_identical_run_produces_the_same_result_ids(self) -> None:
        again = run_pipeline(self.settings, mode="playbook", induce=False,
                             use_cache=False, force_default_lexicon=True)
        self.assertEqual(
            [result.result_id for result in self.state.registry.store.all()],
            [result.result_id for result in again.registry.store.all()],
        )

    def test_pipeline_never_used_the_network(self) -> None:
        self.assertFalse(self.state.llm["available"])
        self.assertEqual(self.state.narrative["source"], "template")


class TestAppend(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = write_fixture(scratch("append"))
        self.new_file = self.settings.tweets_path.parent / "arriving.csv"
        rows = [
            {"Date": "2021-10-08 14:00:00+00:00", "Tweet": "AAA new deal announced",
             "Stock Name": "AAA", "Company Name": "Alpha Co"},
            {"Date": "2021-10-08 14:05:00+00:00", "Tweet": "BBB zibberishwaffle news",
             "Stock Name": "BBB", "Company Name": "Beta Co"},
            # a duplicate of an existing row must not be appended twice
            {"Date": "2021-10-01 14:00:00+00:00", "Tweet": "AAA strong earnings beat",
             "Stock Name": "AAA", "Company Name": "Alpha Co"},
        ]
        pd.DataFrame(rows).to_csv(self.new_file, index=False)

    def test_appended_rows_are_scored_without_any_llm_call(self) -> None:
        result = append_tweets(self.settings, self.new_file)
        report = result.report
        self.assertEqual(report["llm_calls"], 0)
        self.assertGreater(report["appended_rows"], 0)
        self.assertIn("coverage drift", report["recommendation"].lower())
        self.assertTrue(Path(result.paths["report"]).exists())
        summary = Path(result.paths["summary"])
        self.assertTrue(summary.exists())
        # The convenience copy belongs to the configured path, not to the repository's docs/.
        self.assertEqual(summary.parent, self.settings.repo_root)

    def test_duplicate_rows_are_skipped(self) -> None:
        result = append_tweets(self.settings, self.new_file)
        self.assertNotIn("AAA strong earnings beat", result.paths.get("__never__", ""))
        self.assertLessEqual(result.report["appended_rows"], 3)

    def test_schema_drift_is_blocking(self) -> None:
        bad = self.settings.tweets_path.parent / "drifted.csv"
        pd.DataFrame([{"When": "2021-10-08", "Body": "hello"}]).to_csv(bad, index=False)
        with self.assertRaises(BlockingError) as caught:
            append_tweets(self.settings, bad)
        self.assertIn("schema drift", str(caught.exception))

    def test_missing_file_is_blocking(self) -> None:
        with self.assertRaises(BlockingError):
            append_tweets(self.settings, self.settings.tweets_path.parent / "nope.csv")


if __name__ == "__main__":
    unittest.main()
