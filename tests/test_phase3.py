"""Phase 3 tests: the tool registry, the catalog, the playbook and the two renderers.

Two contracts get first-class tests here because the rest of the system leans on them:

- every tool result carries a stable `result_id` and a filled `n_units`, since that is what a
  finding cites;
- a `ChartSpec` produced by any tool is accepted by *both* renderers, which is the only way to
  know that the spec has no rendering concepts leaking into it.
"""

from __future__ import annotations

import unittest

from tests.fixtures import build_synthetic_panel, scratch, write_fixture

from eda_demo.report.chart_menu import ChartMenu, _parse_numbers
from eda_demo.report.build_report import ReportContext, render_markdown
from eda_demo.report.render_ascii import render_ascii
from eda_demo.report.render_png import render_png
from eda_demo.tools import build_registry
from eda_demo.tools.playbook import run_playbook
from eda_demo.tools.registry import ToolError


class ToolTestCase(unittest.TestCase):
    """Shared setup: one synthetic panel and one registry over it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = write_fixture(scratch("tools"))
        cls.panel = build_synthetic_panel(cls.settings)

    def registry(self):
        return build_registry(
            self.settings, self.panel.panel, self.panel.tweets, self.panel.data_version,
            self.panel.tiers, self.panel.calendar,
        )


class TestRegistry(ToolTestCase):
    def test_unknown_tool_is_an_error_that_lists_the_registry(self) -> None:
        registry = self.registry()
        with self.assertRaises(ToolError) as caught:
            registry.call("no_such_tool", {})
        self.assertIn("coverage_by_ticker", str(caught.exception))

    def test_invalid_parameters_are_rejected_rather_than_ignored(self) -> None:
        registry = self.registry()
        with self.assertRaises(ToolError):
            registry.call("distribution", {"column": "sent_net", "surprise": 1})
        with self.assertRaises(ToolError):
            registry.call("distribution", {"column": "not_a_column"})

    def test_result_id_is_stable_and_parameters_are_part_of_it(self) -> None:
        first = self.registry().call("distribution", {"column": "sent_net", "period": "discovery"})
        second = self.registry().call("distribution", {"column": "sent_net", "period": "discovery"})
        other = self.registry().call("distribution", {"column": "n_texts_own", "period": "discovery"})
        self.assertEqual(first.result_id, second.result_id)
        self.assertNotEqual(first.result_id, other.result_id)

    def test_period_argument_overrides_the_parameter(self) -> None:
        """The regression guard for a holdout check that silently re-ran discovery.

        The tool reads `params.period`, so a call-level period that did not overwrite the parameter
        would compute one period while labelling the result the other — and every replication check
        would be a false pass.
        """
        registry = self.registry()
        discovery = registry.call("correlate", {"x": "sent_net", "y": "fwd_ret_1", "period": "discovery"})
        holdout = registry.call("correlate", {"x": "sent_net", "y": "fwd_ret_1", "period": "discovery"},
                                period="holdout")
        self.assertEqual(holdout.period, "holdout")
        self.assertEqual(holdout.params["period"], "holdout")
        self.assertNotEqual(discovery.result_id, holdout.result_id)

    def test_every_result_carries_caveats_and_a_data_version(self) -> None:
        registry = self.registry()
        result = registry.call("distribution", {"column": "sent_net"})
        self.assertTrue(any("n_units" in caveat for caveat in result.caveats))
        self.assertTrue(any(self.panel.data_version in caveat for caveat in result.caveats))

    def test_a_failing_tool_is_recorded_and_does_not_hide_the_error(self) -> None:
        registry = self.registry()
        with self.assertRaises(ToolError):
            registry.call("sample_tweets", {"ticker": "AAA", "date": "1999-01-01"})
        self.assertEqual(registry.errors, [], "a caller error is not a tool failure")


class TestCatalog(ToolTestCase):
    def setUp(self) -> None:
        self.registry = self.registry()

    def test_coverage_reports_tiers_and_both_attention_columns(self) -> None:
        result = self.registry.call("coverage_by_ticker", {"period": "full"}, period="full")
        self.assertEqual(result.stats["tickers"], 3)
        self.assertIn("n_texts_own", result.table[0])
        self.assertIn("n_texts_shared", result.table[0])
        self.assertEqual(result.stats["tier_counts"]["A"], 2)

    def test_correlate_returns_rho_and_a_bootstrap_ci(self) -> None:
        result = self.registry.call("correlate", {"x": "sent_net", "y": "fwd_ret_1"})
        self.assertIn("spearman_rho", result.stats)
        self.assertEqual(len(result.stats["ci_95"]), 2)
        self.assertGreater(result.n_units, 0)

    def test_bucket_forward_returns_reports_every_bucket_and_the_baseline(self) -> None:
        result = self.registry.call("bucket_forward_returns",
                                    {"feature": "sent_net", "horizon": 1, "buckets": 2})
        self.assertEqual([row["bucket"] for row in result.table], ["q1", "q2"])
        self.assertIsNotNone(result.stats["unconditional_mean"])
        self.assertEqual(len(result.stats["spread_ci_95"]), 2)

    def test_bucket_tool_says_so_when_the_feature_has_too_few_distinct_values(self) -> None:
        """A count-like feature collapses the quantile edges; the tool explains instead of crashing."""
        result = self.registry.call("bucket_forward_returns",
                                    {"feature": "n_texts_own", "horizon": 1, "buckets": 5})
        self.assertIn("note", result.stats)
        self.assertIsNone(result.chart)

    def test_bucket_tool_refuses_a_horizon_that_is_not_configured(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.call("bucket_forward_returns", {"feature": "sent_net", "horizon": 5})

    def test_spike_days_reports_robust_z_and_a_capped_table(self) -> None:
        result = self.registry.call("spike_days", {"metric": "n_tweets", "z_threshold": 1.5, "scope": "pooled"})
        self.assertIn("n_spikes", result.stats)
        self.assertLessEqual(len(result.table or []), 50)

    def test_sample_tweets_returns_real_text_and_refuses_a_bad_date(self) -> None:
        result = self.registry.call("sample_tweets", {"ticker": "AAA", "date": "2021-10-01", "n": 3})
        self.assertGreaterEqual(result.stats["rows_in_cell"], 1)
        self.assertTrue(result.table[0]["text"])
        with self.assertRaises(ToolError):
            self.registry.call("sample_tweets", {"ticker": "AAA", "date": "2021-10-02"})

    def test_sentiment_and_attention_dual_axis_tools_produce_charts(self) -> None:
        for tool, params in (("attention_vs_price", {"ticker": "AAA"}),
                             ("sentiment_vs_price", {"ticker": "AAA"})):
            result = self.registry.call(tool, params)
            self.assertIsNotNone(result.chart, tool)
            self.assertEqual(result.chart.kind, "dual_line")
            self.assertEqual(result.chart.source_result_id, result.result_id)

    def test_corr_matrix_levels(self) -> None:
        for level in ("pooled", "per_ticker"):
            result = self.registry.call("corr_matrix", {"metrics": ["sent_net", "n_texts_own"], "level": level})
            self.assertEqual(result.chart.kind, "heatmap")
            self.assertIn("sent_net", result.stats["matrix"])

    def test_holdout_check_actually_changes_the_period(self) -> None:
        discovery = self.registry.call("correlate", {"x": "n_texts_own", "y": "fwd_ret_1"})
        check = self.registry.call("holdout_check", {"result_id": discovery.result_id})
        self.assertEqual(check.stats["tested_result_id"], discovery.result_id)
        self.assertIn("same_sign", check.stats)
        child = self.registry.store.get(check.stats["holdout_result_id"])
        self.assertIsNotNone(child, "the holdout re-run must be stored as its own result")
        self.assertEqual(child.period, "holdout")
        self.assertEqual(child.parent_result_ids, [discovery.result_id])

    def test_holdout_check_refuses_an_unknown_result_id(self) -> None:
        with self.assertRaises(ToolError):
            self.registry.call("holdout_check", {"result_id": "deadbeef"})

    def test_adjust_pvalues_returns_a_q_value_per_test(self) -> None:
        first = self.registry.call("correlate", {"x": "n_texts_own", "y": "fwd_ret_1"})
        second = self.registry.call("correlate", {"x": "sent_net", "y": "fwd_ret_1"})
        adjusted = self.registry.call("adjust_pvalues", {"result_ids": [first.result_id, second.result_id]})
        self.assertEqual(adjusted.stats["n_tests"], 2)
        for row in adjusted.table:
            self.assertIn("q_value", row)
        self.assertLessEqual(min(row["q_value"] for row in adjusted.table),
                             max(row["p_value"] for row in adjusted.table))


class TestCharts(ToolTestCase):
    def test_every_tool_with_a_chart_survives_both_renderers(self) -> None:
        registry = self.registry()
        calls = [
            ("coverage_by_ticker", {"period": "full"}, "full"),
            ("distribution", {"column": "sent_net"}, "discovery"),
            ("text_stats", {"period": "discovery"}, "discovery"),
            ("timeseries", {"metric": "sent_net"}, "discovery"),
            ("attention_vs_price", {"ticker": "AAA"}, "discovery"),
            ("sentiment_vs_price", {"ticker": "AAA"}, "discovery"),
            ("correlate", {"x": "sent_net", "y": "fwd_ret_1"}, "discovery"),
            ("bucket_forward_returns", {"feature": "sent_net", "buckets": 2}, "discovery"),
            ("corr_matrix", {"metrics": ["sent_net", "n_texts_own"]}, "discovery"),
            ("spike_days", {"metric": "n_tweets", "z_threshold": 1.5, "scope": "ticker", "ticker": "AAA"}, "discovery"),
        ]
        tmp = scratch("charts")
        for tool, params, period in calls:
            result = registry.call(tool, params, period=period)
            self.assertIsNotNone(result.chart, tool)
            text = render_ascii(result.chart)
            self.assertIsInstance(text, str)
            self.assertNotIn("no ascii renderer", text, tool)
            path = render_png(result.chart, tmp / f"{result.result_id}.png")
            self.assertTrue(path.exists(), tool)
            self.assertGreater(path.stat().st_size, 0, tool)

    def test_caption_always_names_period_units_and_result(self) -> None:
        result = self.registry().call("timeseries", {"metric": "sent_net"})
        caption = result.chart.caption
        self.assertIn("period=discovery", caption)
        self.assertIn("n_units=", caption)
        self.assertIn(result.result_id, caption)

    def test_menu_lists_only_chart_bearing_results_and_parses_commands(self) -> None:
        registry = self.registry()
        registry.call("timeseries", {"metric": "sent_net"})
        registry.call("sample_tweets", {"ticker": "AAA", "date": "2021-10-01"})
        menu = ChartMenu(registry.store.all(), scratch("menu"), self.settings)
        self.assertEqual(len(menu.results), 1)
        self.assertIn("Available charts", menu.listing())
        self.assertEqual(_parse_numbers("3,5"), [3, 5])
        self.assertEqual(_parse_numbers("ascii 2"), [2])
        self.assertFalse(menu.handle("q"), "q must end the loop")
        self.assertTrue(menu.handle("list"))

    def test_ascii_render_of_every_kind_never_raises(self) -> None:
        from eda_demo.tools.schema import ChartSpec

        for kind in ("line", "bar", "bar_error", "hist", "scatter", "heatmap", "dual_line"):
            spec = ChartSpec(kind=kind, title="t", x={"label": "x", "values": ["a", "b"]},
                             series=[{"label": "s", "values": [1.0, 2.0]}, {"label": "e", "values": [0.1, 0.2]}])
            self.assertIsInstance(render_ascii(spec), str)

    def test_ascii_falls_back_to_ascii_characters_when_the_console_cannot_encode_blocks(self) -> None:
        """A cp1252 console must not turn a chart into a UnicodeEncodeError."""
        import io
        import sys

        from eda_demo.report import render_ascii as ascii_renderer

        class Cp1252Stream(io.StringIO):
            encoding = "cp1252"

        original = sys.stdout
        sys.stdout = Cp1252Stream()  # type: ignore[assignment]
        try:
            line = ascii_renderer.sparkline([1.0, 5.0, 3.0])
            bars = ascii_renderer.horizontal_bars(["a"], [1.0])
        finally:
            sys.stdout = original
        self.assertTrue(all(ord(character) < 128 for character in line), line)
        self.assertTrue(all(ord(character) < 128 for character in bars), bars)


class TestPlaybook(ToolTestCase):
    def test_playbook_runs_and_covers_both_families(self) -> None:
        registry = build_registry(
            self.settings, self.panel.panel, self.panel.tweets, self.panel.data_version,
            self.panel.tiers, self.panel.calendar,
        )
        outcome = run_playbook(registry)
        tools = {entry["tool"] for entry in outcome.digest}
        self.assertIn("correlate", tools)
        self.assertIn("bucket_forward_returns", tools)
        self.assertIn("attention_vs_price", tools)
        # Every pooled call is accompanied by the TSLA-excluded variant somewhere in the run.
        scopes = {result.params.get("scope") for result in registry.store.all()}
        self.assertIn("pooled_ex_TSLA", scopes)

    def test_a_call_that_cannot_run_is_recorded_not_raised(self) -> None:
        """A thin dataset must still produce a report that says which calls were skipped."""
        registry = build_registry(
            self.settings, self.panel.panel, self.panel.tweets, self.panel.data_version,
            self.panel.tiers, self.panel.calendar,
        )
        outcome = run_playbook(registry)
        self.assertIsInstance(outcome.failures, list)
        for failure in outcome.failures:
            self.assertIn("error", failure)


class TestReport(ToolTestCase):
    def test_markdown_keeps_the_three_badges_and_cites_result_ids(self) -> None:
        registry = self.registry()
        registry.call("coverage_by_ticker", {"period": "full"}, period="full")
        context = ReportContext(
            run_id="20260101T000000Z_synthetic", dataset="synthetic",
            data_version=self.panel.data_version, settings=self.settings, meta=self.panel.meta,
            results=registry.store.all(),
            findings=[{
                "finding_id": "F-001", "title": "A finding", "status": "observed",
                "claim_text": "text", "evidence": [{"result_id": registry.store.order[0], "role": "primary"}],
                "effect": {"metric": None, "value": None, "ci_95": None, "n_units": 3, "q_value": None},
                "scope": {"tickers": ["pooled"], "period": "discovery", "tier": None},
                "limitations": [], "alternative_explanations": [], "origin": "null_agent",
            }],
        )
        markdown = render_markdown(context)
        self.assertIn("OBSERVED DATA", markdown)
        self.assertIn(registry.store.order[0], markdown)
        self.assertIn("Structural limits", markdown)
        self.assertIn("not a trading system", markdown)


if __name__ == "__main__":
    unittest.main()
