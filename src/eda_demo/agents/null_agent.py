"""The no-key path: findings selected by rule, with templated prose.

Design decision embodied here: **the no-key mode is the same pipeline, not a degraded one.** It
selects findings by a written-down rule (largest absolute effects, the coverage baseline, the
biggest spike day) and writes text from the cited statistics. Every number in the template comes
from a `ToolResult`, so the same claim checker accepts it — which is the strongest available
evidence that the checker is not just "trust the LLM less".

This is also the fallback when an agent call fails: the report degrades to these cards and says so.
"""

from __future__ import annotations

from typing import Any

from ..findings.schema import Evidence, FindingDraft, FindingScope
from ..tools.registry import ToolRegistry
from ..tools.schema import ToolResult

# Rendered with the same rounding as the stats they quote, so the claim checker can match them.
_FMT = "{:.4g}"


def _scope_from(result: ToolResult) -> FindingScope:
    """Build a finding scope from a result's own parameters."""
    ticker = result.params.get("ticker")
    scope = result.params.get("scope")
    tickers = [ticker] if ticker else ([str(scope)] if scope else ["pooled"])
    period = str(result.params.get("period", result.period))
    return FindingScope(tickers=tickers, period=period, tier=None)


def _draft(title: str, claim: str, results: list[ToolResult], rationale: str,
           limitations: list[str] | None = None,
           alternatives: list[str] | None = None) -> FindingDraft:
    """Assemble a draft from one or more results, citing them all as evidence."""
    return FindingDraft(
        title=title,
        claim_text=claim,
        evidence=[Evidence(result_id=result.result_id, role="primary" if index == 0 else "supporting")
                  for index, result in enumerate(results)],
        scope=_scope_from(results[0]),
        limitations=limitations or [],
        alternative_explanations=alternatives or [],
        rationale=rationale,
    )


def null_agent_findings(registry: ToolRegistry, limit: int = 8) -> list[FindingDraft]:
    """Select and template findings with no model call. Returns validated `FindingDraft`s.

    The selection rule is deliberately dumb and readable: biggest absolute bucket spread, biggest
    absolute correlation, the coverage baseline, and the largest spike count. Nothing here notices
    anything a human could not have noticed from the same table, which is the honest claim for a
    templated agent.
    """
    store = registry.store
    drafts: list[FindingDraft] = []

    coverage = _first(store.all(), "coverage_by_ticker")
    if coverage is not None:
        counts = coverage.stats.get("tier_counts", {})
        drafts.append(
            _draft(
                "Ticker coverage and the tier split",
                (
                    f"{coverage.stats.get('tickers')} tickers, {coverage.stats.get('tweets')} tweets and "
                    f"{coverage.stats.get('own_texts')} own-text tweets; tier A {counts.get('A')}, "
                    f"tier B {counts.get('B')}, tier C {counts.get('C')}."
                ),
                [coverage],
                "baseline coverage card: always present, so the report has an anchoring number",
                limitations=[
                    "raw tweet counts are a scraping artefact for the 5 tier-C tickers",
                    "tier assignment is per dataset slice and must not be compared across slices",
                ],
            )
        )

    buckets = [result for result in store.all() if result.tool == "bucket_forward_returns"
               and result.stats.get("spread_top_minus_bottom") is not None]
    if buckets:
        best = max(buckets, key=lambda result: abs(float(result.stats["spread_top_minus_bottom"])))
        stats = best.stats
        drafts.append(
            _draft(
                f"Sentiment bucket spread at horizon {best.params.get('horizon')}",
                (
                    f"{stats['feature']} buckets against {stats['target']}: top-minus-bottom mean "
                    f"{_FMT.format(float(stats['spread_top_minus_bottom']))} with 95% CI "
                    f"{stats['spread_ci_95']} against an unconditional mean of "
                    f"{_FMT.format(float(stats['unconditional_mean']))} over {stats['n_units']} "
                    f"ticker-sessions in the {best.period} period."
                ),
                [best],
                "selected because it has the largest absolute bucket spread of the playbook's horizons",
                limitations=[
                    "overlapping forward returns at h>1; the CI uses session blocks",
                    "extreme buckets hold the fewest observations, so the largest spread is also the noisiest",
                ],
                alternatives=[
                    "the spread can come from a handful of event sessions rather than from sentiment",
                    "promo-dominated days can sit systematically in one bucket",
                ],
            )
        )

    correlations = [result for result in store.all() if result.tool == "correlate"
                    and result.stats.get("spearman_rho") is not None]
    if correlations:
        best = max(correlations, key=lambda result: abs(float(result.stats["spearman_rho"])))
        stats = best.stats
        drafts.append(
            _draft(
                f"{stats['y']} vs {stats['x']}",
                (
                    f"Spearman rho {_FMT.format(float(stats['spearman_rho']))} between {stats['x']} and "
                    f"{stats['y']} at lag {stats['lag_sessions']}, p {_FMT.format(float(stats['p_value'])) if stats.get('p_value') is not None else 'n/a'}, "
                    f"95% CI {stats['ci_95']} over {stats['n_units']} ticker-sessions."
                ),
                [best],
                "selected because it has the largest absolute correlation in the playbook",
                limitations=[
                    "rank correlation on ticker-sessions: a co-movement, not a cause",
                    "the CI is a session-block bootstrap because daily series are autocorrelated",
                ],
                alternatives=[
                    "both series may be driven by the same news day rather than by each other",
                    "low-coverage sessions are excluded, which can remove the quietest days",
                ],
            )
        )

    spikes = _first(store.all(), "spike_days")
    if spikes is not None and spikes.stats.get("n_spikes"):
        by_ticker = spikes.stats.get("spikes_by_ticker", {})
        top = max(by_ticker.items(), key=lambda pair: pair[1]) if by_ticker else ("n/a", 0)
        drafts.append(
            _draft(
                f"Attention spikes in {spikes.stats.get('metric')}",
                (
                    f"{spikes.stats.get('n_spikes')} ticker-sessions exceed a robust z of "
                    f"{spikes.stats.get('z_threshold')} ({_FMT.format(float(spikes.stats.get('spike_share', 0)))} "
                    f"of units); the most affected ticker is {top[0]} with {top[1]}."
                ),
                [spikes],
                "largest spike card from the playbook's anomaly tool",
                limitations=[
                    "a robust-z outlier is not a significant event",
                    "calendar-wide news days produce clusters of spikes across tickers at once",
                ],
                alternatives=[
                    "promo blasts inflate raw attention without reflecting interest",
                    "the tier-C tickers' spikes are artefacts of cross-posted text",
                ],
            )
        )

    text = _first(store.all(), "text_stats")
    if text is not None:
        stats = text.stats
        drafts.append(
            _draft(
                "Text hygiene: promo, URLs and fan-out",
                (
                    f"promo share {_FMT.format(float(stats['promo_share']))}, URL share "
                    f"{_FMT.format(float(stats['url_share']))}, reply share "
                    f"{_FMT.format(float(stats['reply_share']))} over {stats['n_tweets']} tweets; "
                    f"fan-out affects {_FMT.format(float(stats['fanout_row_share']))} of all rows."
                ),
                [text],
                "context card: the majority pattern in this corpus is promotional, not opinion",
                limitations=[
                    "raw rows, so this is not an independent sample",
                    "there is no author field, so bot and unique-user analysis is impossible, not deferred",
                ],
            )
        )

    return drafts[:limit]


def _first(results: list[ToolResult], tool: str) -> ToolResult | None:
    """Return the first result produced by `tool`, or None when it is absent or failed."""
    return next((result for result in results if result.tool == tool), None)


# --- templated prose (also the Presenter's fallback) -------------------------


def template_narrative(findings: list[dict[str, Any]], llm_enabled: bool) -> dict[str, Any]:
    """Write the executive summary and per-finding paragraphs from the findings themselves.

    Used when the LLM is off and when a presenter call fails or is rejected by the claim checker,
    so the report always has a narrative that says exactly what the findings say.
    """
    if not findings:
        summary = (
            "No finding was produced by this run. The baseline analyses ran and their results are "
            "listed below; nothing in them passed the rules for a finding, and that is reported "
            "rather than filled in."
        )
        if not llm_enabled:
            summary += (
                " LLM features are off for this run, so this summary is templated from the tool "
                "results rather than written by a model."
            )
    else:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding["status"]] = counts.get(finding["status"], 0) + 1
        breakdown = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        strongest = findings[0]
        summary = (
            f"{len(findings)} finding(s) from this run: {breakdown}. The first card is "
            f"\"{strongest['title']}\" with status {strongest['status']}; the wording of every claim "
            "is bounded by the statistics its evidence cites."
        )
        if not llm_enabled:
            summary += (
                " LLM features are off for this run, so this summary and the paragraphs below are "
                "templated from the tool results rather than written by a model."
            )
    paragraphs = {
        finding["finding_id"]: (
            f"{finding['title']}. Status {finding['status']}. "
            + " ".join(finding.get("limitations", [])[:2])
        ).strip()
        for finding in findings
    }
    return {
        "summary": summary,
        "paragraphs": paragraphs,
        "what_to_look_at_next": [
            "drill into the ticker-session behind the largest effect and read its tweets (sample_tweets)",
            "re-run the strongest test per ticker to see whether one ticker drives the pooled number",
            "compare the discovery and holdout periods for the tests that pass, before reading them as stable",
        ],
        "source": "template",
    }
