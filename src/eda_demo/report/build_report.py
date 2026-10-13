"""Build `report.md` (primary), `report.html` (convenience export) and the persisted results.

Design decision embodied here: **Markdown is the durable format and the terminal is the reading
surface** (SPEC.md 14). This module therefore holds one structured description of a run and writes
it three ways: a terminal block, `report.md`, and a self-contained `report.html` with the figures
embedded as data URIs.

Second decision: **the claim checker's rejections appear in the report.** An agent paragraph that
was thrown out is listed with its reason, because a report that quietly drops its own failed
verification is exactly the kind of artifact this demo exists to argue against.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, Settings
from ..features.feature_book import LIMITS
from ..tools.schema import ToolResult

# The three badges the presentation must keep distinguishable everywhere (SPEC.md 4.12).
BADGES = {
    "observed": "OBSERVED DATA",
    "supported": "STATISTICAL ESTIMATE",
    "not_replicated": "STATISTICAL ESTIMATE",
    "inconclusive": "STATISTICAL ESTIMATE",
    "interpretation": "AGENT INTERPRETATION",
}


@dataclass
class ReportContext:
    """Everything a report needs, already computed. No stage is re-run to build a report."""

    run_id: str
    dataset: str
    data_version: str
    settings: Settings
    meta: dict[str, Any]
    results: list[ToolResult] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    narrative: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)
    playbook_failures: list[dict[str, Any]] = field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = field(default_factory=list)
    lexicon: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _bullet_list(items: list[str], empty: str = "none") -> list[str]:
    """Render a Markdown bullet list, or a single 'none' line when it is empty."""
    return [f"- {item}" for item in items] or [f"- {empty}"]


def render_markdown(context: ReportContext) -> str:
    """Render the whole report as Markdown text."""
    meta = context.meta
    lines: list[str] = []
    add = lines.append

    add(f"# EDA report — {context.dataset}")
    add("")
    add(f"- Run: `{context.run_id}`  |  `data_version`: `{context.data_version}`")
    add(f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    add(f"- Lexicon: `{context.lexicon.get('lexicon_id', '?')}` ({context.lexicon.get('source', '?')}, {context.lexicon.get('terms', 0)} terms)")
    llm = context.llm or {}
    add(
        f"- LLM: {'enabled' if llm.get('enabled') else 'disabled'} "
        f"(model={llm.get('model') or 'unset'}, host={llm.get('host') or 'unset'}, mode={llm.get('mode_in_use', 'n/a')})"
        + ("" if llm.get("available") else " — this run used the templated no-key path")
    )
    add("")
    add("**Badges used throughout:** `[OBSERVED DATA]` a measured fact · `[STATISTICAL ESTIMATE]` a test "
        "with a CI · `[AGENT INTERPRETATION]` prose by a model. Every number carries the "
        "`result_id` of the tool call that produced it.")
    add("")

    add("## 1. Overview")
    add("")
    add(f"- Sessions: {meta.get('sessions')}  |  ticker-sessions: {meta.get('n_units'):,}")
    split = meta.get("split", {})
    add(f"- Discovery: {split.get('discovery_sessions')} sessions ({split.get('discovery_start')} -> {split.get('discovery_end')})")
    add(f"- Holdout: {split.get('holdout_sessions')} sessions ({split.get('holdout_start')} -> {split.get('holdout_end')})")
    canonical = meta.get("canonical_report", {})
    if canonical:
        add(f"- Tweets in window: {canonical.get('rows_in_window'):,} of {canonical.get('rows_in'):,} read "
            f"({canonical.get('rows_out_of_window'):,} excluded for falling outside the last session)")
        add(f"- Roll forward to the next session: {canonical.get('rolled_share', 0):.1%}")
        add(f"- Cross-posted texts: {canonical.get('shared_texts'):,} covering "
            f"{canonical.get('rows_in_window', 0) - canonical.get('own_text_rows', 0):,} in-window rows")
        clean = canonical.get("clean", {})
        add(f"- Promo-shaped rows: {clean.get('promo_rows'):,} ({clean.get('promo_rows', 0) / max(1, clean.get('rows', 1)):.1%})")
    gates = [
        ("units with a sentiment score", meta.get("units_with_sentiment")),
        ("units below min_tweets_per_day", meta.get("units_low_coverage")),
        ("units meeting the minimum on all text (diagnostic)", meta.get("units_meeting_min_on_all_text")),
        ("units excluded from inference", meta.get("units_excluded_from_tests")),
    ]
    for label, value in gates:
        add(f"- {label}: {value:,}" if isinstance(value, int) else f"- {label}: {value}")
    add("")
    tiers = meta.get("tiers", {}).get("by_ticker", {})
    if tiers:
        add("### Coverage tiers")
        add("")
        add("| Ticker | Tier | Own-text share |")
        add("|---|---|---|")
        shares = {ticker: share for ticker, share in (meta.get("canonical_report", {}).get("own_text_share_per_ticker", {}) or {}).items()}
        for ticker in sorted(tiers):
            add(f"| {ticker} | {tiers[ticker]} | {shares.get(ticker, float('nan')):.2f} |")
        add("")

    if context.narrative:
        add("## 2. Executive summary  [AGENT INTERPRETATION]")
        add("")
        add(context.narrative.get("summary", "(no summary produced)"))
        add("")
        if context.narrative.get("what_to_look_at_next"):
            add("**What to look at next**")
            add("")
            for item in context.narrative["what_to_look_at_next"]:
                add(f"- {item}")
            add("")

    add(f"## 3. Findings ({len(context.findings)})")
    add("")
    if not context.findings:
        add("No finding was produced. That is a result, not a failure: the templated no-key path "
            "selects findings by rule, and an agent may legitimately return none.")
        add("")
    for finding in context.findings:
        status = finding.get("status", "observed")
        badge = BADGES.get(status, "OBSERVED DATA")
        add(f"### {finding.get('finding_id')} · {finding.get('title')}  `[{badge}]`")
        add("")
        effect = finding.get("effect") or {}
        scope = finding.get("scope") or {}
        add(f"- status: **{status}**  |  scope: {scope.get('tickers')} ({scope.get('period')}, tier {scope.get('tier') or 'mixed'})")
        add(
            f"- effect: {effect.get('metric')} = {effect.get('value')} "
            f"(95% CI {effect.get('ci_95')}), n_units = {effect.get('n_units')}, q = {effect.get('q_value')}"
        )
        add(f"- origin: {finding.get('origin')}")
        add("")
        add(finding.get("claim_text", ""))
        add("")
        if finding.get("evidence"):
            add("Evidence: " + ", ".join(f"`{item.get('result_id')}` ({item.get('role')})" for item in finding["evidence"]))
            add("")
        for label in ("limitations", "alternative_explanations"):
            items = finding.get(label) or []
            if items:
                add(f"**{label.replace('_', ' ').capitalize()}**")
                add("")
                add("\n".join(f"- {item}" for item in items))
                add("")

    add(f"## 4. Baseline results ({len(context.results)})")
    add("")
    add("| result_id | tool | period | n_units | n_dropped | key numbers |")
    add("|---|---|---|---:|---:|---|")
    for result in context.results:
        add(
            f"| `{result.result_id}` | {result.tool} | {result.period} | {result.n_units:,} | "
            f"{result.n_dropped:,} | {_key_numbers(result)} |"
        )
    add("")
    if context.playbook_failures:
        add("**Baseline calls that could not run here**")
        add("")
        add("\n".join(f"- `{item['tool']}` {item.get('params')}: {item.get('error')}" for item in context.playbook_failures))
        add("")

    charts = [result for result in context.results if result.chart is not None]
    add(f"## 5. Charts ({len(charts)})")
    add("")
    if not charts:
        add("- none")
    for result in charts:
        chart = result.chart
        assert chart is not None
        add(f"- **{chart.title}** `[{chart.kind}]` — `{result.result_id}`, n_units={result.n_units}")
        add(f"  - caption: {chart.caption}")
    add("")

    if context.rejected_claims:
        add("## 6. Agent output that failed verification")
        add("")
        add("| finding | verdict | reason |")
        add("|---|---|---|")
        for item in context.rejected_claims:
            add(f"| {item.get('finding_id')} | {item.get('verdict')} | {item.get('reason')} |")
        add("")

    add("## 7. Method and reproducibility")
    add("")
    scoring = meta.get("scoring", {})
    add(f"- Sentiment: `{scoring.get('method')}`; VADER available: {scoring.get('vader_available')}; "
        f"label band ±{scoring.get('label_band')}; lexicon weight {scoring.get('lexicon_weight')}")
    add(f"- Negation: a negator within {scoring.get('negation_window')} tokens flips and damps a term by {scoring.get('negation_damp')}")
    add("- Confidence intervals: session-block bootstrap "
        f"({context.settings.experiments.get('bootstrap', {}).get('block_sessions')} sessions, "
        f"{context.settings.experiments.get('bootstrap', {}).get('resamples')} resamples, fixed seed)")
    add("- Multiplicity: " + str(context.settings.experiments.get("multiple_testing", {})))
    add("- Discovery/holdout discipline is enforced in code: the explorer may only read discovery; "
        "`holdout_check` is triggered by the coordinator")
    add(f"- Every `result_id` is `sha256(tool, parameters, period, data_version)[:12]` with "
        f"`data_version = {context.data_version}`")
    add("")

    add("## 8. Structural limits")
    add("")
    add("\n".join(f"- {limit}" for limit in LIMITS))
    add("")
    add("This is not a trading system and nothing here is trading advice. `supported` means "
        "\"passed this demo's checks\" and nothing stronger.")
    add("")

    add("## 9. Data warnings carried into this run")
    add("")
    add("\n".join(_bullet_list(context.warnings)))
    add("")
    return "\n".join(lines)


def _key_numbers(result: ToolResult) -> str:
    """Pick up to four headline scalars from a result's stats for the results table."""
    preferred = (
        "spearman_rho", "p_value", "spread_top_minus_bottom", "unconditional_mean",
        "mean", "median", "n_tweets", "n_spikes", "tickers", "sessions", "same_sign",
    )
    chosen: list[str] = []
    for key in preferred:
        if key in result.stats and isinstance(result.stats[key], (int, float)):
            chosen.append(f"{key}={result.stats[key]}")
        if len(chosen) == 4:
            break
    if not chosen:
        for key, value in result.stats.items():
            if isinstance(value, (int, float)):
                chosen.append(f"{key}={value}")
            if len(chosen) == 3:
                break
    return ", ".join(chosen)


def write_results_json(results: list[ToolResult], run_dir: Path) -> Path:
    """Persist every `ToolResult` (what makes the run re-inspectable without re-running it)."""
    path = run_dir / "results" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([result.model_dump() for result in results], indent=2, default=str), encoding="utf-8")
    return path


def render_html(context: ReportContext, figures_dir: Path | None = None) -> str:
    """Render a self-contained HTML report: the Markdown content plus embedded figures."""
    markdown = render_markdown(context)
    body = _markdown_to_html(markdown)
    gallery = ""
    if figures_dir is not None and figures_dir.exists():
        images = sorted(figures_dir.glob("*.png"))
        if images:
            from .render_png import figure_to_data_uri

            cards = []
            for image in images:
                resolved = next(
                    (result for result in context.results if result.result_id == image.stem), None
                )
                caption = resolved.chart.caption if resolved and resolved.chart else image.stem
                cards.append(
                    f'<figure><img src="{figure_to_data_uri(image)}" alt="{html.escape(image.stem)}"/>'
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
            gallery = "<h2>Figures</h2>" + "".join(cards)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EDA report — {html.escape(context.dataset)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1080px; margin: 2rem auto; padding: 0 1rem; color: #1c1c1c; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
 th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
 th {{ background: #f4f4f6; }}
 code {{ background: #f4f4f6; padding: 0.05rem 0.3rem; border-radius: 3px; }}
 figure {{ margin: 1rem 0; }}
 img {{ max-width: 100%; border: 1px solid #ddd; }}
 figcaption {{ font-size: 0.8rem; color: #666; }}
</style></head>
<body>
{body}
{gallery}
</body></html>
"""


def _markdown_to_html(markdown: str) -> str:
    """Convert the Markdown this module generates into HTML.

    Deliberately a small, total function over *our own* output (headings, bullets, tables,
    paragraphs, inline code, bold) rather than a general Markdown implementation: the point of
    `report.html` is a convenience export of a document we produced ourselves.
    """
    out: list[str] = []
    in_table = False
    in_list = False

    def close_blocks() -> None:
        nonlocal in_table, in_list
        if in_list:
            out.append("</ul>")
            in_list = False
        if in_table:
            out.append("</table>")
            in_table = False

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line:
            close_blocks()
            continue
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if set("".join(cells)) <= {"-", ":", " "}:
                continue
            if not in_table:
                close_blocks()
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in cells) + "</tr>")
            continue
        if line.startswith("- "):
            if not in_list:
                close_blocks()
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        close_blocks()
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        else:
            out.append(f"<p>{_inline(line)}</p>")
    close_blocks()
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape HTML then re-apply the two inline styles the report uses."""
    escaped = html.escape(text)
    parts = escaped.split("`")
    for index in range(1, len(parts), 2):
        parts[index] = f"<code>{parts[index]}</code>"
    joined = "".join(parts)
    while joined.count("**") >= 2:
        joined = joined.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return joined


def write_reports(context: ReportContext, run_dir: Path) -> dict[str, str]:
    """Write `report.md` and `report.html` into `run_dir` and return their paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = run_dir / "report.md"
    markdown_path.write_text(render_markdown(context), encoding="utf-8")
    html_path = run_dir / "report.html"
    html_path.write_text(render_html(context, run_dir / "figures"), encoding="utf-8")
    return {"report_md": str(markdown_path), "report_html": str(html_path)}


def terminal_overview(context: ReportContext) -> str:
    """Return the compact terminal overview printed before the report path and chart menu."""
    meta = context.meta
    split = meta.get("split", {})
    lines = [
        f"run {context.run_id}  dataset={context.dataset}  data_version={context.data_version}",
        f"  sessions={meta.get('sessions')}  ticker-sessions={meta.get('n_units'):,}  "
        f"discovery={split.get('discovery_sessions')} ({split.get('discovery_end')})  "
        f"holdout={split.get('holdout_sessions')}",
        f"  tiers: A={meta.get('tiers', {}).get('counts', {}).get('A')} "
        f"B={meta.get('tiers', {}).get('counts', {}).get('B')} "
        f"C={meta.get('tiers', {}).get('counts', {}).get('C')}  |  "
        f"sentiment units={meta.get('units_with_sentiment'):,}  "
        f"excluded units={meta.get('units_excluded_from_tests'):,}",
        f"  lexicon={context.lexicon.get('lexicon_id', '?')} ({context.lexicon.get('source', '?')})  "
        f"llm={'on' if (context.llm or {}).get('enabled') else 'off'}  "
        f"results={len(context.results)}  findings={len(context.findings)}",
    ]
    for warning in context.warnings:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def relative(path: Path) -> str:
    """Return a path relative to the repository root when it is inside it (for tidy output)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
