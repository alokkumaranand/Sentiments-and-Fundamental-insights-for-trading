"""The coordinator: plain Python that sequences the stages and owns every rule.

Design decision embodied here: **the coordinator is not an LLM.** It enforces the tool budget,
validates agent output against schemas, decides when the holdout is checked, and writes the run
manifest. The two things an agent most wants to control — which period is read and whether a result
is "supported" — are the two things it never touches.

The run flow is the one in SPEC.md 7.1 and each stage is a separate function, so `playbook` mode
runs stages 1-5 with no endpoint at all and `full` mode adds the two agent stages on top.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents.explorer import ExplorerOutcome, run_explorer
from .agents.induction import InductionResult, default_lexicon, run_induction
from .agents.llm import LLMClient
from .agents.null_agent import null_agent_findings
from .agents.presenter import run_presenter
from .config import Settings
from .features.lexicon import resolve_lexicon
from .features.panel import Panel
from .findings.checker import check_draft
from .findings.schema import Evidence, Finding, FindingEffect, FindingsFile, FindingScope
from .findings.status_rules import assign_status, summarise_status_counts
from .ingest.profile import new_run_id
from .prep.align import build_canonical
from .prep.build import get_panel
from .tools import build_registry
from .tools.playbook import PlaybookOutcome, run_playbook
from .tools.registry import ToolError, ToolRegistry
from .tools.validation import TEST_TOOLS
from .version import CODE_VERSION

# Cap on coordinator-triggered holdout checks: each one re-runs a full tool on the holdout period,
# and the point is to test the tests the findings rest on, not to re-run everything.
MAX_HOLDOUT_CHECKS = 8


@dataclass
class RunState:
    """Everything one run produced. The report and the manifest are views of this."""

    run_id: str
    run_dir: Path
    settings: Settings
    dataset: str
    panel: Panel
    registry: ToolRegistry
    mode: str = "playbook"
    warnings: list[str] = field(default_factory=list)
    lexicon: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    induction: InductionResult | None = None
    playbook: PlaybookOutcome = field(default_factory=PlaybookOutcome)
    explorer: ExplorerOutcome | None = None
    findings: list[Finding] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    narrative: dict[str, Any] | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


def start_run(settings: Settings, dataset: str | None = None) -> tuple[str, Path]:
    """Create the run id and directory. Returns (run_id, run_dir)."""
    run_id = new_run_id(dataset or settings.dataset_name)
    run_dir = settings.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def stage_lexicon(
    settings: Settings,
    client: LLMClient,
    run_dir: Path,
    induce: bool,
    use_cache: bool,
    force_default_lexicon: bool,
    console: Any = None,
) -> tuple[Panel, dict[str, Any], InductionResult | None, list[str]]:
    """Resolve the vocabulary, optionally inducing it first, then build the panel with it.

    Order matters and is the spec's: a frozen induced lexicon if one is active, else induction when
    an endpoint is configured, else the shipped default. Induction never fails the run.
    """
    induction: InductionResult | None = None
    warnings: list[str] = []
    # Induction runs only when there is no usable frozen vocabulary (SPEC.md 7.1 step 4): re-inducing
    # on every run would make the run's numbers depend on a fresh model call that changes nothing,
    # and the frozen id is what a reader uses to attribute a metric change.
    _, provenance = resolve_lexicon(settings, force_default=force_default_lexicon)
    if induce and not force_default_lexicon and provenance.get("source") != "induced":
        canonical = build_canonical(settings)
        warnings = canonical.report["warnings"]
        induction = run_induction(
            settings, canonical, default_lexicon(settings), run_dir, client,
            enabled=True, warnings=warnings,
        )
        if induction.provenance.get("source") == "induced":
            if console is not None:
                console.print(
                    f"  induced lexicon {induction.provenance.get('lexicon_id')}: "
                    f"{len(induction.accepted)} term(s) accepted, {len(induction.rejected)} rejected"
                )
        elif console is not None:
            console.print(f"  induction kept the shipped lexicon: {induction.provenance.get('reason')}")
    elif console is not None and provenance.get("source") == "induced":
        console.print(f"  frozen lexicon {provenance.get('lexicon_id')} in use: induction skipped")

    result = get_panel(settings, use_cache=use_cache, force_default_lexicon=force_default_lexicon)
    if not warnings:
        warnings = result.warnings
    return result.panel, result.panel.meta.get("lexicon", {}), induction, warnings


def stage_playbook(registry: ToolRegistry) -> PlaybookOutcome:
    """Run the fixed baseline analyses. A failing call is recorded, never fatal."""
    return run_playbook(registry)


def stage_findings(
    settings: Settings,
    state: RunState,
    client: LLMClient,
    agents: bool,
) -> None:
    """Produce findings: explore (or template), check every claim, then assign status by rule."""
    registry = state.registry
    explorer_drafts_used = False
    if agents and client.available:
        outcome = run_explorer(
            client,
            registry,
            state.playbook.digest,
            max_tool_calls=int(settings.agent("explorer").get("max_tool_calls", 12)),
            retries_on_schema_error=int(settings.agent("explorer").get("retries_on_schema_error", 1)),
            playbook_failures=state.playbook.failures,
        )
        state.explorer = outcome
        state.rejected.extend(outcome.errors)
        drafts = outcome.drafts
        explorer_drafts_used = bool(drafts)
        if not drafts:
            # The explorer found nothing usable: the templated selection still gives the reader
            # cards, and the report says which path produced them.
            drafts = null_agent_findings(registry, _presenter_limit(settings))
            state.rejected.append(
                {"finding_id": "explorer", "verdict": "downgraded",
                 "reason": "explorer produced no usable draft; templated selection used"}
            )
    else:
        drafts = null_agent_findings(registry, _presenter_limit(settings))

    # Multiplicity: every test the playbook or the explorer ran in the discovery period is
    # adjusted together, which is the batch the screening decision actually covers.
    test_ids = [
        result.result_id for result in registry.store.all()
        if result.tool in TEST_TOOLS and result.period == "discovery" and result.stats.get("p_value") is not None
    ]
    qvalues: dict[str, float | None] = {}
    if test_ids:
        try:
            adjusted = registry.call("adjust_pvalues", {"result_ids": test_ids}, period="discovery")
            qvalues = {
                row["result_id"]: row.get("q_value") for row in (adjusted.table or [])
            }
            state.validation["adjust"] = {
                "result_id": adjusted.result_id, "n_tests": adjusted.stats.get("n_tests"),
                "n_significant": adjusted.stats.get("n_significant"), "alpha": adjusted.stats.get("alpha"),
            }
        except ToolError as error:
            state.rejected.append({"finding_id": "adjust_pvalues", "verdict": "rejected", "reason": str(error)})

    # The coordinator, never the agent, asks about the holdout.
    holdout: dict[str, dict[str, Any]] = {}
    for result_id in _holdout_targets(registry, drafts, test_ids):
        try:
            check = registry.call("holdout_check", {"result_id": result_id}, period="discovery")
            holdout[result_id] = check.stats
        except ToolError as error:
            holdout[result_id] = {"error": str(error)}
    state.validation["holdout"] = holdout

    alpha = float(settings.experiments.get("multiple_testing", {}).get("alpha", 0.05))
    own_text_cfg = (settings.experiments.get("sentiment", {}) or {}).get("own_text_only", {}) or {}
    known_numbers = _all_numbers(registry)

    for index, draft in enumerate(drafts, start=1):
        verdict, cited = check_draft(draft, registry.store, known_numbers)
        if verdict.verdict == "rejected":
            state.rejected.append(
                {"finding_id": f"F-{index:03d}", "verdict": "rejected", "reason": verdict.reason,
                 "action": verdict.action, "claim_text": draft.claim_text[:240]}
            )
            continue
        primary = cited[0]
        tier = _tier_of(draft, state)
        own_text_only = bool(own_text_cfg.get(f"tier_{(tier or 'c').lower()}", True))
        effect, q_value, same_sign = _effect_of(primary, qvalues, holdout)
        tested = primary.tool in TEST_TOOLS
        status, limitations = assign_status(
            has_test=tested,
            q_value=q_value,
            ci_95=effect.ci_95,
            same_sign_on_holdout=same_sign,
            tier=tier if _single_ticker(draft) else None,
            own_text_only=own_text_only,
            alpha=alpha,
        )
        if verdict.status_cap == "observed" and status != "observed":
            status = "observed"
            limitations = list(limitations) + list(verdict.limitations)
        finding = Finding(
            finding_id=f"F-{index:03d}",
            title=draft.title,
            scope=_scope_of(draft, state),
            claim_text=draft.claim_text,
            evidence=[
                Evidence(result_id=item.result_id, role=item.role) for item in draft.evidence
            ] + ([Evidence(result_id=holdout[primary.result_id]["holdout_result_id"], role="holdout")]
                 if holdout.get(primary.result_id, {}).get("holdout_result_id") else []),
            effect=effect,
            status=status,  # type: ignore[arg-type]
            limitations=list(draft.limitations) + list(limitations),
            alternative_explanations=list(draft.alternative_explanations),
            origin="explorer_agent" if explorer_drafts_used else "null_agent",
            run_id=state.run_id,
            lexicon_id=str(state.panel.meta.get("lexicon", {}).get("lexicon_id", "")),
            verification={"verdict": verdict.verdict, "reason": verdict.reason, "action": verdict.action},
        )
        state.findings.append(finding)


def stage_narrative(settings: Settings, state: RunState, client: LLMClient, agents: bool) -> None:
    """Write the summary and per-finding prose, then verify every number in it again."""
    findings = [finding.model_dump() for finding in state.findings]
    if not agents or not client.available:
        from .agents.null_agent import template_narrative

        state.narrative = template_narrative(findings, llm_enabled=False)
        return
    stats_by_result = {
        result.result_id: {"tool": result.tool, "params": result.params, "n_units": result.n_units,
                           "stats": result.stats}
        for finding in state.findings
        for result in [state.registry.store.get(item.result_id) for item in finding.evidence]
        if result is not None
    }
    narrative, rejections = run_presenter(client, findings, stats_by_result)
    state.narrative = narrative
    state.rejected.extend(rejections)


def write_artifacts(state: RunState) -> dict[str, str]:
    """Write results, findings, report and manifest. Returns the artifact paths."""
    from .report.build_report import ReportContext, write_reports

    run_dir = state.run_dir
    results_path = run_dir / "results" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps([result.model_dump() for result in state.registry.store.all()], indent=2, default=str),
        encoding="utf-8",
    )

    findings_file = FindingsFile(
        run_id=state.run_id,
        data_version=state.panel.data_version,
        lexicon_id=str(state.panel.meta.get("lexicon", {}).get("lexicon_id", "")),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        findings=state.findings,
        rejected=state.rejected,
    )
    findings_path = run_dir / "findings.json"
    findings_path.write_text(json.dumps(findings_file.model_dump(), indent=2, default=str), encoding="utf-8")

    context = ReportContext(
        run_id=state.run_id,
        dataset=state.dataset,
        data_version=state.panel.data_version,
        settings=state.settings,
        meta=state.panel.meta,
        results=state.registry.store.all(),
        findings=[finding.model_dump() for finding in state.findings],
        narrative=state.narrative,
        warnings=state.warnings,
        llm=state.llm,
        playbook_failures=state.playbook.failures,
        rejected_claims=state.rejected,
        lexicon=state.lexicon,
    )
    paths = write_reports(context, run_dir)
    paths.update({"results": str(results_path), "findings": str(findings_path)})

    manifest = _manifest(state, context)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    state.manifest = manifest
    state.paths = paths
    return paths


def _manifest(state: RunState, context: Any) -> dict[str, Any]:
    """Assemble the run manifest (SPEC.md 7.2). Never contains a secret: no key, ever."""
    meta = state.panel.meta
    return {
        "run_id": state.run_id,
        "dataset": state.dataset,
        "mode": state.mode,
        "code_version": CODE_VERSION,
        "data_version": state.panel.data_version,
        "started_at": state.manifest.get("started_at"),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_hashes": meta.get("canonical_report", {}).get("hashes", {}),
        "settings": state.settings.raw,
        "experiments": state.settings.experiments,
        "lexicon": state.lexicon,
        "induction": (state.induction.report if state.induction else None),
        "llm": state.llm,
        "llm_calls": [],
        "tool_calls": [
            {"result_id": result.result_id, "tool": result.tool, "params": result.params,
             "period": result.period, "n_units": result.n_units, "n_dropped": result.n_dropped,
             "chart": bool(result.chart)}
            for result in state.registry.store.all()
        ],
        "tool_errors": state.registry.errors,
        "playbook_failures": state.playbook.failures,
        "explorer": (state.explorer.report if state.explorer else {"outcome": "not run"}),
        "validation": state.validation,
        "findings_summary": summarise_status_counts(state.findings),
        "rejected": state.rejected,
        "narrative_source": (state.narrative or {}).get("source", "none"),
        "quality_warnings": state.warnings,
        "artifacts": state.paths,
    }


def run_pipeline(
    settings: Settings,
    dataset: str | None = None,
    mode: str = "playbook",
    use_cache: bool = True,
    force_default_lexicon: bool = False,
    induce: bool = True,
    console: Any = None,
) -> RunState:
    """Run the pipeline end to end and return the finished `RunState`.

    `mode="playbook"` is the no-LLM path: it must produce a complete report on its own.
    `mode="full"` adds lexicon induction, the explorer and the presenter — each of which degrades
    to its templated fallback rather than failing the run.
    """
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    began = time.time()
    run_id, run_dir = start_run(settings, dataset)
    client = LLMClient(settings)
    agents = mode == "full"
    if console is not None:
        console.print(f"[bold]{mode}[/bold] run={run_id} dataset={settings.dataset_name} llm={'on' if client.available else 'off'}")

    panel, lexicon_meta, induction, warnings = stage_lexicon(
        settings, client, run_dir,
        induce=agents and induce and client.available,
        use_cache=use_cache,
        force_default_lexicon=force_default_lexicon,
        console=console,
    )
    registry = build_registry(
        settings, panel.panel, panel.tweets, panel.data_version, panel.tiers, panel.calendar
    )
    state = RunState(
        run_id=run_id,
        run_dir=run_dir,
        settings=settings,
        dataset=settings.dataset_name,
        panel=panel,
        registry=registry,
        mode=mode,
        warnings=warnings,
        lexicon=lexicon_meta,
        llm=client.describe(),
    )
    state.manifest["started_at"] = started

    state.playbook = stage_playbook(registry)
    if console is not None:
        console.print(
            f"  baseline: {len(state.playbook.result_ids)} tool calls, "
            f"{len(state.playbook.failures)} skipped"
        )

    stage_findings(settings, state, client, agents=agents)
    stage_narrative(settings, state, client, agents=agents)
    write_artifacts(state)
    state.manifest["seconds"] = round(time.time() - began, 2)
    # The manifest gains the duration and the LLM call log, so it is rewritten once at the end:
    # a manifest that disagrees with itself is worse than one written twice.
    state.manifest["llm_calls"] = client.calls
    (run_dir / "manifest.json").write_text(json.dumps(state.manifest, indent=2, default=str), encoding="utf-8")
    if console is not None:
        console.print(f"  {mode} finished in {state.manifest['seconds']}s")
    return state


# --- helpers -----------------------------------------------------------------


def _presenter_limit(settings: Settings) -> int:
    """Return the maximum number of findings to carry into the presentation."""
    return int(settings.agent("presenter").get("max_findings", 8))


def _single_ticker(draft: Any) -> bool:
    """True when the draft is about exactly one ticker (so the tier cap applies)."""
    tickers = draft.scope.tickers if draft.scope else []
    return len(tickers) == 1 and tickers[0] not in ("pooled", "pooled_ex_TSLA")


def _tier_of(draft: Any, state: RunState) -> str | None:
    """Return the coverage tier of the draft's single ticker, if it is about one ticker."""
    if not _single_ticker(draft):
        return None
    return state.panel.tiers.get(draft.scope.tickers[0])


def _scope_of(draft: Any, state: RunState) -> FindingScope:
    """Build the presented scope, filling in the tier when the finding is about one ticker."""
    scope = draft.scope or FindingScope()
    if _single_ticker(draft):
        scope = scope.model_copy(update={"tier": state.panel.tiers.get(scope.tickers[0])})
    return scope


def _effect_of(
    primary: Any, qvalues: dict[str, float | None], holdout: dict[str, dict[str, Any]]
) -> tuple[FindingEffect, float | None, bool | None]:
    """Extract the effect a finding cites from its primary result, plus q and holdout sign."""
    stats = primary.stats
    q_value = qvalues.get(primary.result_id)
    same_sign = holdout.get(primary.result_id, {}).get("same_sign")
    if primary.tool == "bucket_forward_returns":
        return (
            FindingEffect(
                metric="spread_top_minus_bottom",
                value=stats.get("spread_top_minus_bottom"),
                ci_95=stats.get("spread_ci_95"),
                n_units=stats.get("n_units"),
                q_value=q_value,
            ),
            q_value,
            same_sign,
        )
    if primary.tool == "correlate":
        return (
            FindingEffect(
                metric="spearman_rho",
                value=stats.get("spearman_rho"),
                ci_95=stats.get("ci_95"),
                n_units=stats.get("n_units"),
                q_value=q_value,
            ),
            q_value,
            same_sign,
        )
    return (
        FindingEffect(metric=stats.get("metric") or stats.get("column"), n_units=primary.n_units),
        None,
        None,
    )


def _holdout_targets(registry: ToolRegistry, drafts: list[Any], test_ids: list[str]) -> list[str]:
    """Choose which test results get a holdout re-run: cited ones first, then the rest, capped.

    Cited-first matters: the holdout exists to test the claims in the report, not every number the
    playbook happened to compute.
    """
    cited = {
        item.result_id
        for draft in drafts
        for item in draft.evidence
        if registry.store.get(item.result_id) is not None
        and registry.store.get(item.result_id).tool in TEST_TOOLS  # type: ignore[union-attr]
    }
    ordered = [result_id for result_id in test_ids if result_id in cited]
    ordered += [result_id for result_id in test_ids if result_id not in cited]
    return ordered[:MAX_HOLDOUT_CHECKS]


def _all_numbers(registry: ToolRegistry) -> set[float]:
    """Collect every number in the store, so a claim may cite any result it is allowed to see.

    The claim checker still requires the cited ids to exist; this only widens the set of numbers a
    claim may use, because a presenter may legitimately repeat a figure from another result it was
    given (for example an `n_units` mentioned in a caption).
    """
    numbers: set[float] = set()
    for result in registry.store.all():
        numbers.update(result.cited_numbers())
        numbers.add(float(result.n_units))
        numbers.add(float(result.n_obs))
        numbers.add(float(result.n_dropped))
    return numbers
