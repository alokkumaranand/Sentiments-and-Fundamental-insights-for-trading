"""Command line entry point.

Commands:
    doctor    check the environment, the input files and (optionally) the LLM endpoint; write nothing
    profile   read-only profile of the active dataset -> profile.json + quality_report.md
    features  canonical tables -> ticker-session panel -> docs/feature_book.md (cached)
    induce    Feature Induction Agent + deterministic validation gate -> frozen lexicon
    playbook  the fixed no-LLM analyses -> ToolResults, report.md/html, numbered chart menu
    run       the full pipeline: playbook + agents + holdout check + findings + report + manifest
    charts    re-open the chart menu for a finished run

Run it as:   .venv\\Scripts\\python.exe run.py <command>        (from the repository root)

Why there is no console script: the project is used straight from `src/` without an install
step, and run.py is the one line that puts `src/` on the path. That keeps setup to
"create venv, install requirements, run run.py" with no editable install and no build.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from .config import REPO_ROOT, Settings, load_settings
from .ingest.profile import build_profile, new_run_id, write_profile, write_quality_report
from .ingest.validate import BlockingError

# Imported lazily inside the commands that need them, so `doctor` can report a missing
# dependency instead of dying on the import that would have detected it.
REQUIRED_MODULES = ("pandas", "numpy", "yaml", "dotenv", "rich")
OPTIONAL_MODULES = ("scipy", "pydantic", "matplotlib", "vaderSentiment")


def _console() -> object:
    """Return a rich Console, or a tiny fallback if rich is not installed yet."""
    try:
        from rich.console import Console

        return Console()
    except Exception:  # pragma: no cover - only when rich is missing

        class _Plain:
            def print(self, *args: object, **_: object) -> None:
                print(*(str(a) for a in args))

        return _Plain()


def cmd_doctor(settings: Settings, ping: bool = True) -> int:
    """Report environment, dependency, input and endpoint state. Never modifies anything.

    Returns 0 when everything required is present, 1 otherwise. Optional modules are listed but
    do not fail the check, because the pipeline is designed to run without them. The LLM ping is
    a one-token request whose only purpose is to prove the endpoint is up and the model name is
    right; it is a warning, never a failure, because a no-key run is a supported mode.
    """
    console = _console()
    problems: list[str] = []

    console.print("[bold]eda doctor[/bold]")
    console.print(f"  python        : {platform.python_version()} ({sys.executable})")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    console.print(f"  venv active   : {'yes' if in_venv else 'NO (using the system interpreter)'}")
    if not in_venv:
        problems.append("not running inside .venv")

    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        console.print(f"  required      : {module:<14} {'ok' if found else 'MISSING'}")
        if not found:
            problems.append(f"missing required module '{module}'")

    for module in OPTIONAL_MODULES:
        found = importlib.util.find_spec(module) is not None
        console.print(f"  optional      : {module:<14} {'ok' if found else 'not installed'}")

    console.print(f"  dataset       : {settings.dataset_name}")
    for label, path in (("tweets", settings.tweets_path), ("prices", settings.prices_path)):
        # Existence and size only: doctor must not hash 6 MB on every invocation, and the
        # authoritative hashes are recorded by `profile` in the run manifest.
        state = f"{path.stat().st_size:,} bytes" if path.exists() else "MISSING"
        console.print(f"  {label:<13} : {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}  ({state})")
        if not path.exists():
            problems.append(f"missing {label} file: {path}")

    env_path = REPO_ROOT / ".env"
    console.print(f"  .env          : {'present' if env_path.exists() else 'absent (a no-key run is supported)'}")

    ledger = REPO_ROOT / "artifacts" / "lexicons" / "active.json"
    console.print(f"  lexicon       : {'induced -> ' + str(ledger) if ledger.exists() else 'shipped config/lexicons/default_v1.json'}")

    from .agents.llm import LLMClient

    client = LLMClient(settings)
    wiring = client.describe()
    console.print(
        f"  llm enabled   : {wiring['enabled']}  model={wiring['model'] or '(unset)'}  "
        f"host={wiring['host'] or '(unset)'}  tool_calling={wiring['tool_calling']}"
    )
    if ping:
        result = client.ping()
        if result.get("ok"):
            console.print(f"  llm ping      : [green]ok[/green] ({result['seconds']}s, {result['model']})")
        elif result.get("attempted"):
            console.print(f"  llm ping      : [yellow]failed[/yellow] {result.get('reason')}")
        else:
            console.print(f"  llm ping      : skipped ({result.get('reason')})")

    if problems:
        console.print("[red]problems:[/red] " + "; ".join(problems))
        return 1
    console.print("[green]all required checks passed[/green]")
    return 0


def cmd_profile(settings: Settings, check_splits: bool) -> int:
    """Build and persist the dataset profile, then print a terminal summary.

    Returns 0 on success and 2 on a blocking data problem, so a caller (or a later phase's
    coordinator) can tell "bad data" apart from "crashed".
    """
    console = _console()
    run_id = new_run_id(settings.dataset_name)
    run_dir = settings.artifacts_dir / run_id

    console.print(f"[bold]profiling[/bold] dataset={settings.dataset_name} run={run_id}")
    try:
        profile = build_profile(settings, enforce_split_check=check_splits)
    except BlockingError as error:
        console.print(f"[red]BLOCKING:[/red] {error}")
        return 2

    profile["run_id"] = run_id
    write_profile(profile, run_dir / "profile.json")
    # Both copies: the one beside profile.json is the durable per-run record, and the
    # configured path is a convenience "latest" pointer for someone browsing docs/.
    write_quality_report(profile, run_dir / "quality_report.md")
    write_quality_report(profile, settings.quality_report_path)
    _print_summary(console, profile, run_dir)
    return 0


def _print_summary(console: object, profile: dict, run_dir: Path) -> None:
    """Print the short terminal overview: the numbers a reader checks first."""
    from rich.table import Table

    t, p, tiers = profile["tweets"], profile["prices"], profile["tiers"]

    table = Table(title=f"dataset '{profile['dataset']}'  ({profile['data_version']})", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("tweets", f"{t['rows']:,} rows, {t['distinct_tickers']} tickers, {t['distinct_texts']:,} distinct texts")
    table.add_row("tweet window", f"{t['date_min'][:10]} -> {t['date_max'][:10]}")
    table.add_row("prices", f"{p['rows']:,} rows, {p['sessions']} sessions, {p['distinct_tickers']} tickers")
    table.add_row("cross-posted text", f"{t['rows_in_shared_text']:,} rows ({t['rows_in_shared_text_share']:.1%}) appear under >1 ticker")
    table.add_row("promo-shaped", f"{t['promo_rows']:,} rows ({t['promo_share']:.1%})")
    # These two buckets OVERLAP -- a weekend tweet posted after 16:00 ET is in both -- so they
    # are shown side by side and never added together.
    table.add_row(
        "rolls forward",
        f"{t['non_trading_day_rows']:,} on non-trading dates | "
        f"{t['at_or_after_close_rows']:,} at/after close (overlapping buckets)",
    )
    table.add_row("adj == close", f"{p['adj_equals_close_rows']:,}/{p['rows']:,} rows")
    table.add_row("split check", f"{'passed' if p['split_check']['passed'] else 'FAILED'} (max |1d ret| {p['split_check']['max_abs_1d_return']:.3f}, {p['split_check']['worst_ticker']})")
    table.add_row("ragged sessions", str(len(p["ragged_sessions"])))
    table.add_row("tiers", f"A={tiers['counts']['A']} B={tiers['counts']['B']} C={tiers['counts']['C']}")
    console.print(table)

    for warning in profile["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    console.print(f"wrote {run_dir / 'profile.json'}")
    console.print(f"wrote {REPO_ROOT / profile.get('quality_report_path', 'docs/quality_report.md')}")


def cmd_features(settings: Settings, use_cache: bool, force_default_lexicon: bool) -> int:
    """Build (or load) the ticker-session panel and write the feature book.

    Returns 2 on a blocking data problem, like `profile`, because this stage can also discover
    one (the split-continuity check runs here too).
    """
    console = _console()
    from .features.feature_book import feature_overview, write_feature_book
    from .prep.build import get_panel

    console.print(f"[bold]features[/bold] dataset={settings.dataset_name}")
    try:
        result = get_panel(settings, use_cache=use_cache, force_default_lexicon=force_default_lexicon)
    except BlockingError as error:
        console.print(f"[red]BLOCKING:[/red] {error}")
        return 2

    panel = result.panel
    overview = feature_overview(panel)
    meta = panel.meta

    from rich.table import Table

    table = Table(title=f"panel '{meta.get('dataset', settings.dataset_name)}'  ({panel.data_version})", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("source", result.source + (" (artifacts/cache)" if result.source == "cache" else " (rebuilt and cached)"))
    table.add_row("lexicon", f"{meta.get('lexicon', {}).get('lexicon_id', '?')} ({meta.get('lexicon', {}).get('source', '?')})")
    table.add_row("units", f"{overview['units']:,} ticker-sessions, {overview['tweets_in_panel']:,} tweets in panel")
    table.add_row("attention", f"A3 own texts {overview['own_texts']:,} | sustained spikes {overview['sustained_spikes']}")
    table.add_row("sentiment", f"{overview['sentiment_units']:,} units scored | {overview['low_coverage_units']:,} below the tweet minimum")
    table.add_row("tiers", f"A={meta.get('tiers', {}).get('counts', {}).get('A')} B={meta.get('tiers', {}).get('counts', {}).get('B')} C={meta.get('tiers', {}).get('counts', {}).get('C')}")
    split = meta.get("split", {})
    table.add_row("split", f"discovery {split.get('discovery_sessions')} sessions -> {split.get('discovery_end')} | holdout {split.get('holdout_sessions')}")
    console.print(table)

    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    book = write_feature_book(panel, settings, REPO_ROOT / "docs" / "feature_book.md")
    console.print(f"wrote {book}")
    if result.paths:
        for label, path in result.paths.items():
            console.print(f"wrote {path}")
    return 0


def cmd_induce(settings: Settings, live: bool) -> int:
    """Run the Feature Induction Agent and the deterministic validation gate.

    `--no-live` skips the model call and shows the fallback path, which is how the no-key
    behaviour is demonstrated without editing `.env`.
    """
    console = _console()
    from .agents.induction import default_lexicon, run_induction
    from .agents.llm import LLMClient
    from .prep.align import build_canonical

    console.print(f"[bold]induce[/bold] dataset={settings.dataset_name}")
    canonical = build_canonical(settings)
    seed = default_lexicon(settings)
    run_id = new_run_id(settings.dataset_name)
    run_dir = settings.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    client = LLMClient(settings)
    console.print(f"  llm: {client.describe()['host'] or '(unset)'} model={client.config.model or '(unset)'}")
    if live and client.available:
        console.print("  calling the endpoint once (capped, stratified, temperature 0)…")
    result = run_induction(
        settings, canonical, seed, run_dir, client,
        enabled=live, warnings=canonical.report["warnings"],
    )

    from rich.table import Table

    sample = result.report.get("sample", {})
    table = Table(title="induction", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("sample", f"{sample.get('sample_rows')} tweets, {sample.get('tickers_covered')} tickers, {sample.get('sample_strata')}")
    table.add_row("validation", f"{sample.get('validation_rows')} tweets, {sample.get('validation_sessions')} sessions (session-disjoint)")
    table.add_row("proposed/accepted/rejected", f"{(result.report.get('proposal') or {}).get('terms_proposed', 0)} / {len(result.accepted)} / {len(result.rejected)}")
    table.add_row("outcome", str(result.report.get("outcome", "?")))
    table.add_row("lexicon in use", f"{result.lexicon.lexicon_id} ({result.provenance.get('source')}), {len(result.lexicon.terms)} terms")
    console.print(table)

    if result.accepted:
        console.print("[green]accepted terms[/green]: " + ", ".join(f"{t.term}({t.polarity:+.2f})" for t in result.accepted[:25]))
    for entry in result.rejected[:12]:
        console.print(f"[yellow]rejected[/yellow]: {entry['term']} — {entry.get('rule')}")
    if len(result.rejected) > 12:
        console.print(f"  … {len(result.rejected) - 12} more in {result.report.get('rejected_proposals_path')}")
    if result.lexicon_path:
        console.print(f"wrote {result.lexicon_path}")
        console.print("the active-lexicon pointer now selects it; `--lexicon default` reverts")
    return 0


def cmd_run(
    settings: Settings,
    mode: str,
    use_cache: bool,
    force_default_lexicon: bool,
    induce: bool,
    menu: bool,
) -> int:
    """Run the baseline (`playbook`) or the full pipeline (`run`) and present the result.

    Returns 0 on success and 2 on a blocking data problem. Both modes print the overview, the
    report path and the numbered chart menu; `playbook` makes no model call at all.
    """
    console = _console()
    from .ingest.validate import BlockingError as _Blocking
    from .pipeline import run_pipeline
    from .report.build_report import ReportContext, terminal_overview
    from .report.chart_menu import ChartMenu

    try:
        state = run_pipeline(
            settings,
            mode=mode,
            use_cache=use_cache,
            force_default_lexicon=force_default_lexicon,
            induce=induce,
            console=console,
        )
    except _Blocking as error:
        console.print(f"[red]BLOCKING:[/red] {error}")
        return 2

    context = ReportContext(
        run_id=state.run_id,
        dataset=state.dataset,
        data_version=state.panel.data_version,
        settings=settings,
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
    console.print(terminal_overview(context))
    _print_findings(console, state.findings)
    for label, path in state.paths.items():
        console.print(f"wrote {path}")

    if menu:
        ChartMenu(state.registry.store.all(), state.run_dir, settings, console).run()
    return 0


def _print_findings(console: object, findings: list) -> None:
    """Print each finding as a short block with its badge, sample size, effect and evidence ids."""
    from .findings.status_rules import status_badge

    if not findings:
        console.print("[yellow]no findings[/yellow]: the run produced results but no finding card")
        return
    for finding in findings:
        effect = finding.effect
        console.print(
            f"\n[bold]{finding.finding_id} {finding.title}[/bold]  "
            f"[{status_badge(finding.status)}] status={finding.status}  origin={finding.origin}"
        )
        console.print(f"  {finding.claim_text}")
        console.print(
            f"  effect: {effect.metric}={effect.value} (95% CI {effect.ci_95})  "
            f"n_units={effect.n_units}  q={effect.q_value}"
        )
        console.print("  evidence: " + ", ".join(f"{item.result_id}({item.role})" for item in finding.evidence))
        for limitation in finding.limitations[:2]:
            console.print(f"  [dim]limitation:[/dim] {limitation}")


def cmd_charts(settings: Settings, run_id: str | None) -> int:
    """Re-open the chart menu for a finished run, rebuilt from its persisted results."""
    console = _console()
    from .report.chart_menu import ChartMenu
    from .tools.schema import ToolResult

    runs_dir = settings.artifacts_dir
    if run_id:
        run_dir = runs_dir / run_id
    else:
        candidates = sorted(
            (path for path in runs_dir.glob("*") if (path / "results" / "results.json").exists()),
            key=lambda path: path.name,
        )
        if not candidates:
            console.print(f"no run with persisted results under {runs_dir}")
            return 1
        run_dir = candidates[-1]

    results_path = run_dir / "results" / "results.json"
    if not results_path.exists():
        console.print(f"{results_path} not found: run `playbook` or `run` first")
        return 1
    results = [ToolResult.model_validate(item) for item in json.loads(results_path.read_text(encoding="utf-8"))]
    console.print(f"charts for run {run_dir.name} ({len(results)} results)")
    ChartMenu(results, run_dir, settings, console).run()
    return 0


def cmd_append(settings: Settings, file: str) -> int:
    """Ingest a new tweets file append-only, re-score it and check lexicon drift.

    Returns 2 on blocking schema drift, because guessing a column mapping would corrupt every
    number rather than merely make it uncertain.
    """
    console = _console()
    from .append import append_tweets
    from .ingest.validate import BlockingError as _Blocking

    try:
        result = append_tweets(settings, Path(file))
    except _Blocking as error:
        console.print(f"[red]BLOCKING:[/red] {error}")
        return 2
    report = result.report
    console.print(
        f"[bold]append[/bold] {report['appended_rows']:,} new row(s), "
        f"{report['in_window_rows']:,} placed on a session, {report['past_last_session_rows']:,} past the last session"
    )
    console.print(
        f"  lexicon {report['lexicon']['lexicon_id']} ({report['lexicon']['source']}), "
        f"{report['llm_calls']} LLM call(s)"
    )
    console.print(
        f"  coverage drift {report['drift']['uncovered_token_share']:.1%} "
        f"(threshold {report['drift_threshold']:.0%}) -> {report['recommendation']}"
    )
    for label, path in result.paths.items():
        console.print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser. Every command takes --dataset so a one-off switch needs no edit."""
    parser = argparse.ArgumentParser(
        prog="eda_demo",
        description="Agentic EDA demo — LLM-induced features, deterministic EDA, terminal findings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="override datasets.active for this run (e.g. dev or full); changes nothing on disk",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check environment, dependencies, inputs and the endpoint")
    doctor.add_argument("--ping", action=argparse.BooleanOptionalAction, default=True,
                        help="send a 1-token request to the LLM endpoint when it is enabled")

    profile_parser = sub.add_parser("profile", help="profile the active dataset read-only")
    # BooleanOptionalAction yields both --check-splits and --no-check-splits; the check is ON
    # by default because an unadjusted split silently corrupts every return.
    profile_parser.add_argument(
        "--check-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail if a 1-day price move suggests an unadjusted split",
    )

    features = sub.add_parser("features", help="build the ticker-session panel and the feature book")
    features.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                          help="reuse the cached panel when the data_version matches")
    features.add_argument("--lexicon", choices=["default", "active"], default="active",
                          help="score with the shipped lexicon instead of an induced one")

    induce = sub.add_parser("induce", help="run the Feature Induction Agent and its validation gate")
    induce.add_argument("--live", action=argparse.BooleanOptionalAction, default=True,
                        help="actually call the endpoint; --no-live demonstrates the fallback path")

    playbook = sub.add_parser("playbook", help="run the fixed no-LLM analyses and present the report")
    _add_run_flags(playbook)

    run = sub.add_parser("run", help="run the full pipeline: playbook + agents + holdout + findings")
    _add_run_flags(run)
    run.add_argument("--agents", action=argparse.BooleanOptionalAction, default=None,
                     help="force the agent layer on or off; default is on when a key is configured")
    run.add_argument("--induce", action=argparse.BooleanOptionalAction, default=True,
                     help="induce a lexicon first when no induced one is active")

    charts = sub.add_parser("charts", help="re-open the numbered chart menu for a finished run")
    charts.add_argument("--run-id", default=None, help="run directory name; default is the most recent")

    append = sub.add_parser("append", help="ingest new tweets append-only and check lexicon drift")
    append.add_argument("--file", required=True, help="CSV of arriving tweets with the same columns")

    return parser


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by `playbook` and `run`."""
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse the cached panel when the data_version matches")
    parser.add_argument("--lexicon", choices=["default", "active"], default="active",
                        help="score with the shipped lexicon instead of an induced one")
    parser.add_argument("--menu", action=argparse.BooleanOptionalAction, default=True,
                        help="show the numbered chart menu after the report")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load config, and dispatch. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.dataset)
    except SystemExit as error:
        print(error, file=sys.stderr)
        return 2
    if args.command == "doctor":
        return cmd_doctor(settings, ping=args.ping)
    if args.command == "profile":
        return cmd_profile(settings, check_splits=args.check_splits)
    if args.command == "features":
        return cmd_features(settings, use_cache=args.cache, force_default_lexicon=args.lexicon == "default")
    if args.command == "induce":
        return cmd_induce(settings, live=args.live)
    if args.command == "playbook":
        return cmd_run(
            settings, mode="playbook", use_cache=args.cache,
            force_default_lexicon=args.lexicon == "default", induce=False, menu=args.menu,
        )
    if args.command == "run":
        agents = args.agents
        if agents is None:
            from .agents.llm import LLMClient

            agents = LLMClient(settings).available
        return cmd_run(
            settings, mode="full" if agents else "playbook", use_cache=args.cache,
            force_default_lexicon=args.lexicon == "default", induce=args.induce, menu=args.menu,
        )
    if args.command == "charts":
        return cmd_charts(settings, args.run_id)
    if args.command == "append":
        return cmd_append(settings, args.file)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
