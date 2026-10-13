"""Feature Induction Agent — the LLM writes vocabulary, and a deterministic gate decides.

Design decision embodied here: **the model is a hypothesis generator, never an authority.** It
reads a capped sample and returns candidate terms with a claimed polarity. Every candidate is then
tested on a session-disjoint slice by plain code:

1. it must occur at least `min_support` times;
2. its claimed sign must agree with the polarity of the text *around* it — the surrounding words
   scored with the shipped lexicon, with the candidate's own contribution removed — at least
   `sign_agreement` of the time;
3. a block-bootstrap CI for that surrounding polarity must exclude zero.

Only then does it enter the frozen lexicon, and the whole frozen file is hashed into the id that
every downstream result carries. The gate's "truth" is the shipped seed lexicon, not a human
label set; that is a real limitation and it is stated in the Phase 1 review rather than hidden.

Failure is always survivable: no endpoint, invalid JSON twice, or an empty accepted set all end in
the shipped `default_v1.json`, and the reason is recorded (SPEC.md 4.3 "can never fail a run").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from ..agents.llm import LLMClient, LLMError
from ..agents.sampler import format_sample, sample_for_induction
from ..config import REPO_ROOT, Settings
from ..features.lexicon import (
    FrozenLexicon,
    LexiconProposal,
    Term,
    default_lexicon_path,
    lexicon_id_for,
    set_active_lexicon,
    write_lexicon,
)
# The matcher lives with the scorer it serves, so the gate and the feature layer can never
# disagree about how a term is matched against text.
from ..features.sentiment import LexiconMatcher

# Families whose "polarity" claim is not meaningful: they are scored as shares, so the gate can
# only test that the term is actually present in the corpus, and it says so in the record.
_MARKER_FAMILIES = {"uncertainty", "attribution", "sarcasm"}

# Cap on terms requested in one call: a long list is mostly noise, and every extra term costs a
# gate test and a line in the review.
_MAX_TERMS_REQUESTED = 60


@dataclass
class InductionResult:
    """What `run_induction` produced: the lexicon to use, the gate record, and how it went."""

    lexicon: FrozenLexicon
    provenance: dict[str, Any]
    accepted: list[Term]
    rejected: list[dict[str, Any]]
    report: dict[str, Any]
    lexicon_path: Path | None = None

    @property
    def fallback(self) -> bool:
        """True when the shipped default was kept rather than an induced vocabulary."""
        return self.provenance.get("source") != "induced"


def _term_mask(texts: pd.Series, term: str) -> pd.Series:
    """Return the rows whose text contains `term`: token-exact for one word, substring for a phrase."""
    if " " in term:
        return texts.str.contains(term, regex=False)
    tokens = texts.str.split()
    return tokens.map(lambda words: term in words)


def _bootstrap_mean_ci(
    values: np.ndarray,
    session_ids: np.ndarray,
    block_sessions: int,
    resamples: int,
    seed: int,
) -> list[float]:
    """Block-bootstrap 95% CI for the mean of `values`, or [nan, nan] when it cannot be estimated.

    Blocks of consecutive sessions, not single observations: tweets within a session are the same
    news event seen many times, so an observation-level bootstrap would report a CI that is far
    too narrow. Two blocks are the minimum for the resample to mean anything.
    """
    if len(values) == 0:
        return [0.0, 0.0]
    order = np.argsort(session_ids, kind="stable")
    values = values[order]
    sessions = session_ids[order]
    boundaries = np.arange(0, len(values), max(1, block_sessions))
    blocks = [values[start:start + max(1, block_sessions)] for start in boundaries]
    if len(blocks) < 2:
        # One block is not a distribution: resampling it would return the point estimate as its own
        # confidence interval, which reads as certainty. The gate rejects on a non-finite CI.
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for i in range(resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        means[i] = np.concatenate([blocks[j] for j in chosen]).mean()
    return [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]


def gate_terms(
    proposed: list[Term],
    validation: pd.DataFrame,
    seed_lexicon: FrozenLexicon,
    min_support: int,
    sign_agreement: float,
    resamples: int,
    block_sessions: int,
    seed: int,
    label_band: float = 0.05,
) -> tuple[list[Term], list[dict[str, Any]]]:
    """Test every proposed term on the validation slice; return (accepted, rejected).

    Rejected entries name the failing rule, so `rejected_proposals.json` explains what happened
    without anyone having to re-run the gate.
    """
    texts = validation["text_clean"].fillna("").astype(str)
    session_ids = validation["session_date"].astype(str).to_numpy()
    accepted: list[Term] = []
    rejected: list[dict[str, Any]] = []

    for term in proposed:
        record = {"term": term.term, "family": term.family, "polarity": term.polarity}
        mask = _term_mask(texts, term.term.lower().strip())
        support = int(mask.sum())
        record["support"] = support
        if support < min_support:
            rejected.append({**record, "rule": f"support < min_support ({min_support})"})
            continue

        if term.family in _MARKER_FAMILIES:
            # A marker's sign claim is meaningless; presence above min_support is the whole test.
            accepted.append(term.model_copy(update={"support": support, "sign_agreement": None, "ci_95": None}))
            continue

        # Surrounding polarity, computed with the candidate's own term removed from the seed
        # lexicon: otherwise a term that is already in the seed would validate itself.
        seed_no_term = seed_lexicon.model_copy(
            update={"terms": [t for t in seed_lexicon.terms if t.term.lower() != term.term.lower()]}
        )
        matcher = LexiconMatcher(seed_no_term)
        context = np.array(
            [matcher.polarity_of(text)[0] for text in texts[mask]],
            dtype=float,
        )
        decisive = context[np.abs(context) >= label_band]
        record["context_nonzero"] = int(len(decisive))
        if len(decisive) == 0:
            rejected.append({**record, "rule": "surrounding text carries no polarity signal to agree with"})
            continue

        wanted = 1.0 if term.polarity > 0 else -1.0
        agreement = float(np.mean(np.sign(decisive) == wanted))
        record["sign_agreement"] = round(agreement, 4)
        if agreement < sign_agreement:
            rejected.append({**record, "rule": f"sign agreement {agreement:.2f} < {sign_agreement}"})
            continue

        ci = _bootstrap_mean_ci(
            decisive,
            session_ids[mask.to_numpy()][np.abs(context) >= label_band],
            block_sessions,
            resamples,
            seed,
        )
        record["ci_95"] = ci
        if not (math.isfinite(ci[0]) and math.isfinite(ci[1])):
            # Not estimable (too few sessions to resample): reject rather than accept. A gate that
            # passes when it cannot measure is not a gate.
            rejected.append({**record, "rule": "bootstrap CI could not be estimated on this slice"})
            continue
        if ci[0] <= 0.0 <= ci[1]:
            rejected.append({**record, "rule": "bootstrap CI for surrounding polarity includes zero"})
            continue

        accepted.append(term.model_copy(update={"support": support, "sign_agreement": round(agreement, 4), "ci_95": ci}))

    return accepted, rejected


def build_induction_prompt(
    settings: Settings,
    sample: pd.DataFrame,
    canonical_report: dict[str, Any],
    warnings: list[str],
    seed_lexicon: FrozenLexicon,
) -> list[dict[str, str]]:
    """Build the two messages sent to the inductor. Kept separate so it can be reviewed as text."""
    existing = ", ".join(sorted({t.term for t in seed_lexicon.terms})[:200])
    schema = LexiconProposal.model_json_schema()
    system = (
        "You are a financial-text lexicographer. You read stock-tweet samples and propose "
        "sentiment vocabulary. You never score individual tweets and never state facts about "
        "returns. Your entire answer is one JSON object matching the schema you are given."
    )
    user = f"""Corpus facts (measured, not assumed):
- window {canonical_report['first_session']} -> {canonical_report['last_session']}, {canonical_report['sessions']} sessions
- {canonical_report['rows_in_window']:,} tweets in window; {canonical_report['rolled_share']:.1%} roll to the next session
- {canonical_report['shared_texts']:,} texts are cross-posted under more than one ticker

Known data warnings:
{chr(10).join('- ' + w for w in warnings) or '- none'}

Terms already in the shipped lexicon (extend this list; do not just repeat it):
{existing}

Sample ({len(sample)} tweets, stratified by promo / reply / cashtag-dense, drawn by hash):
{format_sample(sample)}

Propose at most {_MAX_TERMS_REQUESTED} terms that this corpus uses and the shipped lexicon misses.
Rules:
- family: "polarity" (general signed sentiment), "bullish" or "bearish" (finance directional),
  "uncertainty" (hedging), "attribution" (reported speech), "sarcasm" (marker).
- polarity in [-1,1]: the sentiment the term contributes. weight in [0,1]: how strongly.
- Every term must appear in the sample you were given; cite the ticker and a few words of the
  tweet in `evidence_tweet_ids`.
- Prefer multi-word phrases when the single word is ambiguous in finance.
- Put anything you deliberately did not propose, and why, in `abstain_reasons`.
- Reply with the JSON object only. Schema:
{json.dumps(schema, separators=(',', ':'))}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_induction(
    settings: Settings,
    canonical: Any,
    seed_lexicon: FrozenLexicon,
    run_dir: Path,
    client: LLMClient,
    enabled: bool = True,
    warnings: list[str] | None = None,
) -> InductionResult:
    """Run the induction stage and return the lexicon a run should use.

    Never raises for an LLM reason: every failure path returns the shipped default with the reason
    recorded, because a demo that cannot run without an endpoint would violate design rule 3.
    """
    induction_cfg = settings.agent("induction")
    experiment_cfg = settings.experiments.get("induction", {}) or {}
    seed = int(settings.experiments.get("bootstrap", {}).get("seed", 20260912))
    warnings = warnings if warnings is not None else canonical.report.get("warnings", [])

    sample = sample_for_induction(
        canonical.tweets,
        per_ticker=int(induction_cfg.get("per_ticker", 60)),
        max_total=int(induction_cfg.get("max_total", 1500)),
        validation_parity=int(experiment_cfg.get("validation_parity", 1)),
        strata_caps=experiment_cfg.get("strata_caps", {}) or {},
    )
    report: dict[str, Any] = {"sample": sample.report, "llm": client.describe()}

    if not enabled or not client.available:
        report["outcome"] = "llm unavailable -> shipped default lexicon"
        default_path = default_lexicon_path(settings)
        return InductionResult(
            lexicon=seed_lexicon,
            provenance={"source": "default", "lexicon_id": seed_lexicon.lexicon_id,
                        "path": str(default_path),
                        "reason": "LLM not configured or disabled"},
            accepted=[],
            rejected=[],
            report=report,
        )

    messages = build_induction_prompt(
        settings,
        sample.sample,
        canonical.report,
        warnings,
        seed_lexicon,
    )
    schema = LexiconProposal.model_json_schema()
    proposal: LexiconProposal | None = None
    raw: dict[str, Any] = {}
    # A proposal of up to 60 terms is a long document; the per-agent budget is explicit so a long
    # answer is a config decision rather than a truncation the caller discovers afterwards.
    budget = int(induction_cfg.get("max_tokens", 8000))
    try:
        payload, response = client.complete_json(messages, schema, purpose="induction", max_tokens=budget)
        proposal = LexiconProposal.model_validate(payload)
        raw = {"mode": response.mode, "cached": response.cached, "prompt_hash": response.prompt_hash,
               "repaired": response.repaired}
    except (LLMError, ValidationError) as first_error:
        # One repair attempt, with the validator's own message, then give up and fall back.
        report["first_error"] = str(first_error)[:400]
        try:
            messages = messages + [
                {"role": "assistant", "content": "(previous answer rejected)"},
                {"role": "user", "content": f"That answer was rejected: {first_error}. Reply with corrected JSON only."},
            ]
            payload, response = client.complete_json(messages, schema, purpose="induction_repair", max_tokens=budget)
            proposal = LexiconProposal.model_validate(payload)
            raw = {"mode": response.mode, "cached": response.cached, "prompt_hash": response.prompt_hash,
                   "repaired": response.repaired, "repaired_call": True}
        except (LLMError, ValidationError) as second_error:
            report["second_error"] = str(second_error)[:400]
            report["outcome"] = "two invalid answers -> shipped default lexicon"
            _write_rejected(run_dir, {"proposal": None, "accepted": [], "rejected": [], "reason": report["second_error"]})
            return InductionResult(
                lexicon=seed_lexicon,
                provenance={"source": "default", "lexicon_id": seed_lexicon.lexicon_id,
                            "path": "", "reason": "induction failed twice; default kept"},
                accepted=[], rejected=[], report=report,
            )

    assert proposal is not None  # only reachable with a validated proposal
    accepted, rejected = gate_terms(
        proposal.terms,
        sample.validation,
        seed_lexicon,
        min_support=int(induction_cfg.get("min_support", 10)),
        sign_agreement=float(induction_cfg.get("sign_agreement", 0.6)),
        resamples=int(experiment_cfg.get("bootstrap_resamples", 1000)),
        block_sessions=int(experiment_cfg.get("bootstrap_block_sessions", 5)),
        seed=seed,
        label_band=float((settings.raw.get("sentiment", {}) or {}).get("label_band", 0.05)),
    )
    report["proposal"] = {
        "terms_proposed": len(proposal.terms),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "relevance_rules": len(proposal.relevance_rules),
        "abstain_reasons": proposal.abstain_reasons[:10],
        **raw,
    }

    rejected_path = _write_rejected(
        run_dir,
        {
            "model": client.config.model,
            "prompt_hash": raw.get("prompt_hash", ""),
            "proposed": [t.model_dump() for t in proposal.terms],
            "accepted": [t.model_dump() for t in accepted],
            "rejected": rejected,
        },
    )
    report["rejected_proposals_path"] = str(rejected_path)

    if not accepted:
        report["outcome"] = "no proposed term survived the gate -> shipped default lexicon"
        return InductionResult(
            lexicon=seed_lexicon,
            provenance={"source": "default", "lexicon_id": seed_lexicon.lexicon_id, "path": "",
                        "reason": "no proposed term survived the validation gate"},
            accepted=[], rejected=rejected, report=report,
        )

    # Additive, never subtractive (SPEC.md 7.3 step 4): the induced lexicon is the shipped core
    # plus the accepted terms. A model list of 20 words cannot silently delete 160 hand-checked
    # ones, and a term that passed the gate overrides the shipped entry of the same name.
    merged = {t.term.lower(): t for t in seed_lexicon.terms}
    for term in accepted:
        merged[term.term.lower()] = term
    terms = [merged[key] for key in sorted(merged)]
    lexicon_id = lexicon_id_for(terms, experiment_cfg, client.config.model, seed)
    frozen = FrozenLexicon(
        lexicon_id=lexicon_id,
        source="induced",
        version=str(int(seed_lexicon.version) + 1) if seed_lexicon.version.isdigit() else "1",
        created=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        sentiment_method=seed_lexicon.sentiment_method,
        model=client.config.model,
        notes=[
            f"induced from a {sample.report['sample_rows']}-tweet stratified sample on the full corpus",
            f"gate: support >= {induction_cfg.get('min_support', 10)}, sign agreement >= {induction_cfg.get('sign_agreement', 0.6)}, bootstrap CI excludes zero",
            f"{len(accepted)} accepted, {len(rejected)} rejected; additive on top of default_v1",
            "the gate measures agreement with the polarity of surrounding text, not human labels",
        ],
        negation_patterns=proposal.negation_patterns or seed_lexicon.negation_patterns,
        relevance_rules=proposal.relevance_rules,
        abstain_reasons=proposal.abstain_reasons,
        terms=terms,
    )
    path = write_lexicon(frozen, REPO_ROOT / "artifacts" / "lexicons" / f"{lexicon_id}.json")
    set_active_lexicon(settings, frozen, path)
    report["outcome"] = f"induced lexicon {lexicon_id} written and activated"
    return InductionResult(
        lexicon=frozen,
        provenance={"source": "induced", "lexicon_id": lexicon_id, "path": str(path),
                    "reason": "induced and validated in this run"},
        accepted=accepted,
        rejected=rejected,
        report=report,
        lexicon_path=path,
    )


def _write_rejected(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Write `rejected_proposals.json` (SPEC.md 4.3 step 4) and return its path."""
    path = run_dir / "rejected_proposals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def default_lexicon(settings: Settings) -> FrozenLexicon:
    """Load the shipped `default_v1.json` regardless of any active induced lexicon.

    `induce` seeds its proposal from this file rather than from the active one, so a second
    induction extends the reviewed baseline instead of compounding its own earlier output.
    """
    from ..features.lexicon import load_lexicon

    return load_lexicon(default_lexicon_path(settings))
