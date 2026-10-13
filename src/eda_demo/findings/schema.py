"""The finding contract: simplified from the full PRD insight schema so it can be upgraded later.

Design decision embodied here: **unknown values stay null.** Every field an agent might be tempted
to fill in with something plausible — the effect size, the CI, `n_units`, the q-value — is filled in
by code from a `ToolResult`, and stays `null` until a tool has computed it. That is what stops a
confident-looking number from appearing in a finding that no call supports.

`FindingDraft` is what an agent may return; `Finding` is what the pipeline writes after the claim
checker and the status rules have run. The two are separate types so an unchecked draft can never
be presented as a finding.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["observed", "supported", "not_replicated", "inconclusive"]
Origin = Literal["playbook", "null_agent", "explorer_agent"]

SCHEMA_VERSION = "0.2"


class Evidence(BaseModel):
    """One cited tool result, and the role it plays in the finding."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    role: Literal["primary", "supporting", "holdout"] = "primary"


class FindingScope(BaseModel):
    """What the finding is about, so a number can never be read out of context."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    period: str = "discovery"
    tier: str | None = None


class FindingEffect(BaseModel):
    """The measured effect. Every field is null until a tool computes it."""

    model_config = ConfigDict(extra="forbid")

    metric: str | None = None
    value: float | None = None
    ci_95: list[float | None] | None = None
    n_units: int | None = None
    q_value: float | None = None


class FindingDraft(BaseModel):
    """What an agent may propose. Unverified: it carries no status and no effect numbers."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=4, max_length=160)
    claim_text: str = Field(min_length=10)
    evidence: list[Evidence] = Field(default_factory=list)
    scope: FindingScope = Field(default_factory=FindingScope)
    limitations: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    rationale: str = ""


class Finding(BaseModel):
    """A checked finding, ready to present."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    finding_id: str
    title: str
    scope: FindingScope
    claim_text: str
    evidence: list[Evidence]
    effect: FindingEffect
    status: Status
    limitations: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    origin: Origin = "playbook"
    run_id: str = ""
    lexicon_id: str = ""
    # The claim-checker verdict travels with the finding instead of being logged separately, so
    # a reader always sees that this text passed verification and what was changed.
    verification: dict = Field(default_factory=dict)


class FindingsFile(BaseModel):
    """The `findings.json` payload: the findings plus how the run went."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    data_version: str
    lexicon_id: str
    generated_at: str
    findings: list[Finding] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
