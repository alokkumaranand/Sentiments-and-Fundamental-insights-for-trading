"""The Presenter agent: prose over findings that are already checked, and nothing else.

Design decision embodied here: **the presenter cannot introduce a number.** It receives only
findings that survived the claim checker, together with the stats of the results they cite, and its
output is checked again by the same arithmetic rule. A paragraph that cites a number no tool
produced is discarded and replaced with the templated one, and that replacement is recorded rather
than hidden.

Its style contract is the one the spec asks for: what was observed, how strong, how uncertain, and
what the result does **not** show. Advisory language is a hard failure here too.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..findings.checker import check_numbers_in_text
from .llm import LLMClient, LLMError
from .null_agent import template_narrative

_SYSTEM = (
    "You write the narrative section of an exploratory data analysis report for a technical reader. "
    "You are given findings that have already been verified. Rules: every sentence containing a "
    "number must carry the [R:<result_id>] of the result that contains it; never introduce a number "
    "that is not in the supplied stats; never claim causation; never give trading advice or a "
    "direction; state plainly what the result does not show. Neutral wording: 'largest observed "
    "change', 'the estimate is imprecise'."
)


class Narrative(BaseModel):
    """The presenter's output contract."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=10)
    paragraphs: dict[str, str] = Field(default_factory=dict, description="finding_id -> paragraph")
    what_to_look_at_next: list[str] = Field(default_factory=list)
    captions: dict[str, str] = Field(default_factory=dict, description="result_id -> chart caption")


def run_presenter(
    client: LLMClient,
    findings: list[dict[str, Any]],
    stats_by_result: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (narrative, rejections).

    The narrative always exists: on any failure the templated text is used, and the reason is
    returned so the manifest and the report can state which paragraphs were model-written.
    """
    rejections: list[dict[str, Any]] = []
    if not client.available or not findings:
        narrative = template_narrative(findings, llm_enabled=False)
        if not client.available:
            narrative["note"] = "LLM disabled: narrative is templated from tool results"
        return narrative, rejections

    allowed = _allowed_numbers(stats_by_result)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Verified findings:\n"
                + json.dumps(findings, indent=1, default=str)
                + "\n\nStats of the results they cite (these are the only numbers you may use):\n"
                + json.dumps(stats_by_result, indent=1, default=str)
                + "\n\nReturn JSON with: summary (3-5 sentences, each numeric sentence carrying its "
                "[R:<result_id>] tag), paragraphs (finding_id -> one paragraph: what was observed, how "
                "strong, how uncertain, what it does not show), what_to_look_at_next (2-4 items), "
                "and captions (result_id -> one-line chart caption)."
            ),
        },
    ]
    try:
        payload, response = client.complete_json(messages, Narrative.model_json_schema(), purpose="presenter")
        narrative_model = Narrative.model_validate(payload)
    except (LLMError, ValidationError, json.JSONDecodeError) as error:
        return (
            {
                **template_narrative(findings, llm_enabled=True),
                "note": f"presenter failed, templated fallback used: {str(error)[:160]}",
            },
            [{"finding_id": "narrative", "verdict": "rejected", "reason": f"presenter output unusable: {str(error)[:160]}"}],
        )

    narrative: dict[str, Any] = {
        "summary": narrative_model.summary,
        "paragraphs": dict(narrative_model.paragraphs),
        "what_to_look_at_next": narrative_model.what_to_look_at_next,
        "captions": dict(narrative_model.captions),
        "source": "presenter_agent",
        "mode": response.mode,
        "cached": response.cached,
    }
    fallback = template_narrative(findings, llm_enabled=True)
    known_ids = list(stats_by_result)

    unmatched = check_numbers_in_text(narrative_model.summary, allowed, known_ids)
    if unmatched:
        rejections.append({
            "finding_id": "summary",
            "verdict": "rejected",
            "reason": f"summary cited number(s) present in no cited stat: {', '.join(unmatched[:5])}",
        })
        narrative["summary"] = fallback["summary"]

    for finding in findings:
        identifier = finding["finding_id"]
        paragraph = narrative_model.paragraphs.get(identifier)
        if not paragraph:
            continue
        unmatched = check_numbers_in_text(paragraph, allowed, known_ids)
        if unmatched:
            rejections.append({
                "finding_id": identifier,
                "verdict": "rejected",
                "reason": f"paragraph cited number(s) present in no cited stat: {', '.join(unmatched[:5])}",
            })
            narrative["paragraphs"][identifier] = fallback["paragraphs"].get(identifier, "")
    return narrative, rejections


def _allowed_numbers(stats_by_result: dict[str, dict[str, Any]]) -> set[float]:
    """Collect every number the presenter may cite, from the supplied stats."""

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            numbers.add(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    numbers: set[float] = set()
    walk(stats_by_result)
    # n_units is not in stats but is a legitimate thing to state about a result.
    return numbers
