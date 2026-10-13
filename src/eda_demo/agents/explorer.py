"""The Explorer agent: it picks follow-up analyses and proposes findings, and nothing else.

Design decision embodied here: **the agent cannot escape the registry or the discovery period.**
It receives the tool schemas, chooses calls, and gets back `ToolResult`s. Two guardrails are
enforced in code rather than asked for in a prompt:

- a tool call that is not in the registry is answered with an error message the model can recover
  from, and counted;
- `period` is forced to `discovery` on every call, so the agent physically cannot read the holdout
  (SPEC.md 4.8: discovery/holdout discipline is enforced in code, not in prompts).

Its output is a list of `FindingDraft`s. Those are unverified: the claim checker and the status
rules run afterwards, and the drafts it produces can be downgraded or thrown away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..findings.schema import FindingDraft
from ..tools.playbook import _trim
from ..tools.registry import ToolError, ToolRegistry
from ..tools.schema import ToolResult
from .llm import LLMClient, LLMError

# The agent sees tool names and parameters, never the panel and never a row of data.
_SYSTEM = (
    "You are an exploratory data analyst working on a stock-tweet and daily-price dataset. "
    "You choose follow-up analyses from a fixed tool registry and then propose findings. "
    "Rules: cite only result_ids you were given; never state a number that is not in a cited "
    "result's stats; never claim causation; never use buy/sell/profit language; a null result is a "
    "valid finding. You may only read the discovery period."
)


@dataclass
class ExplorerOutcome:
    """Drafts, the calls the agent made, and every reason a call or draft was refused."""

    drafts: list[FindingDraft] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _drafts_schema() -> dict[str, Any]:
    """Return the JSON schema the agent's final answer must match."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 6,
                "items": FindingDraft.model_json_schema(),
            }
        },
    }


def _tool_message(result: ToolResult) -> str:
    """Render a `ToolResult` compactly for the model: stats and caveats, never raw rows."""
    return json.dumps(
        {
            "result_id": result.result_id,
            "tool": result.tool,
            "params": result.params,
            "period": result.period,
            "n_units": result.n_units,
            "n_dropped": result.n_dropped,
            "stats": _trim(result.stats, limit=20),
            "caveats": result.caveats,
        },
        default=str,
    )


def run_explorer(
    client: LLMClient,
    registry: ToolRegistry,
    digest: list[dict[str, Any]],
    max_tool_calls: int,
    retries_on_schema_error: int,
    playbook_failures: list[dict[str, Any]] | None = None,
) -> ExplorerOutcome:
    """Run the explorer loop and return its unverified finding drafts.

    Never raises for an LLM reason: a failed agent call returns an empty draft list and the caller
    falls back to the templated path, so the run still produces a complete report.
    """
    outcome = ExplorerOutcome()
    if not client.available:
        outcome.report["outcome"] = "llm unavailable: explorer skipped"
        return outcome

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                "Baseline playbook results (period=discovery unless stated):\n"
                + json.dumps(digest, indent=1, default=str)
                + "\n\nBaseline calls that could not run here:\n"
                + json.dumps(playbook_failures or [], indent=1, default=str)
                + "\n\nUse the tools to drill into what stands out — a specific ticker, a spike day, "
                "a bucket comparison. You have a hard budget of "
                f"{max_tool_calls} tool calls. When you are done, answer with JSON only: "
                '{"findings": [{"title", "claim_text", "evidence": [{"result_id", "role"}], '
                '"scope": {"tickers", "period", "tier"}, "limitations", "alternative_explanations", '
                '"rationale"}]}. Cite result_ids that exist; keep every number inside a cited stat.'
            ),
        },
    ]

    schemas = registry.schemas()
    calls = 0
    schema_retries = 0
    while calls < max_tool_calls:
        try:
            response = client.complete(messages, tools=schemas, purpose="explorer")
        except LLMError as error:
            outcome.errors.append({"stage": "explorer_call", "error": str(error)})
            outcome.report["outcome"] = f"explorer call failed: {error}"
            return outcome

        if not response.tool_calls:
            break
        messages.append({"role": "assistant", "content": response.text or "", "tool_calls": [
            {"id": f"call_{calls + index}", "type": "function",
             "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
            for index, call in enumerate(response.tool_calls)
        ]})
        for index, call in enumerate(response.tool_calls):
            if calls >= max_tool_calls:
                break
            calls += 1
            name = call.get("name", "")
            arguments = dict(call.get("arguments") or {})
            # Guardrail: the agent may not choose the period. Overriding rather than refusing keeps
            # the call useful while making the holdout unreachable from here.
            if arguments.pop("period", None) not in (None, "discovery"):
                outcome.errors.append({"stage": "explorer_guardrail", "tool": name,
                                       "reason": "period forced to discovery"})
            arguments["period"] = "discovery"
            try:
                result = registry.call(name, arguments, period="discovery")
                outcome.calls.append({"tool": name, "params": result.params, "result_id": result.result_id})
                content = _tool_message(result)
            except ToolError as error:
                outcome.errors.append({"stage": "explorer_tool", "tool": name, "error": str(error)})
                if schema_retries < retries_on_schema_error:
                    schema_retries += 1
                content = json.dumps({"error": str(error), "registered_tools": registry.names()})
            messages.append({"role": "tool", "tool_call_id": f"call_{calls - 1}", "content": content})

    try:
        payload, response = client.complete_json(messages, _drafts_schema(), purpose="explorer_findings")
    except (LLMError, json.JSONDecodeError) as error:
        outcome.errors.append({"stage": "explorer_findings", "error": str(error)[:300]})
        outcome.report["outcome"] = f"explorer produced no parseable findings: {str(error)[:160]}"
        outcome.report["tool_calls"] = calls
        outcome.report["mode"] = client.mode
        return outcome

    drafts: list[FindingDraft] = []
    for raw in payload.get("findings", []) if isinstance(payload, dict) else []:
        try:
            drafts.append(FindingDraft.model_validate(raw))
        except ValidationError as error:
            # A malformed draft is dropped and counted; it is never repaired into something the
            # model did not say, because the schema is the model's side of the contract.
            outcome.errors.append({"stage": "explorer_draft", "error": error.errors()[:2], "raw": str(raw)[:200]})
    outcome.drafts = drafts
    outcome.report.update(
        outcome="explorer returned "
        f"{len(drafts)} draft(s) after {calls} tool call(s)",
        tool_calls=calls,
        mode=response.mode,
        cached=response.cached,
    )
    return outcome
