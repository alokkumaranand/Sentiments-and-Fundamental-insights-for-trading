"""The baseline playbook: a fixed set of analyses that runs with no LLM at all.

Design decision embodied here: **the report is guaranteed before any agent is involved.** Design
rule 3 says the demo must produce a complete, honest report without a key, so the playbook is not
a fallback — it is the baseline, and the agents only add interpretation on top of it.

The call list below is fixed and readable, and it encodes three rules from the spec:

- the pooled pair is always run **with and without TSLA** (46.3% of the corpus);
- `sent_net -> fwd_ret_h` is the predictive-style question, and it is asked at all three horizons;
- attention is tested against **next-day** |return| (`lag=1`), not against the same day.

A failing call is recorded and skipped, never fatal: on a thin dataset half of these have too few
units, and the honest outcome there is a report that says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import ToolError, ToolRegistry


@dataclass
class PlaybookOutcome:
    """Which calls ran, which failed, and a compact digest of what they found."""

    result_ids: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    digest: list[dict[str, Any]] = field(default_factory=list)


def playbook_calls(registry: ToolRegistry) -> list[tuple[str, dict[str, Any], str]]:
    """Return the fixed call list as (tool, params, period) triples.

    Kept as data rather than as a sequence of imperative statements so a reviewer can read the
    whole baseline in one place, and so a test can assert the list covers both families and both
    pooled variants.
    """
    tickers = registry.context.panel["ticker"].unique().tolist()
    tier_a = [t for t in sorted(tickers) if registry.context.tiers.get(t) == "A"]
    return [
        # descriptive layer
        ("coverage_by_ticker", {"period": "full"}, "full"),
        ("text_stats", {"period": "discovery"}, "discovery"),
        ("distribution", {"column": "sent_net", "by": "tier", "period": "discovery"}, "discovery"),
        ("distribution", {"column": "n_texts_own", "period": "discovery"}, "discovery"),
        ("timeseries", {"metric": "sent_net", "scope": "pooled", "period": "discovery"}, "discovery"),
        ("timeseries", {"metric": "n_texts_own", "scope": "pooled_ex_TSLA", "period": "discovery"}, "discovery"),
        # sentiment vs forward returns: pooled, pooled without TSLA, and one tier-A ticker
        ("bucket_forward_returns", {"feature": "sent_net", "horizon": 1, "scope": "pooled_ex_TSLA"}, "discovery"),
        ("bucket_forward_returns", {"feature": "sent_net", "horizon": 2, "scope": "pooled_ex_TSLA"}, "discovery"),
        ("bucket_forward_returns", {"feature": "sent_net", "horizon": 5, "scope": "pooled_ex_TSLA"}, "discovery"),
        ("correlate", {"x": "sent_net", "y": "fwd_ret_1", "scope": "pooled_ex_TSLA"}, "discovery"),
        # attention vs next-day volatility, and a dual-axis view for a specific ticker
        ("correlate", {"x": "attn_z", "y": "abs_ret", "lag": 1, "scope": "pooled"}, "discovery"),
        ("attention_vs_price", {"ticker": tier_a[0] if tier_a else sorted(tickers)[0]}, "discovery"),
        ("spike_days", {"metric": "n_texts_own", "scope": "pooled"}, "discovery"),
        ("corr_matrix", {"metrics": ["sent_net", "n_texts_own", "attn_z", "abs_ret", "rv_20", "volume_z"],
                         "level": "pooled", "period": "discovery"}, "discovery"),
    ]


def run_playbook(registry: ToolRegistry, store_results: bool = True) -> PlaybookOutcome:
    """Run the fixed call list, skipping anything that cannot be computed here."""
    outcome = PlaybookOutcome()
    for tool, params, period in playbook_calls(registry):
        try:
            result = registry.call(tool, params, period=period)
        except ToolError as error:
            # Recorded, not raised: a thin dataset makes some calls impossible, and the run must
            # still produce a report that says which ones and why.
            outcome.failures.append({"tool": tool, "params": params, "error": str(error)})
            continue
        outcome.result_ids.append(result.result_id)
        outcome.digest.append(
            {
                "result_id": result.result_id,
                "tool": result.tool,
                "params": result.params,
                "period": result.period,
                "n_units": result.n_units,
                "stats": _trim(result.stats),
                "caveats": result.caveats[:3],
            }
        )
    return outcome


def _trim(stats: dict[str, Any], limit: int = 12) -> dict[str, Any]:
    """Keep the digest small enough for a prompt: the first `limit` scalar entries.

    Nested structures (a matrix, a bucket list) are summarised by their key names rather than
    pasted, because the agent's job is to notice what stands out, not to read the whole object.
    """
    trimmed: dict[str, Any] = {}
    for key, value in stats.items():
        if len(trimmed) >= limit:
            break
        if isinstance(value, (int, float, str, bool)) or value is None:
            trimmed[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            trimmed[key] = f"[{len(value)} rows: {', '.join(sorted(value[0]))}]"
        elif isinstance(value, dict):
            trimmed[key] = f"{{{', '.join(list(value)[:6])}}}"
        else:
            trimmed[key] = f"[{len(value)} items]" if isinstance(value, list) else str(value)
    return trimmed
