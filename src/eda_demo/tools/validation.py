"""Validation tools: the holdout re-run and the multiplicity adjustment.

Design decision embodied here: **the coordinator triggers these, never an agent** (SPEC.md 4.8).
Discovery/holdout discipline is enforced in code rather than in a prompt, because a model that
could choose to check the holdout could also choose to check it twenty times and report the one
that agreed.

`holdout_check` re-runs the *same* tool with the *same* parameters on the other period, which is
possible only because every tool is a pure function of the panel and validated parameters. Its
result is a new `ToolResult` whose `parent_result_ids` names the discovery result it tested.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..stats.inference import benjamini_hochberg
from .common import bar_chart, numeric
from .registry import ToolContext, ToolError, ToolParams
from .schema import ToolResult

# Tool names whose output is a test with a p-value, so the coordinator knows what to adjust.
TEST_TOOLS = ("correlate", "bucket_forward_returns")


class HoldoutParams(ToolParams):
    """Parameters of `holdout_check`."""

    result_id: str = Field(description="result_id of a discovery-period result to re-run")


class AdjustParams(ToolParams):
    """Parameters of `adjust_pvalues`."""

    result_ids: list[str] = Field(default_factory=list, description="result_ids of tests to adjust together")


def register(registry: Any) -> None:
    """Register the validation tools on `registry`."""

    @registry.add("holdout_check", HoldoutParams,
                  "Re-runs one discovery-period result on the holdout period and compares.",
                  chart_kind="bar")
    def holdout_check(context: ToolContext, params: HoldoutParams) -> dict[str, Any]:
        parent = registry.store.get(params.result_id)
        if parent is None:
            raise ToolError(
                f"result_id '{params.result_id}' is not in this run's store. "
                f"Known ids: {registry.store.order[:8]}"
            )
        if parent.period != "discovery":
            raise ToolError(
                f"holdout_check expects a discovery-period result; '{params.result_id}' is {parent.period}"
            )
        child = registry.call(parent.tool, parent.params, period="holdout", parent_result_ids=[parent.result_id])
        discovery_value, holdout_value, metric = _comparable_values(parent, child)
        same_sign = None
        if discovery_value is not None and holdout_value is not None:
            same_sign = (discovery_value > 0) == (holdout_value > 0)
        stats = {
            "tested_result_id": parent.result_id,
            "tool": parent.tool,
            "metric": metric,
            "discovery": {"value": discovery_value, "n_units": parent.n_units, "period": parent.period},
            "holdout": {"value": holdout_value, "n_units": child.n_units, "period": child.period},
            "same_sign": same_sign,
            "holdout_result_id": child.result_id,
        }
        chart = bar_chart(
            f"{metric}: discovery vs holdout",
            "period",
            ["discovery", "holdout"],
            [{"label": metric, "values": [discovery_value, holdout_value]}],
            "full",
            parent.n_units + child.n_units,
            extra_caption=f"re-runs {parent.tool} with identical parameters on the holdout sessions",
        )
        return {
            "n_obs": int(child.n_obs),
            "n_units": int(child.n_units),
            "stats": stats,
            "table": [
                {"period": "discovery", "value": discovery_value, "n_units": parent.n_units},
                {"period": "holdout", "value": holdout_value, "n_units": child.n_units},
            ],
            "chart": chart,
            "caveats": [
                "the holdout is the last 30% of sessions: fewer sessions and one regime, so a sign flip is weak evidence either way",
                "the holdout period is used only through this tool, which the coordinator triggers",
            ],
        }

    @registry.add("adjust_pvalues", AdjustParams,
                  "Benjamini-Hochberg q-values for a batch of tested results.")
    def adjust_pvalues(context: ToolContext, params: AdjustParams) -> dict[str, Any]:
        ids = params.result_ids or [
            result.result_id for result in registry.store.all() if result.tool in TEST_TOOLS and result.period == "discovery"
        ]
        rows: list[dict[str, Any]] = []
        pvalues: list[float] = []
        for result_id in ids:
            result = registry.store.get(result_id)
            if result is None or "p_value" not in result.stats:
                continue
            value = result.stats.get("p_value")
            pvalues.append(float("nan") if value is None else float(value))
            rows.append({"result_id": result_id, "tool": result.tool, "p_value": value})
        qvalues = benjamini_hochberg(pvalues)
        for row, qvalue in zip(rows, qvalues):
            row["q_value"] = None if qvalue != qvalue else qvalue
            row["significant"] = bool(qvalue == qvalue and qvalue < float(
                context.settings.experiments.get("multiple_testing", {}).get("alpha", 0.05)
            ))
        alpha = float(context.settings.experiments.get("multiple_testing", {}).get("alpha", 0.05))
        stats = {
            "method": context.settings.experiments.get("multiple_testing", {}).get("method", "benjamini_hochberg"),
            "alpha": alpha,
            "n_tests": len(rows),
            "n_significant": int(sum(1 for row in rows if row["significant"])),
            "tested_result_ids": [row["result_id"] for row in rows],
        }
        return {
            "n_obs": len(rows),
            "n_units": len(rows),
            "stats": stats,
            "table": rows,
            "caveats": [
                f"BH controls the false discovery rate across the {len(rows)} tests listed, not the probability that any one is true",
                f"at alpha={alpha} with {len(rows)} tests, about {max(1, round(alpha * len(rows)))} would look significant by chance",
            ],
        }


def _comparable_values(parent: ToolResult, child: ToolResult) -> tuple[Any, Any, str]:
    """Pick the scalar both periods share, preferring a correlation then a bucket spread.

    Returns (discovery_value, holdout_value, metric_name). A result with no comparable scalar
    returns (None, None, ...) rather than inventing one.
    """
    for key in ("spearman_rho", "spread_top_minus_bottom"):
        if key in parent.stats and key in child.stats:
            return (
                numeric(parent.stats[key]),
                numeric(child.stats[key]),
                key,
            )
    return None, None, "value"
