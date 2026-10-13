"""The two contracts that make this demo's numbers traceable: `ToolResult` and `ChartSpec`.

Design decision embodied here: **a tool returns data, never a figure.** A `ChartSpec` is a
declarative description (kind, axis label, series, caption) that either renderer can draw. That is
what keeps the whole analysis layer unit-testable without a display, and what makes the chart set
changeable without touching an analysis.

Second decision: **`result_id` is a hash of the tool, its canonical parameters and the data
version.** The same inputs therefore produce the same id, a re-run can be diffed, and a number in
prose can always be traced to the one call that computed it. Prose and rendered figure bytes are
deliberately not part of it (SPEC.md 7.2).

`n_obs` and `n_units` are both reported and they are not the same number: `n_obs` counts rows the
tool used, `n_units` counts independent ticker-sessions. Reporting only the first would overstate
the evidence by an order of magnitude.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChartKind = Literal["line", "bar", "bar_error", "hist", "scatter", "heatmap", "box", "dual_line"]


class ChartSpec(BaseModel):
    """A declarative, JSON-safe chart description. No figure object, no backend assumed."""

    model_config = ConfigDict(extra="forbid")

    kind: ChartKind
    title: str
    x: dict[str, Any] = Field(default_factory=dict)                 # {label, values}
    series: list[dict[str, Any]] = Field(default_factory=list)      # [{label, values, kind?}]
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    source_result_id: str = ""
    caption: str = ""


class ToolResult(BaseModel):
    """The uniform result of every analysis (SPEC.md 4.8)."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    period: str = "discovery"
    n_obs: int = 0
    n_units: int = 0
    n_dropped: int = 0
    stats: dict[str, Any] = Field(default_factory=dict)
    table: list[dict[str, Any]] | None = None
    chart: ChartSpec | None = None
    caveats: list[str] = Field(default_factory=list)
    data_version: str = ""
    # Which result_ids this one derives from, so a holdout check can be traced back to its test.
    parent_result_ids: list[str] = Field(default_factory=list)

    def cited_numbers(self) -> list[float]:
        """Return every numeric value in `stats` that a claim may legitimately cite.

        The claim checker compares a number written in prose against this list, so an agent
        cannot cite a figure that no tool produced. Nested dicts and lists are flattened, because
        a CI is a list and a bucket table is a list of dicts.
        """
        numbers: list[float] = []

        def walk(value: Any) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                numbers.append(float(value))
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)

        walk(self.stats)
        return numbers


class ClaimCheck(BaseModel):
    """One claim-checker verdict, kept even when it passes so the log is auditable."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    verdict: Literal["accepted", "downgraded", "rejected"]
    reason: str = ""
    action: str = ""
