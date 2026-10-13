"""Tool registry: typed parameters in, a uniform `ToolResult` out.

Design decision embodied here: **agents act only through registered tools** (SPEC.md 1.4 rule 2).
A tool is a pure function of the panel plus validated parameters; the registry owns parameter
validation, the result id, the result store and the standard caveats. Nothing else in the system
is callable by an agent, and no tool reaches the raw dataset.

Second decision: **every result carries the caveats of its own selection.** How many units were
excluded, which period was read, which tickers were pooled and whether TSLA was removed are
appended by the registry rather than remembered by each tool, so a new tool cannot forget them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import Settings
from .schema import ChartSpec, ToolResult


class ToolParams(BaseModel):
    """Base class for tool parameters: unknown keys are an error, not a silent ignore.

    A model that invents a parameter must be told, because silently dropping `horizon` would
    return a result that answers a different question than the one asked.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass
class Selection:
    """The rows a tool may use, plus the counts and caveats of that choice."""

    frame: pd.DataFrame
    n_obs: int
    n_units: int
    n_dropped: int
    caveats: list[str] = field(default_factory=list)


@dataclass
class ToolContext:
    """Everything a tool may read: the panel, the scored tweets, config and the data version."""

    panel: pd.DataFrame
    tweets: pd.DataFrame
    settings: Settings
    data_version: str
    tiers: dict[str, str]
    calendar: list[pd.Timestamp]

    def period_frame(self, period: str) -> pd.DataFrame:
        """Return the panel restricted to discovery, holdout or the whole window."""
        if period == "full":
            return self.panel
        return self.panel[self.panel["period"] == period]

    def select(
        self,
        period: str = "discovery",
        scope: str = "pooled",
        ticker: str | None = None,
        require: tuple[str, ...] = (),
    ) -> Selection:
        """Return the rows to use and a record of what was excluded.

        Exclusions, in order: the period, then the ticker (per-ticker scope) or the pool
        (excluding tier-C and below-minimum units, and TSLA for the `pooled_ex_TSLA` variant),
        then rows missing a required metric. Each one is counted.
        """
        caveats: list[str] = []
        frame = self.period_frame(period)
        if len(frame) == 0:
            return Selection(frame, 0, 0, 0, [f"no rows in the {period} period"])

        if scope == "ticker":
            if not ticker:
                raise ValueError("scope='ticker' needs a ticker")
            frame = frame[frame["ticker"] == ticker]
            if self.tiers.get(ticker) == "C":
                caveats.append(
                    f"{ticker} is coverage tier C: 100% of its text is cross-posted, so any "
                    "per-ticker text measure describes the generic pool, not the company"
                )
        else:
            excluded = int(frame["excluded_from_tests"].sum())
            if excluded:
                caveats.append(
                    f"{excluded} of {len(frame)} ticker-sessions in the {period} period are "
                    "excluded from tests (coverage tier C or below the tweet minimum)"
                )
            frame = frame[~frame["excluded_from_tests"]]
            if scope == "pooled_ex_TSLA":
                removed = int((frame["ticker"] == "TSLA").sum())
                frame = frame[frame["ticker"] != "TSLA"]
                caveats.append(
                    f"TSLA removed ({removed} ticker-sessions): it is 46.3% of the corpus, so a "
                    "plain pooled statistic is nearly a TSLA statistic"
                )
            else:
                caveats.append(
                    "pooled across tickers: shown beside the TSLA-excluded variant, because TSLA "
                    "dominates the pool"
                )

        n_before = len(frame)
        for column in require:
            frame = frame[frame[column].notna()]
        n_dropped = n_before - len(frame)
        if n_dropped:
            caveats.append(f"{n_dropped} ticker-sessions dropped for a missing {', '.join(require)} value")

        return Selection(
            frame=frame,
            n_obs=int((frame["n_tweets"].sum()) if len(frame) else 0),
            n_units=int(len(frame)),
            n_dropped=n_dropped,
            caveats=caveats,
        )


class ResultStore:
    """The run's `ToolResult`s, in call order, addressable by `result_id`."""

    def __init__(self) -> None:
        self.results: dict[str, ToolResult] = {}
        self.order: list[str] = []

    def add(self, result: ToolResult) -> ToolResult:
        """Store a result (idempotent, so a re-run of the same call does not duplicate)."""
        if result.result_id not in self.results:
            self.order.append(result.result_id)
        self.results[result.result_id] = result
        return result

    def get(self, result_id: str) -> ToolResult | None:
        """Return one result by id, or None when the id is unknown to this run."""
        return self.results.get(result_id)

    def all(self) -> list[ToolResult]:
        """Return every result in call order."""
        return [self.results[result_id] for result_id in self.order]

    def to_payload(self) -> list[dict[str, Any]]:
        """Return the store as JSON-safe dicts, ready to be written to `results/`."""
        return [result.model_dump() for result in self.all()]


# A tool function receives the context and validated parameters and returns a partial result.
ToolFunction = Callable[[ToolContext, Any], dict[str, Any]]


@dataclass
class Tool:
    """One registered analysis: its parameters, its purpose and the function to run."""

    name: str
    params_model: type[ToolParams]
    description: str
    function: ToolFunction
    chart_kind: str | None = None

    def json_schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function schema, which is also the agent-facing contract."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_model.model_json_schema(),
            },
        }


class ToolRegistry:
    """The closed set of analyses an agent (or the playbook) may call."""

    def __init__(self, context: ToolContext, store: ResultStore | None = None) -> None:
        self.context = context
        self.store = store or ResultStore()
        self.tools: dict[str, Tool] = {}
        self.errors: list[dict[str, Any]] = []

    # --- registration -------------------------------------------------------

    def add(
        self,
        name: str,
        params_model: type[ToolParams],
        description: str,
        chart_kind: str | None = None,
    ) -> Callable[[ToolFunction], ToolFunction]:
        """Decorator registering a tool function under `name`."""

        def decorate(function: ToolFunction) -> ToolFunction:
            if name in self.tools:
                raise ValueError(f"tool '{name}' is already registered")
            self.tools[name] = Tool(name, params_model, description, function, chart_kind)
            return function

        return decorate

    # --- calling ------------------------------------------------------------

    def names(self) -> list[str]:
        """Registered tool names, sorted for a stable prompt."""
        return sorted(self.tools)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        """Function-calling schemas for the tool set, optionally restricted to some names."""
        names = only or self.names()
        return [self.tools[name].json_schema() for name in names if name in self.tools]

    def result_id(self, name: str, params: dict[str, Any], period: str) -> str:
        """Return sha256(tool, canonical params, period, data_version)[:12].

        `period` is inside the hash even though the spec lists it outside: the same call on the
        holdout is a different computation, and giving it the same id would let a holdout check
        silently overwrite its own discovery result in the store.
        """
        payload = json.dumps(
            {"tool": name, "params": params, "period": period, "data_version": self.context.data_version},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def call(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        period: str | None = None,
        parent_result_ids: list[str] | None = None,
    ) -> ToolResult:
        """Validate, run and store one tool call.

        The `period` argument is authoritative for every tool that declares a period parameter: it
        overwrites `params['period']`. That matters because `holdout_check` re-runs a result on the
        other period, and if the parameter silently won the call would compute the discovery period
        again while labelling the result "holdout" — a bug that would make every replication check a
        false pass. When `period` is omitted it is read from the parameters.

        Raises `ToolError` on a bad name or bad parameters, so the caller (playbook or agent loop)
        can report the problem and try something else; a tool that raises internally is recorded as
        a failed call rather than killing the run.
        """
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool '{name}'. Registered: {self.names()}")
        supplied = dict(params or {})
        has_period = "period" in tool.params_model.model_fields
        if period is None:
            period = str(supplied.get("period", "discovery")) if has_period else "discovery"
        elif has_period:
            supplied["period"] = period
        try:
            validated = tool.params_model.model_validate(supplied)
        except ValidationError as error:
            raise ToolError(f"invalid parameters for '{name}': {error.errors()[:3]}") from error

        payload = validated.model_dump()
        result_id = self.result_id(name, payload, period)
        existing = self.store.get(result_id)
        if existing is not None:
            return existing

        try:
            produced = tool.function(self.context, validated)
        except ToolError:
            raise
        except Exception as error:  # a broken tool must not take the run down with it
            self.errors.append({"tool": name, "params": payload, "error": f"{type(error).__name__}: {error}"})
            raise ToolError(f"tool '{name}' failed: {type(error).__name__}: {error}") from error

        chart = produced.get("chart")
        caveats = list(produced.get("caveats", []))
        caveats.append(f"period={period}; n_units={produced.get('n_units', 0)} independent ticker-sessions")
        caveats.append(f"data_version={self.context.data_version}")
        result = ToolResult(
            result_id=result_id,
            tool=name,
            params=payload,
            period=period,
            n_obs=int(produced.get("n_obs", 0)),
            n_units=int(produced.get("n_units", 0)),
            n_dropped=int(produced.get("n_dropped", 0)),
            stats=produced.get("stats", {}),
            table=produced.get("table"),
            chart=chart,
            caveats=caveats,
            data_version=self.context.data_version,
            parent_result_ids=parent_result_ids or [],
        )
        if chart is not None:
            chart.source_result_id = result_id
            # The registry owns the caption, not the tool. A caption that does not name the period,
            # the sample size and the result it came from is a picture nobody can trace, and the way
            # to guarantee that is to append the identifiers here rather than trust every tool to
            # remember them.
            base = chart.caption.strip() or f"{chart.title} — period={period}, n_units={result.n_units}"
            if f"result {result_id}" not in base:
                base = f"{base}; result {result_id}"
            if "data_version" not in base:
                base = f"{base}; data_version={self.context.data_version}"
            chart.caption = base
        return self.store.add(result)


class ToolError(RuntimeError):
    """A caller error: unknown tool or invalid parameters. Recoverable by the caller."""
