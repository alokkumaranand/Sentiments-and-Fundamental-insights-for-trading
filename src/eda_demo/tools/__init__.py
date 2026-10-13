"""The EDA tool library: typed, pure, deterministic analyses with uniform results.

`build_registry` is the single place the catalog is assembled, so "what can be called" is one
readable list rather than a discovery mechanism an agent could extend.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import Settings
from . import descriptive, relational, validation
from .registry import ResultStore, ToolContext, ToolRegistry
from .schema import ChartSpec, ToolResult

__all__ = ["build_registry", "ToolContext", "ResultStore", "ToolRegistry", "ChartSpec", "ToolResult"]


def build_registry(
    settings: Settings,
    panel: pd.DataFrame,
    tweets: pd.DataFrame,
    data_version: str,
    tiers: dict[str, str],
    calendar: list[pd.Timestamp] | None = None,
    store: ResultStore | None = None,
) -> ToolRegistry:
    """Assemble the full tool catalog over one panel.

    The registry is built per run, because its context holds that run's panel, tier map and data
    version: sharing one across runs would let a result from another dataset be cited as evidence.
    """
    context = ToolContext(
        panel=panel,
        tweets=tweets,
        settings=settings,
        data_version=data_version,
        tiers=tiers,
        calendar=calendar or [],
    )
    registry = ToolRegistry(context, store=store)
    descriptive.register(registry)
    relational.register(registry)
    validation.register(registry)
    return registry
