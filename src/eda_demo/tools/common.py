"""Shared helpers for the tool catalog: metric lookup, chart building, pooled standardisation.

Design decision embodied here: **pooling standardises within ticker first.** Tickers differ by
orders of magnitude in tweet volume and by level in polarity, so a raw pooled rank correlation
would mostly measure "which ticker is this". Removing each ticker's own mean and scale before
pooling makes the pooled number a statement about ticker-days instead of about TSLA's share of
the corpus.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .registry import ToolContext, ToolError
from .schema import ChartSpec

# Columns that are not analyses: identifiers, flags and bookkeeping. Listing a metric that is
# one of these is a caller error, not a silent empty result.
NON_METRIC = {
    "ticker", "session_date", "session_idx", "period", "tier", "exclusion_reason",
    "min_tweets_per_day", "low_coverage", "excluded_from_tests", "meets_min_on_all_text",
    "spike", "spike_sustained",
}


def metric_column(context: ToolContext, name: str) -> str:
    """Return `name` if it is a usable numeric panel column, else raise with the alternatives.

    The error lists the real columns, because the most common failure is a plausible-but-wrong
    metric name and the caller can only recover if it is told what exists.
    """
    if name in context.panel.columns and name not in NON_METRIC:
        return name
    available = sorted(
        column for column in context.panel.columns
        if column not in NON_METRIC and pd.api.types.is_numeric_dtype(context.panel[column])
    )
    raise ToolError(f"unknown metric '{name}'. Available metrics: {available}")


def values_of(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column as float, so an all-object column cannot leak into a statistic."""
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def within_ticker_z(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return the column standardised inside each ticker (NaN where the ticker has no variance).

    Mean and standard deviation per ticker, pooled afterwards: the result answers "was this
    ticker-day unusual for this ticker", which is the only comparable question across tickers
    with 46x different volumes.
    """
    values = values_of(frame, column)
    grouped = values.groupby(frame["ticker"], observed=True)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (values - mean) / std.replace(0.0, np.nan)


def numeric(value: Any, digits: int = 6) -> Any:
    """Round a float for output; pass NaN through as None so JSON stays valid.

    NaN is not valid JSON, and a `null` in `stats` is read as "not estimable", which is what a
    NaN always means here.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return None
    return round(number, digits)


def caption(title: str, period: str, n_units: int, extra: str = "") -> str:
    """Build the standard chart caption: what it shows, on what sample, in which period."""
    parts = [title, f"period={period}", f"n_units={n_units}"]
    if extra:
        parts.append(extra)
    return f"{'; '.join(parts)}"


def line_chart(
    title: str,
    x_label: str,
    x_values: list[Any],
    series: list[dict[str, Any]],
    period: str,
    n_units: int,
    annotations: list[dict[str, Any]] | None = None,
    extra_caption: str = "",
    kind: str = "line",
) -> ChartSpec:
    """Return a line `ChartSpec`; `kind="dual_line"` marks a chart whose series use two y-axes."""
    return ChartSpec(
        kind=kind,  # type: ignore[arg-type]
        title=title,
        x={"label": x_label, "values": list(x_values)},
        series=series,
        annotations=annotations or [],
        caption=caption(title, period, n_units, extra_caption),
    )


def bar_chart(
    title: str,
    x_label: str,
    x_values: list[Any],
    series: list[dict[str, Any]],
    period: str,
    n_units: int,
    kind: str = "bar",
    extra_caption: str = "",
) -> ChartSpec:
    """Return a bar (or bar_error) `ChartSpec`."""
    return ChartSpec(
        kind=kind,  # type: ignore[arg-type]
        title=title,
        x={"label": x_label, "values": list(x_values)},
        series=series,
        caption=caption(title, period, n_units, extra_caption),
    )


def histogram(values: pd.Series, bins: int = 20) -> tuple[list[float], list[float], list[int]]:
    """Return (centres, edges, counts) for a plain frequency histogram.

    Built here rather than with matplotlib so the spec is renderer-independent: both the PNG and
    the ASCII renderer draw from the same numbers.
    """
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return [], [], []
    counts, edges = np.histogram(clean, bins=min(bins, max(1, len(np.unique(clean)))))
    centres = ((edges[:-1] + edges[1:]) / 2).round(6)
    return [float(c) for c in centres], [float(e) for e in edges], [int(c) for c in counts]


def session_labels(frame: pd.DataFrame) -> list[str]:
    """Return ISO date labels for a frame's sessions, in order and without duplicates."""
    return [str(value)[:10] for value in pd.Series(frame["session_date"]).drop_duplicates().sort_values()]
