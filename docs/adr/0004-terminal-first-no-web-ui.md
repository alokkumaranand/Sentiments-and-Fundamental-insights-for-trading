# 0004. Terminal-first presentation; no web UI

**Status:** Accepted · **Date:** 2026-10-12

## Context

The original draft specified a four-page Streamlit app plus Plotly. The human's instruction was
explicit: avoid separate UIs, print all non-chart information as Markdown and a terminal
overview, and let the user pick charts from a numbered list of what is available.

Two practical facts supported that: neither Streamlit nor Plotly is installed here, and a second
UI is a second thing to keep in sync with the pipeline.

## Decision

Presentation is **terminal first**:

- a `rich` overview and summary printed at the end of a run;
- `report.md` as the durable artifact (with `report.html` as a convenience export);
- charts chosen from a numbered menu of the charts the run actually produced, rendered to PNG
  via matplotlib (`Agg`) and opened in the OS viewer, with an `ascii <n>` in-terminal fallback.

Tools do not build figures. They return a declarative, JSON-safe `ChartSpec`; a separate
renderer turns it into a PNG or ASCII. Tools stay pure and testable without a display.

Badges distinguish `[OBSERVED DATA]`, `[STATISTICAL ESTIMATE]` and `[AGENT INTERPRETATION]`
everywhere, and no output contains trading advice.

## Consequences

- One surface to build and maintain; nothing to keep in sync with a web layer.
- Every figure is traceable to a `result_id`, and captions carry it.
- Charts are static images, so there is no interactive cross-filtering. Accepted: the menu plus
  the report covers the demo's needs, and the app was out of scope for v0 anyway.
- A future interactive UI is not forbidden — `ChartSpec` is deliberately renderer-agnostic, so a
  web renderer can be added later without touching any analysis.
