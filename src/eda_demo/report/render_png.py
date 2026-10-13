"""PNG rendering with matplotlib's Agg backend, plus the "open it" step.

Design decision embodied here: **Agg, not an interactive backend.** A figure is written to
`artifacts/runs/<run_id>/figures/` and only then handed to the OS viewer. That keeps rendering
testable with no display, makes the figure an artifact like any other, and means a chart on screen
is always traceable to a PNG on disk whose caption carries its `result_id`.

The renderer is a pure consumer of `ChartSpec`: it knows nothing about the panel, the tools or the
dataset, so a change of chart library touches this file only.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from ..tools.schema import ChartSpec

# Dual-axis charts put one series on a second y-axis; the label marks which one in the caption.
_RIGHT_AXIS = "right"


def render_png(spec: ChartSpec, path: Path, dpi: int = 110) -> Path:
    """Render one `ChartSpec` to `path`. Raises RuntimeError with a clear message if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # must precede pyplot: no display is required or used
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - matplotlib is pinned in requirements.txt
        raise RuntimeError(f"matplotlib is not available: {error}") from error

    figure, axis = plt.subplots(figsize=(9, 4.2))
    try:
        if spec.kind in ("line", "dual_line"):
            _draw_lines(axis, spec, figure)
        elif spec.kind in ("bar", "bar_error", "hist", "box"):
            _draw_bars(axis, spec)
        elif spec.kind == "scatter":
            _draw_scatter(axis, spec)
        elif spec.kind == "heatmap":
            _draw_heatmap(axis, spec, figure)
        else:  # pragma: no cover - ChartKind is a closed literal
            raise RuntimeError(f"no PNG renderer for chart kind '{spec.kind}'")

        axis.set_title(spec.title, fontsize=11)
        if spec.x.get("label"):
            axis.set_xlabel(str(spec.x["label"]))
        figure.text(0.01, 0.01, spec.caption[:180], fontsize=6.5, color="#555555")
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return path


def _draw_lines(axis: Any, spec: ChartSpec, figure: Any) -> None:
    """Draw one or more series as lines; a series marked axis='right' gets a twin axis."""
    x_values = list(range(len(spec.x.get("values", []))))
    labels = [str(label) for label in spec.x.get("values", [])]
    twin = None
    for series in spec.series:
        values = [float(v) if isinstance(v, (int, float)) else float("nan") for v in series.get("values", [])]
        if series.get("axis") == _RIGHT_AXIS:
            twin = twin or axis.twinx()
            twin.plot(x_values[: len(values)], values, linewidth=1.0, alpha=0.8,
                      label=str(series.get("label", "series")))
            twin.set_ylabel(str(series.get("label", "")))
        else:
            axis.plot(x_values[: len(values)], values, linewidth=1.1, label=str(series.get("label", "series")))
    for annotation in spec.annotations or []:
        if annotation.get("x") in labels:
            axis.axvline(labels.index(str(annotation["x"])), color="#cc3333", alpha=0.4, linewidth=0.8)
    step = max(1, len(labels) // 10)
    axis.set_xticks(x_values[::step])
    axis.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
    handles, names = axis.get_legend_handles_labels()
    if twin is not None:
        extra_handles, extra_names = twin.get_legend_handles_labels()
        handles, names = handles + extra_handles, names + extra_names
    if names:
        axis.legend(handles, names, fontsize=7, loc="upper left")


def _draw_bars(axis: Any, spec: ChartSpec) -> None:
    """Draw a bar chart; `bar_error` uses the second series as the error half-width."""
    labels = [str(label) for label in spec.x.get("values", [])]
    positions = list(range(len(labels)))
    first = spec.series[0] if spec.series else {"values": []}
    values = [float(v) if isinstance(v, (int, float)) else float("nan") for v in first.get("values", [])]
    errors = None
    if spec.kind == "bar_error" and len(spec.series) > 1:
        errors = [float(v) if isinstance(v, (int, float)) else 0.0 for v in spec.series[1].get("values", [])]
    axis.bar(positions[: len(values)], values, width=0.6, color="#3c78b4",
             yerr=errors if errors else None, capsize=3)
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_xticks(positions[: len(labels)])
    axis.set_xticklabels(labels, fontsize=7, rotation=45 if max((len(l) for l in labels), default=0) > 4 else 0,
                         ha="right" if max((len(l) for l in labels), default=0) > 4 else "center")
    axis.set_ylabel(str(first.get("label", "")))


def _draw_scatter(axis: Any, spec: ChartSpec) -> None:
    """Draw a scatter of the x values against the first series."""
    x_values = [float(v) if isinstance(v, (int, float)) else float("nan") for v in spec.x.get("values", [])]
    series = spec.series[0] if spec.series else {"values": [], "label": ""}
    y_values = [float(v) if isinstance(v, (int, float)) else float("nan") for v in series.get("values", [])]
    axis.scatter(x_values[: len(y_values)], y_values, s=8, alpha=0.5, color="#3c78b4")
    axis.set_ylabel(str(series.get("label", "y")))


def _draw_heatmap(axis: Any, spec: ChartSpec, figure: Any) -> None:
    """Draw a small correlation heatmap with a colour bar."""
    matrix = [
        [float(v) if isinstance(v, (int, float)) else float("nan") for v in series.get("values", [])]
        for series in spec.series
    ]
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(spec.x.get("values", []))))
    axis.set_xticklabels([str(label) for label in spec.x.get("values", [])], rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(spec.series)))
    axis.set_yticklabels([str(series.get("label", "")) for series in spec.series], fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.03)


def figure_to_data_uri(path: Path) -> str:
    """Return a PNG as a base64 data URI, so `report.html` stays self-contained."""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def open_in_viewer(path: Path) -> str:
    """Open a rendered figure in the OS default viewer and return a short status string.

    Windows-only in this environment; other platforms report that they cannot open a window rather
    than pretending they did. The file is already on disk either way, which is the part that
    matters for traceability.
    """
    import os
    import platform
    import subprocess
    import sys

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return f"opened {path.name}"
    except Exception as error:
        return f"rendered {path.name} but could not open a viewer ({type(error).__name__}); file is on disk"
