"""ASCII rendering: a quick in-terminal look at a `ChartSpec`, with no display required.

Design decision embodied here: the ASCII backend reads the **same `ChartSpec`** as the PNG one. It
is not a debug view of the data — it is a second renderer of the same declarative spec, which is
what proves the spec has no matplotlib concepts leaking into it.

Deliberately plain text rather than rich markup: the output has to survive being pasted into a
report, a log or a test assertion, and a broken terminal should never be the reason a chart is
unavailable when the PNG path fails.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from ..tools.schema import ChartSpec

# Eight-step ramps, low to high. A sparkline of one series needs no axis, which is what makes it
# useful in a terminal. The Unicode blocks look better; the ASCII ramp is used when the output
# stream cannot encode them (a cp1252 console on Windows raises UnicodeEncodeError otherwise, which
# is a rendering crash, not a data problem).
_UNICODE_BLOCKS = "▁▂▃▄▅▆▇█"
_ASCII_BLOCKS = "_.-~=+*#"
_DENSITY = " .:-=+*#%@"


def _blocks() -> str:
    """Return the sparkline ramp the active stdout can actually encode."""
    return _UNICODE_BLOCKS if _can_encode(_UNICODE_BLOCKS) else _ASCII_BLOCKS


def _bar_char() -> str:
    """Return the bar character the active stdout can encode."""
    return "█" if _can_encode("█") else "#"


def _can_encode(text: str) -> bool:
    """True when stdout's encoding can represent `text`."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _finite(values: list[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def sparkline(values: list[Any], width: int = 60) -> str:
    """Return a sparkline, downsampled to at most `width` characters."""
    blocks = _blocks()
    clean = _finite(values)
    if not clean:
        return "(no finite values)"
    if len(clean) > width:
        step = len(clean) / width
        clean = [clean[min(len(clean) - 1, int(i * step))] for i in range(width)]
    low, high = min(clean), max(clean)
    if high == low:
        return blocks[3] * len(clean)
    return "".join(blocks[min(len(blocks) - 1, int((value - low) / (high - low) * (len(blocks) - 1)))] for value in clean)


def horizontal_bars(labels: list[Any], values: list[Any], width: int = 40, indent: int = 2) -> str:
    """Return labelled horizontal bars scaled to the largest absolute value."""
    bar = _bar_char()
    clean = _finite(values)
    if not clean:
        return " " * indent + "(no finite values)"
    scale = max(abs(v) for v in clean) or 1.0
    lines = []
    label_width = max((len(str(label)) for label in labels), default=1)
    for label, value in zip(labels, values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            lines.append(f"{' ' * indent}{str(label):<{label_width}}  (not estimable)")
            continue
        length = int(round(abs(float(value)) / scale * width))
        drawn = bar * max(1, length) if value != 0 else ""
        lines.append(f"{' ' * indent}{str(label):<{label_width}}  {drawn} {float(value):+.4g}")
    return "\n".join(lines)


def scatter_grid(x: list[Any], y: list[Any], width: int = 56, height: int = 14) -> str:
    """Return a coarse character grid of a scatter, for a terminal-sized look at the shape."""
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if isinstance(a, (int, float)) and isinstance(b, (int, float))
        and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return "(not enough finite points to plot)"
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = min(ys), max(ys)
    if x_high == x_low:
        x_high = x_low + 1
    if y_high == y_low:
        y_high = y_low + 1
    grid = [[" "] * width for _ in range(height)]
    for x_value, y_value in pairs:
        column = min(width - 1, int((x_value - x_low) / (x_high - x_low) * (width - 1)))
        row = min(height - 1, int((1 - (y_value - y_low) / (y_high - y_low)) * (height - 1)))
        grid[row][column] = "*"
    header = f"  y {y_high:.4g}"
    footer = f"  y {y_low:.4g}   x {x_low:.4g} -> {x_high:.4g}   ({len(pairs)} points)"
    return "\n".join([header] + ["  " + "".join(row) for row in grid] + [footer])


def heatmap(matrix: list[list[Any]], row_labels: list[Any], column_labels: list[Any]) -> str:
    """Return a heatmap as a character grid with a numeric legend per cell row."""
    clean = [abs(float(v)) for row in matrix for v in row if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean:
        return "(no finite values)"
    peak = max(clean) or 1.0
    lines = []
    header = "      " + " ".join(f"{str(label)[:6]:>6}" for label in column_labels)
    lines.append(header)
    for label, row in zip(row_labels, matrix):
        cells = []
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                cells.append(f"{'--':>6}")
                continue
            level = min(len(_DENSITY) - 1, int(abs(float(value)) / peak * (len(_DENSITY) - 1)))
            cells.append(f"{_DENSITY[level]}{float(value):+.2f}"[:6].rjust(6))
        lines.append(f"{str(label)[:5]:>5} " + " ".join(cells))
    return "\n".join(lines)


def render_ascii(spec: ChartSpec, width: int = 60) -> str:
    """Render any `ChartSpec` as terminal text. Never raises for an unusual spec."""
    head = f"{spec.title}\n{'-' * min(len(spec.title), 72)}"
    kind = spec.kind
    try:
        if kind in ("line", "dual_line"):
            lines = [head]
            for series in spec.series:
                values = series.get("values", [])
                stats = _summary(values)
                lines.append(f"  {series.get('label', 'series')}: {sparkline(values, width)}  {stats}")
            lines.append(f"  x: {spec.x.get('label', 'x')} across {len(spec.x.get('values', []))} points")
            return "\n".join(lines)
        if kind in ("bar", "bar_error", "hist", "box"):
            lines = [head, horizontal_bars(spec.x.get("values", []), spec.series[0].get("values", []) if spec.series else [], width)]
            if kind == "bar_error" and len(spec.series) > 1:
                lines.append("  CI half-widths: " + sparkline(spec.series[1].get("values", []), width))
            return "\n".join(lines)
        if kind == "scatter":
            return "\n".join([head, scatter_grid(spec.x.get("values", []), (spec.series[0].get("values", []) if spec.series else []))])
        if kind == "heatmap":
            matrix = [series.get("values", []) for series in spec.series]
            return "\n".join([head, heatmap(matrix, [series.get("label", "") for series in spec.series], spec.x.get("values", []))])
    except Exception as error:  # a renderer must never be the reason a run fails
        return f"{head}\n  (ascii render failed: {type(error).__name__}: {error})"
    return f"{head}\n  (no ascii renderer for kind '{kind}')"


def _summary(values: list[Any]) -> str:
    """One-line min/mean/max for a series, so a sparkline has numbers beside it."""
    clean = _finite(values)
    if not clean:
        return "(no finite values)"
    return f"min {min(clean):.4g}  mean {sum(clean) / len(clean):.4g}  max {max(clean):.4g}"
