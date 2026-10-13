"""Presentation layer: Markdown/HTML reports, the numbered chart menu and the two renderers.

Terminal first, one window (SPEC.md 4.12): the CLI prints an overview and a numbered chart menu,
the durable artifacts are `report.md` and `findings.json`, and `report.html` is a convenience
export with the figures embedded.
"""

from __future__ import annotations

from .build_report import ReportContext, render_markdown, write_reports
from .chart_menu import ChartMenu

__all__ = ["ChartMenu", "ReportContext", "render_markdown", "write_reports"]
