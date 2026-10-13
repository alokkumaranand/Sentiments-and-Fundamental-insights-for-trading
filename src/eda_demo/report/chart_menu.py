"""The numbered chart menu: every figure traceable to a tool result.

Design decision embodied here: **a chart is only reachable through a `ToolResult`.** The menu lists
the results that actually produced a `ChartSpec` in this run, numbered, each with its title, kind,
`result_id` and `n_units`. So there is no way to look at a picture whose source call cannot be
named — which is the presentation half of design rule 1.

The menu is skipped when stdin is not a terminal, so a scripted or piped run finishes instead of
hanging on an input prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..tools.schema import ToolResult
from .render_ascii import render_ascii
from .render_png import open_in_viewer, render_png


class ChartMenu:
    """Interactive numbered menu over this run's chart-bearing results."""

    def __init__(self, results: list[ToolResult], run_dir: Path, settings: Settings, console: Any = None) -> None:
        self.results = [result for result in results if result.chart is not None]
        self.run_dir = run_dir
        self.settings = settings
        self.console = console or _PlainConsole()

    # --- listing ------------------------------------------------------------

    def listing(self) -> str:
        """Return the numbered chart list exactly as the user sees it."""
        if not self.results:
            return "Available charts: none (no tool in this run produced a chart)"
        lines = ["Available charts"]
        for index, result in enumerate(self.results, start=1):
            chart = result.chart
            assert chart is not None  # filtered in __init__
            lines.append(
                f"{index:>3}  {chart.title[:56]:<56} [{chart.kind:<9}] {result.result_id}  n_units={result.n_units}"
            )
        return "\n".join(lines)

    def prompt(self) -> str:
        """Return the input prompt line."""
        return 'Enter a number, "3,5", "all", "ascii 2", "md", "log", or "q":'

    # --- actions ------------------------------------------------------------

    def render(self, index: int, open_viewer: bool = False, ascii_only: bool = False) -> str:
        """Render chart `index` (1-based) and return a status line.

        `ascii_only` prints in-terminal; otherwise a PNG is written and optionally opened. The PNG
        always lands on disk, so a figure shown in a viewer can be found again by `result_id`.
        """
        if not 1 <= index <= len(self.results):
            return f"no chart {index}; this run has {len(self.results)}"
        result = self.results[index - 1]
        chart = result.chart
        assert chart is not None
        if ascii_only:
            return f"\n{render_ascii(chart)}\n"
        path = self.run_dir / "figures" / f"{result.result_id}.png"
        try:
            render_png(chart, path)
        except RuntimeError as error:
            return f"{error}; falling back to ascii\n{render_ascii(chart)}"
        status = f"wrote {path}"
        if open_viewer:
            status += f" — {open_in_viewer(path)}"
        return status

    def render_all(self) -> list[str]:
        """Render every chart as a PNG without opening viewers, and return the status lines.

        No viewer windows: `all` is for producing the figure set, and a dozen windows at once is
        not a feature. Rendering one by number opens it.
        """
        return [self.render(index, open_viewer=False) for index in range(1, len(self.results) + 1)]

    def export_markdown(self) -> str:
        """Return every chart as a fenced block, plus the run's report path."""
        report = self.run_dir / "report.md"
        blocks = [f"report: {report}" if report.exists() else "report: not written yet"]
        for index, result in enumerate(self.results, start=1):
            chart = result.chart
            assert chart is not None
            blocks.append(f"\n### {index}. {chart.title}\n{render_ascii(chart)}\ncaption: {chart.caption}")
        return "\n".join(blocks)

    def handle(self, command: str) -> bool:
        """Run one menu command. Returns False when the user asked to quit."""
        text = command.strip().lower()
        if text in ("q", "quit", "exit"):
            return False
        if text in ("", "list", "?"):
            self.console.print(self.listing())
            return True
        if text == "all":
            for status in self.render_all():
                self.console.print(status)
            return True
        if text == "md":
            self.console.print(self.export_markdown())
            return True
        if text == "log":
            self.console.print(
                "\n".join(
                    f"{result.result_id} {result.tool} {result.params} n_units={result.n_units} "
                    f"n_dropped={result.n_dropped}"
                    for result in self.results
                )
            )
            return True
        if text.startswith("ascii"):
            parts = text.split()
            numbers = _parse_numbers(parts[1]) if len(parts) > 1 else []
            for index in numbers or range(1, len(self.results) + 1):
                self.console.print(self.render(index, ascii_only=True))
            return True
        numbers = _parse_numbers(text)
        if not numbers:
            self.console.print(f"did not understand '{command}'. {self.prompt()}")
            return True
        for index in numbers:
            self.console.print(self.render(index, open_viewer=True))
        return True

    def run(self, input_fn: Callable[[str], str] | None = None) -> None:
        """Run the menu loop until the user quits or stdin is not a terminal."""
        self.console.print(self.listing())
        if not sys.stdin.isatty() and input_fn is None:
            self.console.print("(not a terminal: skipping the interactive menu; charts list above)")
            return
        reader = input_fn or input
        while True:
            try:
                command = reader(self.prompt())
            except (EOFError, KeyboardInterrupt):
                break
            if not self.handle(command):
                break


def _parse_numbers(text: str) -> list[int]:
    """Parse '3', '3,5' or '3 5' into a list of 1-based chart numbers."""
    numbers = []
    for part in text.replace(",", " ").split():
        if part.isdigit():
            numbers.append(int(part))
    return numbers


class _PlainConsole:
    """Fallback console so the menu works without rich (and in tests).

    It encodes to the stream's own encoding with `errors="replace"`, because a chart drawn with
    block characters on a cp1252 console must degrade to a question mark rather than raise inside
    the menu.
    """

    def print(self, *args: object, **_: object) -> None:
        text = " ".join(str(arg) for arg in args)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
