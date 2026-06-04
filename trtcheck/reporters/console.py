"""Console reporter using `rich` for colored tables and panels.

Sorting: issues arrive pre-sorted (critical first). We render them in
arrival order so the worst news is at the top of the table.

The `color` flag exists for tests -- with color=False, rich renders a
plain-text version that's easy to assert against.
"""

from __future__ import annotations

import io
import re

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from trtcheck.types import AnalysisReport, Severity

_SEV_COLOR = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "blue",
}

# Strip ASCII control characters (keep tab/newline) from model-derived text so a
# hostile model can't smuggle ANSI escapes / cursor moves into the terminal.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Make model-derived text safe to print: drop control chars and neutralize
    Rich console markup so a crafted node name like ``[red]x[/red]`` renders
    literally instead of being interpreted."""
    return escape(_CONTROL_CHARS.sub("", text))


class ConsoleReporter:
    name = "console"

    def __init__(self, color: bool = True) -> None:
        self._color = color

    def render(self, report: AnalysisReport) -> str:
        buf = io.StringIO()
        console = Console(
            file=buf,
            force_terminal=self._color,
            no_color=not self._color,
            width=120,
        )
        console.print(self._header(report))
        if report.issues:
            console.print(self._issues_table(report))
        else:
            console.print("[bold]No issues detected.[/bold]")
        console.print(self._summary(report))
        return buf.getvalue()

    def _header(self, report: AnalysisReport) -> Panel:
        if report.conversion_likely:
            title = "[bold green]LIKELY TO CONVERT[/bold green]"
            border = "green"
        else:
            title = "[bold red]CONVERSION WILL FAIL[/bold red]"
            border = "red"
        body = (
            f"{title}\n"
            f"file: {_sanitize(report.filename)}\n"
            f"opset: {report.opset_version}  producer: {_sanitize(report.producer)}  "
            f"nodes: {report.total_nodes}\n"
            f"{report.critical_count} critical  "
            f"{report.warning_count} warning  "
            f"{report.info_count} info"
        )
        return Panel(body, title="trtcheck report", border_style=border)

    def _issues_table(self, report: AnalysisReport) -> Table:
        table = Table(title="Detected issues", show_lines=True)
        table.add_column("Severity", style="bold")
        # overflow="fold" hard-wraps long unbroken tokens (export commands, URLs,
        # paths) instead of clipping them with an ellipsis -- the remediation a
        # user must apply is never silently truncated.
        table.add_column("Node", overflow="fold")
        table.add_column("Operator", overflow="fold")
        table.add_column("Issue", max_width=44, overflow="fold")
        table.add_column("Fix", max_width=44, overflow="fold")
        for issue in report.issues:
            color = _SEV_COLOR[issue.severity]
            table.add_row(
                f"[{color}]{issue.severity.value.upper()}[/{color}]",
                _sanitize(issue.node_name),
                _sanitize(issue.operator),
                _sanitize(issue.message),
                _sanitize(issue.remediation),
            )
        return table

    def _summary(self, report: AnalysisReport) -> str:
        if report.conversion_likely:
            return f"\nEstimated fix time: {report.estimated_fix_time}"
        return (
            f"\nEstimated fix time: {report.estimated_fix_time}.\n"
            "Address critical issues first; warnings can often wait."
        )
