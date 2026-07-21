"""Console reporter using `rich` for colored tables and panels.

Sorting: issues arrive pre-sorted (critical first). We render them in
arrival order so the worst news is at the top of the table.

The `color` flag exists for tests -- with color=False, rich renders a
plain-text version that's easy to assert against.
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from trtcheck._text import strip_unsafe
from trtcheck.types import AnalysisReport, Severity, Verdict

_SEV_COLOR = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.INFO: "blue",
}

# Four-state verdict -> (headline, border color). The wording is deliberately
# conservative: LIKELY means "static analysis found no known blocker", never
# "guaranteed to convert".
_VERDICT_STYLE = {
    Verdict.BLOCKED: ("CONVERSION BLOCKED -- known critical incompatibilities", "red"),
    Verdict.UNVERIFIED: ("UNVERIFIED -- no known blocker, unresolved conditions remain", "yellow"),
    Verdict.LIKELY: ("LIKELY -- static analysis found no known blocker", "green"),
    Verdict.VERIFIED: ("VERIFIED -- TensorRT runtime build succeeded", "green"),
}


def _sanitize(text: str) -> str:
    """Make model-derived text safe to print: drop control / bidi-override chars
    (see :mod:`trtcheck._text`) and neutralize Rich console markup so a crafted
    node name like ``[red]x[/red]`` renders literally instead of being
    interpreted."""
    return escape(strip_unsafe(text))


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
        headline, border = _VERDICT_STYLE[report.verdict]
        title = f"[bold {border}]{headline}[/bold {border}]"
        target = f"  target: TensorRT {report.target_trt}" if report.target_trt else ""
        body = (
            f"{title}\n"
            f"file: {_sanitize(report.filename)}{target}\n"
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
        table.add_column("Rule", overflow="fold")
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
                _sanitize(issue.rule_id),
                _sanitize(issue.node_name),
                _sanitize(issue.operator),
                _sanitize(issue.message),
                _sanitize(issue.remediation),
            )
        return table

    def _summary(self, report: AnalysisReport) -> str:
        if report.verdict is Verdict.BLOCKED:
            return (
                f"\nEstimated fix time: {report.estimated_fix_time}.\n"
                "Address critical issues first; warnings can often wait."
            )
        if report.verdict is Verdict.UNVERIFIED:
            unresolved = sum(1 for i in report.issues if i.verify_required)
            return (
                f"\nEstimated fix time: {report.estimated_fix_time}.\n"
                f"{unresolved} finding(s) need runtime verification "
                "(trtexec) or manual review before this model can be called safe."
            )
        return f"\nEstimated fix time: {report.estimated_fix_time}"
