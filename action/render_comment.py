"""Render the aggregated trtcheck JSON into a sticky PR comment.

Input: path to an aggregate JSON document of the shape
    {"files": [{"path": "<repo-relative path>", "report": <AnalysisReport.to_dict()>}]}
Output: markdown to stdout, with an invisible HTML marker as the first line
so the poster can find and update this comment on subsequent runs.
"""

from __future__ import annotations

import json
import sys
from typing import Any

MARKER = "<!-- trtcheck-sticky:v1 -->"


def render(aggregate: dict[str, Any]) -> str:
    files = aggregate.get("files", [])
    lines: list[str] = [MARKER, ""]
    lines.append("## trtcheck pre-flight report")
    lines.append("")

    if not files:
        lines.append("_No ONNX files were analyzed._")
        lines.append("")
        return "\n".join(lines)

    total_crit = sum(f["report"].get("critical_count", 0) for f in files)
    total_warn = sum(f["report"].get("warning_count", 0) for f in files)
    overall = "**Conversion will fail**" if total_crit else "**Likely to convert**"
    lines.append(
        f"{overall} - {len(files)} file(s) analyzed, "
        f"{total_crit} critical, {total_warn} warning."
    )
    lines.append("")

    lines.append("| File | Status | Critical | Warning | Notable issue |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for entry in files:
        path = entry.get("path", "?")
        report = entry.get("report", {})
        crit = report.get("critical_count", 0)
        warn = report.get("warning_count", 0)
        status = "❌ fail" if crit else "✅ pass"
        notable = ""
        for issue in report.get("issues", []):
            if issue.get("severity") == "critical":
                notable = f"`{issue.get('operator', '?')}`: {issue.get('message', '')}"
                break
        if not notable and report.get("issues"):
            first = report["issues"][0]
            notable = f"`{first.get('operator', '?')}`: {first.get('message', '')}"
        lines.append(f"| `{path}` | {status} | {crit} | {warn} | {_truncate(notable, 80)} |")

    lines.append("")
    # Per-file details
    for entry in files:
        path = entry.get("path", "?")
        report = entry.get("report", {})
        if not report.get("issues"):
            continue
        lines.append(f"<details><summary>Details for <code>{path}</code></summary>")
        lines.append("")
        lines.append("| Severity | Node | Operator | Issue | Fix |")
        lines.append("| --- | --- | --- | --- | --- |")
        for issue in report["issues"]:
            sev = issue.get("severity", "?").upper()
            node = issue.get("node_name", "")
            op = issue.get("operator", "")
            msg = _truncate(issue.get("message", ""), 120).replace("|", "\\|")
            rem = _truncate(issue.get("remediation", ""), 120).replace("|", "\\|")
            lines.append(f"| {sev} | `{node}` | `{op}` | {msg} | {rem} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: render_comment.py <aggregate.json>", file=sys.stderr)
        return 2
    aggregate = json.loads(open(argv[0]).read())
    sys.stdout.write(render(aggregate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
