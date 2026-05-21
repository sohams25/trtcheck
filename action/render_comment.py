"""Render the aggregated trtcheck JSON into a sticky PR comment.

Input: path to an aggregate JSON document of the shape
    {"files": [{"path": "<repo-relative path>", "report": <AnalysisReport.to_dict()>}]}
Output: markdown to stdout, with an invisible HTML marker as the first line
so the poster can find and update this comment on subsequent runs.

All ONNX-derived strings (paths, node names, operator names, messages) are
escaped before being inserted into the markdown. The strings ultimately
come from PR-controlled bytes, so we must defang them.
"""

from __future__ import annotations

import json
import sys
from typing import Any

MARKER = "<!-- trtcheck-sticky:v1 -->"


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _escape_md(text: str | None) -> str:
    """Defang `text` for use in a markdown table cell or inline context."""
    if text is None:
        return ""
    cleaned = "".join(ch if ch >= " " else " " for ch in str(text))
    # Backtick: would break out of code spans.
    cleaned = cleaned.replace("`", "ʼ")
    cleaned = (
        cleaned.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    return cleaned


def _escape_code(text: str | None) -> str:
    """Defang content meant to live inside backticks or <code>...</code> HTML.

    Strips backticks (would close a code span), escapes < and > so the
    string is safe inside an HTML element, and removes control chars.
    """
    if text is None:
        return ""
    cleaned = "".join(ch if ch >= " " else " " for ch in str(text))
    cleaned = cleaned.replace("`", "ʼ")
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")
    return cleaned


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

        notable_op = ""
        notable_msg = ""
        for issue in report.get("issues", []):
            if issue.get("severity") == "critical":
                notable_op = issue.get("operator", "?")
                notable_msg = issue.get("message", "")
                break
        if not notable_op and report.get("issues"):
            first = report["issues"][0]
            notable_op = first.get("operator", "?")
            notable_msg = first.get("message", "")
        if notable_op:
            notable = f"`{_escape_code(notable_op)}`: {_escape_md(notable_msg)}"
        else:
            notable = ""

        lines.append(
            f"| `{_escape_code(path)}` | {status} | {crit} | {warn} | "
            f"{_truncate(notable, 80)} |"
        )

    lines.append("")
    # Per-file details
    for entry in files:
        path = entry.get("path", "?")
        report = entry.get("report", {})
        if not report.get("issues"):
            continue
        lines.append(
            "<details><summary>Details for " f"<code>{_escape_code(path)}</code></summary>"
        )
        lines.append("")
        lines.append("| Severity | Node | Operator | Issue | Fix |")
        lines.append("| --- | --- | --- | --- | --- |")
        for issue in report["issues"]:
            sev = issue.get("severity", "?").upper()
            node = _escape_code(issue.get("node_name", ""))
            op = _escape_code(issue.get("operator", ""))
            msg = _truncate(_escape_md(issue.get("message", "")), 120)
            rem = _truncate(_escape_md(issue.get("remediation", "")), 120)
            lines.append(f"| {sev} | `{node}` | `{op}` | {msg} | {rem} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


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
