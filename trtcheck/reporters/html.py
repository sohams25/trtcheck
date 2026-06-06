"""HTML reporter -- self-contained single file, no external assets."""

from __future__ import annotations

import html

from trtcheck._text import strip_unsafe
from trtcheck.types import AnalysisReport, Severity


def _safe(text: str) -> str:
    """Sanitize untrusted model-derived text for HTML output.

    ``html.escape`` neutralizes ``< > & " '`` but does NOT touch ASCII control
    characters (NUL/ESC/BEL) or Unicode bidi overrides -- a NUL makes the
    document byte-invalid and an embedded ANSI escape corrupts any terminal that
    later cats the .html. Strip those first (the same class the console reporter
    drops), then escape. Keeps the two formats' safety contract in lock-step.
    """
    return html.escape(strip_unsafe(text))


_CSS = """
:root {
  --bg: #0f1419;
  --surface: #1a1f29;
  --text: #e6e9ed;
  --muted: #8a93a3;
  --crit: #ff5c5c;
  --warn: #ffb454;
  --info: #6ab0ff;
  --ok: #4ec9b0;
  --border: #2c333f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  padding: 2rem;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 1rem 0; }
.verdict {
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid var(--border);
}
.verdict.pass { background: rgba(78, 201, 176, 0.08); border-color: var(--ok); }
.verdict.fail { background: rgba(255, 92, 92, 0.08); border-color: var(--crit); }
.verdict h2 { margin: 0 0 0.5rem 0; font-size: 1.25rem; }
.verdict.pass h2 { color: var(--ok); }
.verdict.fail h2 { color: var(--crit); }
.meta { color: var(--muted); font-size: 0.9rem; }
.meta span + span::before { content: " · "; }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
th { background: rgba(255,255,255,0.03); font-weight: 600; font-size: 0.85rem;
     text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
tr:last-child td { border-bottom: none; }
.sev {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.sev.critical { background: rgba(255, 92, 92, 0.15); color: var(--crit); }
.sev.warning  { background: rgba(255, 180, 84, 0.15); color: var(--warn); }
.sev.info     { background: rgba(106, 176, 255, 0.15); color: var(--info); }
code { font-family: ui-monospace, "JetBrains Mono", monospace; font-size: 0.9em; }
.docs a { color: var(--info); text-decoration: none; }
.docs a:hover { text-decoration: underline; }
.footer { color: var(--muted); font-size: 0.8rem; margin-top: 2rem; text-align: center; }
.diff-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  max-width: 100%;
}
.diff-column { min-width: 0; }
.diff-column h2.col-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0 0 0.75rem 0;
}
@media (max-width: 900px) {
  .diff-grid { grid-template-columns: 1fr; }
}
"""


class HTMLReporter:
    name = "html"

    def render(self, report: AnalysisReport) -> str:
        """Render a complete, self-contained HTML document."""
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en"><head>',
                '<meta charset="utf-8">',
                f"<title>trtcheck report -- {_safe(report.filename)}</title>",
                f"<style>{_CSS}</style>",
                "</head><body>",
                self.render_fragment(report),
                "</body></html>",
            ]
        )

    def render_fragment(self, report: AnalysisReport) -> str:
        """Render only the report body (without <!doctype>, <html>, <head>, <body>).

        Use this to splice multiple reports into a single document, e.g. for
        --diff mode. The CSS is shared at the document level; render() injects
        it once.
        """
        verdict_class = "pass" if report.conversion_likely else "fail"
        verdict_title = "Likely to convert" if report.conversion_likely else "Conversion will fail"
        parts: list[str] = []
        parts.append('<div class="container">')
        parts.append("<h1>trtcheck report</h1>")
        parts.append(f'<div class="verdict {verdict_class}">')
        parts.append(f"<h2>{verdict_title}</h2>")
        parts.append('<div class="meta">')
        parts.append(f"<span><code>{_safe(report.filename)}</code></span>")
        parts.append(f"<span>opset {report.opset_version}</span>")
        parts.append(f"<span>{report.total_nodes} nodes</span>")
        parts.append(f"<span>{report.critical_count} critical</span>")
        parts.append(f"<span>{report.warning_count} warning</span>")
        parts.append(f"<span>{report.info_count} info</span>")
        if report.estimated_fix_time:
            parts.append(f"<span>fix time: {html.escape(report.estimated_fix_time)}</span>")
        parts.append("</div></div>")

        if report.issues:
            parts.append("<table>")
            parts.append(
                "<thead><tr>"
                "<th>Severity</th><th>Node</th><th>Operator</th>"
                "<th>Issue</th><th>Fix</th><th>Docs</th>"
                "</tr></thead><tbody>"
            )
            for issue in report.issues:
                sev = issue.severity.value
                # html.escape does not make a URL safe in an href attribute.
                # Restrict to http(s) so a poisoned operator matrix can't
                # smuggle in a javascript: or data: URI.
                if issue.docs_link and issue.docs_link.startswith(("http://", "https://")):
                    docs_cell = (
                        f'<a href="{_safe(issue.docs_link)}" '
                        f'target="_blank" rel="noopener">link</a>'
                    )
                else:
                    docs_cell = ""
                parts.append(
                    "<tr>"
                    f'<td><span class="sev {sev}">{sev.upper()}</span></td>'
                    f"<td><code>{_safe(issue.node_name)}</code></td>"
                    f"<td><code>{_safe(issue.operator)}</code></td>"
                    f"<td>{_safe(issue.message)}</td>"
                    f"<td>{_safe(issue.remediation)}</td>"
                    f'<td class="docs">{docs_cell}</td>'
                    "</tr>"
                )
            parts.append("</tbody></table>")
        else:
            parts.append("<p><strong>No issues detected.</strong></p>")

        parts.append('<div class="footer">Generated by trtcheck.</div>')
        parts.append("</div>")
        return "\n".join(parts)

    def render_diff(self, before: AnalysisReport, after: AnalysisReport) -> str:
        """Render two reports side by side in one self-contained document.

        Each column is a render_fragment() of its report. Filenames sit at
        the top of each column so the reader knows which side is which.
        """
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en"><head>',
                '<meta charset="utf-8">',
                "<title>trtcheck diff -- "
                + _safe(before.filename)
                + " vs "
                + _safe(after.filename)
                + "</title>",
                f"<style>{_CSS}</style>",
                "</head><body>",
                '<div class="diff-grid">',
                '<div class="diff-column">',
                f'<h2 class="col-title">before: {_safe(before.filename)}</h2>',
                self.render_fragment(before),
                "</div>",
                '<div class="diff-column">',
                f'<h2 class="col-title">after: {_safe(after.filename)}</h2>',
                self.render_fragment(after),
                "</div>",
                "</div>",
                "</body></html>",
            ]
        )
