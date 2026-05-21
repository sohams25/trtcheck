"""Click-based command line interface for trtcheck."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from trtcheck import __version__
from trtcheck.analyzer import Analyzer, AnalyzerConfig
from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter
from trtcheck.types import AnalysisReport, Severity

_FORMATS = ["console", "json", "html"]
_SEVERITIES = ["critical", "warning", "info"]
_KNOWN_TARGETS = ["8.0", "8.6", "10.0", "10.3"]


def _filter_issues(report: AnalysisReport, minimum: str) -> AnalysisReport:
    """Return a copy of `report` keeping only issues at or above the threshold.

    Threshold semantics:
      --severity critical -> only critical
      --severity warning  -> critical + warning
      --severity info     -> everything (default, no filter)
    """
    if minimum == "info":
        return report
    threshold = Severity(minimum)
    cutoff = Severity.rank(threshold)
    kept = [i for i in report.issues if Severity.rank(i.severity) <= cutoff]
    # Don't mutate the original -- make a shallow copy of the report.
    filtered = AnalysisReport(
        filename=report.filename,
        onnx_ir_version=report.onnx_ir_version,
        opset_version=report.opset_version,
        producer=report.producer,
        total_nodes=report.total_nodes,
        issues=kept,
        estimated_fusions=list(report.estimated_fusions),
        estimated_precision=dict(report.estimated_precision),
    )
    return filtered


def _render(report: AnalysisReport, fmt: str) -> str:
    if fmt == "json":
        return JSONReporter().render(report)
    if fmt == "html":
        return HTMLReporter().render(report)
    return ConsoleReporter().render(report)


def _emit(text: str, output_path: Path | None) -> None:
    if output_path is None:
        click.echo(text, nl=False)
        if not text.endswith("\n"):
            click.echo()
    else:
        output_path.write_text(text, encoding="utf-8")


@click.command(name="trtcheck", context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="trtcheck")
@click.argument(
    "models",
    nargs=-1,
    required=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
@click.option(
    "--target-trt",
    type=click.Choice(_KNOWN_TARGETS),
    default="10.3",
    show_default=True,
    help="TensorRT version to check compatibility against.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(_FORMATS),
    default="console",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the report to this path instead of stdout.",
)
@click.option(
    "--severity",
    type=click.Choice(_SEVERITIES),
    default="info",
    show_default=True,
    help="Minimum severity to include in the report.",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    help="Verbose mode currently equivalent to --severity info.",
)
@click.option(
    "--diff",
    is_flag=True,
    default=False,
    help="Compare two ONNX files. Requires exactly two model arguments.",
)
def main(
    models: tuple[Path, ...],
    target_trt: str,
    fmt: str,
    output: Path | None,
    severity: str,
    verbose: bool,
    diff: bool,
) -> None:
    """Run trtcheck against one or two ONNX models.

    \b
    Examples:
      trtcheck model.onnx
      trtcheck model.onnx --target-trt 8.6
      trtcheck model.onnx --format json --output report.json
      trtcheck before.onnx after.onnx --diff
    """
    if diff:
        if len(models) != 2:
            raise click.BadParameter("--diff requires exactly two model arguments")
        _run_diff(models, target_trt, fmt, output, severity)
        return

    if len(models) != 1:
        raise click.BadParameter("Pass one ONNX file, or use --diff with exactly two files.")

    path = models[0]
    if not path.exists():
        raise click.ClickException(f"ONNX file not found: {path}")

    analyzer = Analyzer(AnalyzerConfig(target_trt=target_trt))
    report = analyzer.analyze_path(path)
    # --verbose lowers the threshold to 'info' unless the user explicitly
    # asked for a stricter --severity. Explicit --severity wins.
    effective_severity = "info" if verbose else severity
    report = _filter_issues(report, effective_severity)

    text = _render(report, fmt)
    _emit(text, output)

    if not report.conversion_likely:
        sys.exit(1)


def _run_diff(
    models: tuple[Path, ...],
    target_trt: str,
    fmt: str,
    output: Path | None,
    severity: str,
) -> None:
    before, after = models
    if not before.exists():
        raise click.ClickException(f"ONNX file not found: {before}")
    if not after.exists():
        raise click.ClickException(f"ONNX file not found: {after}")

    analyzer = Analyzer(AnalyzerConfig(target_trt=target_trt))
    report_before = _filter_issues(analyzer.analyze_path(before), severity)
    report_after = _filter_issues(analyzer.analyze_path(after), severity)

    if fmt == "json":
        payload = json.dumps(
            {"before": report_before.to_dict(), "after": report_after.to_dict()},
            indent=2,
        )
        _emit(payload, output)
    else:
        if fmt == "html":
            from trtcheck.reporters.html import HTMLReporter, _CSS

            reporter = HTMLReporter()
            combined = (
                '<!doctype html><html><head><meta charset="utf-8">'
                f"<title>trtcheck diff</title><style>{_CSS}</style>"
                "</head><body>"
                + reporter.render_fragment(report_before)
                + "<hr>"
                + reporter.render_fragment(report_after)
                + "</body></html>"
            )
            _emit(combined, output)
        else:
            before_text = _render(report_before, fmt)
            after_text = _render(report_after, fmt)
            sep = "\n" + ("=" * 80) + "\n"
            _emit(
                f"BEFORE: {before}\n{before_text}\n{sep}AFTER: {after}\n{after_text}",
                output,
            )

    # The point of --diff is "did my fix work?". Exit based on the *after*
    # model only -- a passing 'after' must be a passing CI signal even when
    # the 'before' fixture still fails.
    if not report_after.conversion_likely:
        sys.exit(1)


if __name__ == "__main__":
    main()
