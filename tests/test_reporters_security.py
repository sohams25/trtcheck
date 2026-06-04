"""Console reporter must neutralize hostile model-derived strings.

A node name / operator / message comes straight from an untrusted ONNX file.
The console reporter renders through Rich, which interprets ``[...]`` markup and
passes raw control characters through. A crafted name must not inject styles or
ANSI escapes, and long remediations must not be truncated (RPT-002).
"""

from __future__ import annotations

from trtcheck.reporters.console import ConsoleReporter
from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity


def _report_with(node_name: str, message: str, remediation: str) -> AnalysisReport:
    return AnalysisReport(
        filename="m.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=1,
        issues=[
            Issue(
                severity=Severity.CRITICAL,
                category=CheckCategory.OPERATOR_SUPPORT,
                node_name=node_name,
                operator="Op",
                message=message,
                remediation=remediation,
                docs_link=None,
            )
        ],
    )


def test_markup_in_node_name_is_neutralized() -> None:
    report = _report_with("[red]evil[/red]", "msg", "fix")
    out = ConsoleReporter(color=False).render(report)  # must not raise
    # The literal brackets survive (rendered as text), proving markup was escaped
    # rather than interpreted as a style tag.
    assert "[red]evil[/red]" in out


def test_control_chars_in_metadata_are_stripped() -> None:
    report = _report_with("n\x1b[31mide", "ms\x07g", "fix")
    out = ConsoleReporter(color=False).render(report)
    assert "\x1b" not in out
    assert "\x07" not in out


def test_long_no_space_remediation_is_not_truncated() -> None:
    long_cmd = "python -m torch.onnx.export --opset=17 --no-such-flag=" + "x" * 80
    report = _report_with("n", "msg", long_cmd)
    out = ConsoleReporter(color=False).render(report)
    # The tail of the command (the x's) must survive -- folded, never ellipsised.
    assert "x" * 40 in out.replace("\n", "")
