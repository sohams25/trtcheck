"""Reporters must neutralize hostile model-derived strings.

A node name / operator / message comes straight from an untrusted ONNX file.
The console reporter renders through Rich (which interprets ``[...]`` markup and
passes raw control characters through); the HTML reporter writes a document to
disk that may be served or piped. A crafted name must not inject styles or ANSI
escapes, must not leak raw control bytes (NUL/ESC/BEL) into any format, must not
smuggle a Trojan-Source bidi-override past the sanitizer, and long remediations
must not be truncated (RPT-002).

The contract is held identically across formats: console strips, HTML strips +
escapes, JSON escapes -- so a regression in any single reporter is caught here.
"""

from __future__ import annotations

import pytest

from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter
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


# --------------------------------------------------------------------------- #
# Cross-format contract: no raw control characters leak in ANY format          #
# --------------------------------------------------------------------------- #


def _hostile_report() -> AnalysisReport:
    """A report whose every model-derived field carries hostile bytes."""
    nul, esc, bel = "\x00", "\x1b", "\x07"
    return AnalysisReport(
        filename=f"m{nul}{esc}[2J.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer=f"prod{esc}[31mucer",
        total_nodes=1,
        issues=[
            Issue(
                severity=Severity.CRITICAL,
                category=CheckCategory.OPERATOR_SUPPORT,
                node_name=f"n{nul}{esc}[31mide",
                operator=f"Op{bel}",
                message=f"ms{bel}g{esc}[2J",
                remediation=f"fix{nul} it",
                docs_link=None,
            )
        ],
    )


@pytest.mark.parametrize(
    "reporter",
    [ConsoleReporter(color=False), HTMLReporter(), JSONReporter()],
    ids=["console", "html", "json"],
)
def test_no_raw_control_chars_in_any_format(reporter) -> None:
    out = reporter.render(_hostile_report())
    for ch in ("\x00", "\x1b", "\x07"):
        assert ch not in out, f"{reporter.name} leaked raw control char {ch!r}"


def test_html_report_has_no_nul_bytes() -> None:
    # A single NUL byte makes the .html document byte-invalid for many consumers.
    out = HTMLReporter().render(_hostile_report())
    assert "\x00" not in out
    # And the screen-clear ANSI sequence must not survive intact either.
    assert "\x1b[2J" not in out


def test_html_control_chars_are_stripped() -> None:
    report = _report_with("n\x1b[31m\x00ide", "ms\x07g", "fix")
    out = HTMLReporter().render(report)
    assert "\x00" not in out
    assert "\x1b" not in out
    assert "\x07" not in out


# --------------------------------------------------------------------------- #
# Trojan-Source: Unicode bidi-override characters must be stripped too          #
# --------------------------------------------------------------------------- #

_RLO = "‮"  # right-to-left override -- visually reverses following text


def test_bidi_override_chars_are_stripped_console() -> None:
    report = _report_with(f"Conv{_RLO}gpno.txt", "msg", "fix")
    out = ConsoleReporter(color=False).render(report)
    assert _RLO not in out


def test_bidi_override_chars_are_stripped_html() -> None:
    report = _report_with(f"Conv{_RLO}gpno.txt", "msg", "fix")
    out = HTMLReporter().render(report)
    assert _RLO not in out
