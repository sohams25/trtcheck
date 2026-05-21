"""Tests for the three reporters: JSON, console, HTML.

Reporters are pure formatters -- they accept an AnalysisReport and return
a string. We don't snapshot the full output (too brittle); we check that
key facts surface for both passing and failing reports.
"""

import json

import pytest

from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity
from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter


def _passing_report() -> AnalysisReport:
    r = AnalysisReport(
        filename="ok.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="pytorch",
        total_nodes=10,
        issues=[],
    )
    r.derive_verdict()
    return r


def _failing_report() -> AnalysisReport:
    r = AnalysisReport(
        filename="bad.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="pytorch",
        total_nodes=10,
        issues=[
            Issue(
                severity=Severity.CRITICAL,
                category=CheckCategory.OPERATOR_SUPPORT,
                node_name="n4",
                operator="SequenceEmpty",
                message="Not supported in TensorRT 10.3.",
                remediation="Replace List[Tensor] with torch.stack().",
                docs_link="https://example.invalid/issue",
            ),
            Issue(
                severity=Severity.WARNING,
                category=CheckCategory.PRECISION,
                node_name="weights",
                operator="Initializer",
                message="INT64 initializer detected.",
                remediation="Cast to int32 before export.",
                docs_link=None,
            ),
        ],
    )
    r.derive_verdict()
    return r


class TestJSONReporter:
    def test_passing_report_is_valid_json(self) -> None:
        out = JSONReporter().render(_passing_report())
        payload = json.loads(out)
        assert payload["filename"] == "ok.onnx"
        assert payload["conversion_likely"] is True
        assert payload["issues"] == []

    def test_failing_report_includes_issues_array(self) -> None:
        out = JSONReporter().render(_failing_report())
        payload = json.loads(out)
        assert payload["conversion_likely"] is False
        assert payload["critical_count"] == 1
        assert len(payload["issues"]) == 2
        assert payload["issues"][0]["operator"] == "SequenceEmpty"

    def test_indentation_is_human_readable(self) -> None:
        out = JSONReporter().render(_passing_report())
        # Indented JSON has newlines; one-line JSON does not.
        assert "\n" in out


class TestConsoleReporter:
    def test_passing_report_mentions_likely_to_convert(self) -> None:
        out = ConsoleReporter(color=False).render(_passing_report())
        assert "LIKELY" in out.upper() or "PASS" in out.upper() or "convert" in out.lower()

    def test_failing_report_lists_critical_first(self) -> None:
        out = ConsoleReporter(color=False).render(_failing_report())
        crit_pos = out.find("SequenceEmpty")
        warn_pos = out.find("INT64")
        assert crit_pos != -1 and warn_pos != -1
        assert crit_pos < warn_pos, "critical issue must appear before warning"

    def test_failing_report_shows_remediation(self) -> None:
        out = ConsoleReporter(color=False).render(_failing_report())
        assert "torch.stack" in out


class TestHTMLReporter:
    def test_output_is_complete_html_document(self) -> None:
        out = HTMLReporter().render(_failing_report())
        # Self-contained means a doctype and a closing html tag, no external links.
        assert out.lstrip().lower().startswith("<!doctype html")
        assert "</html>" in out.lower()

    def test_output_has_no_external_resource_references(self) -> None:
        # Self-contained: no <link rel="stylesheet"> or <script src="...">
        out = HTMLReporter().render(_failing_report()).lower()
        assert "<link " not in out  # external stylesheet would use <link rel="stylesheet"...>
        assert "<script src=" not in out

    def test_failing_report_includes_remediation_text(self) -> None:
        out = HTMLReporter().render(_failing_report())
        assert "torch.stack" in out
        assert "SequenceEmpty" in out

    def test_passing_report_renders_without_issues_table(self) -> None:
        out = HTMLReporter().render(_passing_report())
        assert "ok.onnx" in out
        # Verdict text should be positive
        assert "likely" in out.lower() or "no issues" in out.lower()
