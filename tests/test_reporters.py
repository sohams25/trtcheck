"""Tests for the three reporters: JSON, console, HTML.

Reporters are pure formatters -- they accept an AnalysisReport and return
a string. We don't snapshot the full output (too brittle); we check that
key facts surface for both passing and failing reports.
"""

import json

import pytest

from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter
from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity


def _passing_report() -> AnalysisReport:
    r = AnalysisReport(
        filename="ok.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="pytorch",
        total_nodes=10,
        issues=[],
    )
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


def test_html_reporter_drops_non_http_docs_link() -> None:
    """A poisoned operator matrix could carry a javascript: URL in docs_link.
    The HTML reporter must refuse to render anything but http(s) hrefs.
    """
    r = AnalysisReport(
        filename="x.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=1,
        issues=[
            Issue(
                severity=Severity.CRITICAL,
                category=CheckCategory.OPERATOR_SUPPORT,
                node_name="n",
                operator="Op",
                message="msg",
                remediation="fix",
                docs_link="javascript:alert(1)",
            )
        ],
    )
    out = HTMLReporter().render(r)
    assert "javascript:" not in out
    assert "alert(1)" not in out


def test_html_suppresses_data_uri_docs_link() -> None:
    """A data: URI is as dangerous as javascript: -- the http(s)-only allow-list
    must drop it too. Guards against a future refactor to a 'javascript:'
    deny-list that would silently reopen the data: hole.
    """
    r = AnalysisReport(
        filename="x.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=1,
        issues=[
            Issue(
                severity=Severity.CRITICAL,
                category=CheckCategory.OPERATOR_SUPPORT,
                node_name="n",
                operator="Op",
                message="msg",
                remediation="fix",
                docs_link="data:text/html,<script>alert(1)</script>",
            )
        ],
    )
    out = HTMLReporter().render(r)
    assert 'href="data:' not in out
    assert "<script>alert(1)</script>" not in out
    assert ">link</a>" not in out


def test_all_reporters_agree_on_issue_set() -> None:
    """The three formats must agree on the issue set -- a silent divergence
    between the machine verdict (JSON, used for CI gating) and the human-visible
    console/HTML report would be a real correctness risk.
    """
    severities = [
        Severity.CRITICAL,
        Severity.CRITICAL,
        Severity.WARNING,
        Severity.WARNING,
        Severity.INFO,
        Severity.INFO,
    ]
    categories = [
        CheckCategory.OPERATOR_SUPPORT,
        CheckCategory.PRECISION,
        CheckCategory.DYNAMIC_SHAPES,
        CheckCategory.CONTROL_FLOW,
        CheckCategory.GRAPH_STRUCTURE,
        CheckCategory.OPERATOR_SUPPORT,
    ]
    issues = [
        Issue(
            severity=sev,
            category=cat,
            node_name=f"node_{idx}",
            operator=f"OperatorKind{idx}",
            message=f"message {idx}",
            remediation=f"remediation {idx}",
            docs_link=None,
        )
        for idx, (sev, cat) in enumerate(zip(severities, categories))
    ]
    report = AnalysisReport(
        filename="agree.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=6,
        issues=issues,
    )

    json_payload = json.loads(JSONReporter().render(report))
    json_ops = sorted(i["operator"] for i in json_payload["issues"])
    assert json_payload["critical_count"] == 2
    assert json_payload["warning_count"] == 2
    assert json_payload["info_count"] == 2

    html_out = HTMLReporter().render(report)
    console_out = ConsoleReporter(color=False).render(report)

    # Every operator name surfaces in every format.
    expected_ops = sorted(i.operator for i in issues)
    assert json_ops == expected_ops
    for op in expected_ops:
        assert op in html_out, f"HTML dropped operator {op}"
        assert op in console_out, f"console dropped operator {op}"

    # Per-severity counts agree between JSON and HTML (HTML renders one sev badge
    # per issue row).
    for sev, count in (("critical", 2), ("warning", 2), ("info", 2)):
        assert html_out.count(f"sev {sev}") == count


def test_html_render_fragment_excludes_document_wrappers() -> None:
    """render_fragment must return just the inner content for safe splicing."""
    r = AnalysisReport(
        filename="x.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=1,
        issues=[],
    )
    fragment = HTMLReporter().render_fragment(r)
    assert "<!doctype" not in fragment.lower()
    assert "<html" not in fragment.lower()
    assert "<head" not in fragment.lower()
    assert "<body" not in fragment.lower()
    # But it does contain the report container
    assert 'class="container"' in fragment


def test_html_render_still_includes_fragment_content() -> None:
    """render() must include everything render_fragment() includes."""
    r = AnalysisReport(
        filename="x.onnx",
        onnx_ir_version="8",
        opset_version=17,
        producer="p",
        total_nodes=1,
        issues=[],
    )
    reporter = HTMLReporter()
    full = reporter.render(r)
    fragment = reporter.render_fragment(r)
    assert fragment in full


class TestHTMLDiff:
    def _build(self, name: str, with_issues: bool) -> AnalysisReport:
        r = AnalysisReport(
            filename=name,
            onnx_ir_version="8",
            opset_version=17,
            producer="p",
            total_nodes=3,
            issues=(
                [
                    Issue(
                        severity=Severity.CRITICAL,
                        category=CheckCategory.OPERATOR_SUPPORT,
                        node_name="n4",
                        operator="SequenceEmpty",
                        message="msg",
                        remediation="fix",
                        docs_link=None,
                    )
                ]
                if with_issues
                else []
            ),
        )
        return r

    def test_render_diff_returns_complete_document(self) -> None:
        before = self._build("before.onnx", with_issues=True)
        after = self._build("after.onnx", with_issues=False)
        out = HTMLReporter().render_diff(before, after)
        assert out.lstrip().lower().startswith("<!doctype html")
        assert "</html>" in out.lower()

    def test_render_diff_has_two_columns(self) -> None:
        before = self._build("before.onnx", with_issues=True)
        after = self._build("after.onnx", with_issues=False)
        out = HTMLReporter().render_diff(before, after)
        # The layout uses two columns -- look for a flex/grid class marker we'll
        # define in the impl, and for both filenames in the body.
        assert "before.onnx" in out
        assert "after.onnx" in out
        assert "diff-column" in out  # implementation-defined CSS class

    def test_render_diff_has_no_external_resources(self) -> None:
        out = HTMLReporter().render_diff(self._build("a.onnx", True), self._build("b.onnx", False))
        lowered = out.lower()
        assert "<link " not in lowered
        assert "<script src=" not in lowered

    def test_render_diff_distinguishes_passing_from_failing(self) -> None:
        before = self._build("before.onnx", with_issues=True)  # 1 critical -> fail
        after = self._build("after.onnx", with_issues=False)  # 0 critical -> pass
        out = HTMLReporter().render_diff(before, after)
        # Each side surfaces its verdict.
        assert "conversion blocked" in out.lower()
        assert "no known blocker" in out.lower()
