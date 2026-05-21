"""Tests for the core type definitions used across the analyzer."""

import json

import pytest

from trtcheck.types import (
    AnalysisReport,
    CheckCategory,
    Issue,
    Severity,
)


def _make_issue(**overrides: object) -> Issue:
    defaults = dict(
        severity=Severity.CRITICAL,
        category=CheckCategory.OPERATOR_SUPPORT,
        node_name="n4",
        operator="SequenceEmpty",
        message="Not supported.",
        remediation="Replace List[Tensor].",
        docs_link=None,
    )
    defaults.update(overrides)
    return Issue(**defaults)  # type: ignore[arg-type]


class TestSeverity:
    def test_string_values_are_stable(self) -> None:
        # External callers (JSON consumers, CI configs) rely on these strings.
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_ordering_by_severity_rank(self) -> None:
        ranked = sorted(
            [Severity.INFO, Severity.CRITICAL, Severity.WARNING],
            key=Severity.rank,
        )
        assert ranked == [Severity.CRITICAL, Severity.WARNING, Severity.INFO]


class TestIssue:
    def test_required_fields(self) -> None:
        issue = _make_issue()
        assert issue.operator == "SequenceEmpty"
        assert issue.docs_link is None

    def test_to_dict_round_trip_through_json(self) -> None:
        issue = _make_issue(docs_link="https://example.invalid/x")
        as_dict = issue.to_dict()
        # Must survive JSON serialization unchanged.
        reloaded = json.loads(json.dumps(as_dict))
        assert reloaded["severity"] == "critical"
        assert reloaded["category"] == "operator_support"
        assert reloaded["docs_link"] == "https://example.invalid/x"


class TestAnalysisReport:
    def test_counts_are_recomputable_from_issues(self) -> None:
        issues = [
            _make_issue(severity=Severity.CRITICAL),
            _make_issue(severity=Severity.WARNING),
            _make_issue(severity=Severity.WARNING),
            _make_issue(severity=Severity.INFO),
        ]
        report = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=42,
            issues=issues,
        )
        assert report.critical_count == 1
        assert report.warning_count == 2
        assert report.info_count == 1

    def test_conversion_likely_false_when_critical_present(self) -> None:
        report = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=1,
            issues=[_make_issue(severity=Severity.CRITICAL)],
        )
        assert report.conversion_likely is False

    def test_conversion_likely_true_when_only_warnings_and_info(self) -> None:
        report = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=1,
            issues=[
                _make_issue(severity=Severity.WARNING),
                _make_issue(severity=Severity.INFO),
            ],
        )
        assert report.conversion_likely is True

    def test_estimated_fix_time_grows_with_critical_count(self) -> None:
        small = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=1,
            issues=[_make_issue(severity=Severity.CRITICAL)],
        )
        large = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=1,
            issues=[_make_issue(severity=Severity.CRITICAL) for _ in range(5)],
        )
        # Both must be non-empty; the heuristic should reflect more work for more issues.
        assert small.estimated_fix_time
        assert large.estimated_fix_time
        assert small.estimated_fix_time != large.estimated_fix_time

    def test_to_dict_is_json_serializable(self) -> None:
        report = AnalysisReport(
            filename="m.onnx",
            onnx_ir_version="8",
            opset_version=17,
            producer="pytorch",
            total_nodes=1,
            issues=[_make_issue()],
        )
        payload = json.dumps(report.to_dict())  # must not raise
        decoded = json.loads(payload)
        assert decoded["filename"] == "m.onnx"
        assert decoded["issues"][0]["severity"] == "critical"
        assert "conversion_likely" in decoded
