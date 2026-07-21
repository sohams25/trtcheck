"""Verdict model, stable rule ids, and honest-uncertainty findings."""

from __future__ import annotations

import json

import onnx
from onnx import TensorProto, helper

from trtcheck import remediation
from trtcheck.analyzer import Analyzer, AnalyzerConfig
from trtcheck.checkers.operator_support import (
    RULE_CONDITION,
    RULE_CONDITION_UNRESOLVED,
    RULE_CUSTOM_DOMAIN,
    RULE_PARTIAL,
    RULE_UNCLASSIFIED,
    RULE_UNSUPPORTED,
)
from trtcheck.reporters.json import JSONReporter
from trtcheck.types import REPORT_SCHEMA_VERSION, AnalysisReport, Confidence, Severity, Verdict

_ANALYZER = Analyzer(AnalyzerConfig(discover_entry_point_plugins=False))

# The complete public rule-id registry. Renaming or removing an id here is a
# BREAKING change for CI consumers filtering on rule_id -- this test is the
# tripwire. Additions are fine.
_DOCUMENTED_RULE_IDS = {
    "TRT-DTYPE-INT64-WEIGHTS",
    "TRT-DTYPE-INT64-INPUT",
    "TRT-DTYPE-UINT8-INPUT",
    "TRT-DTYPE-BF16",
    "TRT-DTYPE-FP64",
    "TRT-DTYPE-STRING",
    "TRT-SHAPE-PROFILE-MISSING",
    "TRT-GRAPH-NO-OUTPUT",
    "TRT-GRAPH-ISOLATED-NODE",
    "TRT-GRAPH-DUP-NODE-NAME",
    "TRT-GRAPH-LARGE-CONSTANT",
    "TRT-GRAPH-EXTERNAL-DATA",
    "TRT-GRAPH-INPUT-UNTYPED",
    "TRT-GRAPH-ALIASING",
    "TRT-CONTROL-LOOP-RUNTIME-TRIP",
    "TRT-CONTROL-LOOP-DYNAMIC-TRIP",
    "TRT-CONTROL-LOOP-NESTED",
    "TRT-CONTROL-IF-SHAPE-MISMATCH",
    "TRT-CONTROL-IF-UNVERIFIED",
    "TRT-CONTROL-SCAN",
    "TRT-OPSET-OLD",
    "TRT-OP-UNSUPPORTED",
    "TRT-OP-PARTIAL",
}
_CHECKER_OWNED_IDS = {
    RULE_UNSUPPORTED,
    RULE_PARTIAL,
    RULE_UNCLASSIFIED,
    RULE_CUSTOM_DOMAIN,
    RULE_CONDITION,
    RULE_CONDITION_UNRESOLVED,
}


def test_rule_id_registry_is_stable() -> None:
    assert remediation.rule_ids() == frozenset(_DOCUMENTED_RULE_IDS)
    # Checker-owned ids never collide with DB-owned ones (except the two the
    # operator checker shares with legacy DB entries by design).
    overlap = _CHECKER_OWNED_IDS & remediation.rule_ids()
    assert overlap == {RULE_UNSUPPORTED, RULE_PARTIAL}


def _model_with_node(node: helper.NodeProto, opsets=None) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    vi = onnx.ValueInfoProto()
    vi.name = node.output[0]
    graph = helper.make_graph([node], "m", [inp], [vi])
    model = helper.make_model(graph, opset_imports=opsets or [helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestVerdicts:
    def test_clean_model_is_likely(self, clean_model: onnx.ModelProto) -> None:
        report = _ANALYZER.analyze_model(clean_model)
        assert report.verdict is Verdict.LIKELY
        assert report.conversion_likely is True

    def test_critical_issue_means_blocked(self, uint8_input_model: onnx.ModelProto) -> None:
        report = _ANALYZER.analyze_model(uint8_input_model)
        assert report.verdict is Verdict.BLOCKED
        assert report.conversion_likely is False

    def test_unknown_default_domain_op_means_unverified(self) -> None:
        node = helper.make_node("TotallyNovelOp", ["input"], ["y"], name="n")
        report = _ANALYZER.analyze_model(_model_with_node(node))
        assert report.verdict is Verdict.UNVERIFIED
        finding = next(i for i in report.issues if i.rule_id == RULE_UNCLASSIFIED)
        assert finding.severity is Severity.INFO
        assert finding.verify_required is True
        assert finding.confidence is Confidence.LOW

    def test_custom_domain_op_means_unverified(self) -> None:
        node = helper.make_node("PluginOp", ["input"], ["y"], name="n", domain="com.acme")
        model = _model_with_node(
            node,
            opsets=[helper.make_opsetid("", 17), helper.make_opsetid("com.acme", 1)],
        )
        report = _ANALYZER.analyze_model(model)
        assert report.verdict is Verdict.UNVERIFIED
        finding = next(i for i in report.issues if i.rule_id == RULE_CUSTOM_DOMAIN)
        assert "com.acme" in finding.operator
        # Not a guaranteed blocker: honest uncertainty, not a critical.
        assert finding.severity is Severity.INFO

    def test_declared_plugin_domain_suppresses_finding(self) -> None:
        node = helper.make_node("PluginOp", ["input"], ["y"], name="n", domain="com.acme")
        model = _model_with_node(
            node,
            opsets=[helper.make_opsetid("", 17), helper.make_opsetid("com.acme", 1)],
        )
        analyzer = Analyzer(
            AnalyzerConfig(discover_entry_point_plugins=False, plugin_domains=["com.acme"])
        )
        report = analyzer.analyze_model(model)
        assert all(i.rule_id != RULE_CUSTOM_DOMAIN for i in report.issues)

    def test_unclassified_findings_aggregate_per_op_type(self) -> None:
        n1 = helper.make_node("NovelOp", ["input"], ["y1"], name="n1")
        n2 = helper.make_node("NovelOp", ["y1"], ["y2"], name="n2")
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        vi = onnx.ValueInfoProto()
        vi.name = "y2"
        graph = helper.make_graph([n1, n2], "m", [inp], [vi])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        report = _ANALYZER.analyze_model(model)
        unclassified = [i for i in report.issues if i.rule_id == RULE_UNCLASSIFIED]
        assert len(unclassified) == 1
        assert "2 nodes" in unclassified[0].node_name

    def test_runtime_verified_upgrades_to_verified(self) -> None:
        report = AnalysisReport(
            filename="f",
            onnx_ir_version="8",
            opset_version=17,
            producer="p",
            total_nodes=1,
            runtime_verified=True,
        )
        assert report.verdict is Verdict.VERIFIED

    def test_runtime_verified_never_overrides_blocked(
        self, uint8_input_model: onnx.ModelProto
    ) -> None:
        report = _ANALYZER.analyze_model(uint8_input_model)
        report.runtime_verified = True
        assert report.verdict is Verdict.BLOCKED


class TestJsonSchema:
    def test_report_carries_schema_version_verdict_and_rule_ids(
        self, uint8_input_model: onnx.ModelProto
    ) -> None:
        report = _ANALYZER.analyze_model(uint8_input_model)
        payload = json.loads(JSONReporter().render(report))
        assert payload["schema_version"] == REPORT_SCHEMA_VERSION
        assert payload["verdict"] == "blocked"
        assert payload["target_trt"] == "10.3"
        # Deprecated 1.x keys are still present for old consumers.
        assert payload["conversion_likely"] is False
        issue = payload["issues"][0]
        for key in ("rule_id", "confidence", "verify_required", "target_trt", "graph_scope"):
            assert key in issue
        assert issue["rule_id"].startswith("TRT-")

    def test_every_emitted_issue_has_a_rule_id(
        self, sequence_empty_model: onnx.ModelProto, control_flow_loop_model: onnx.ModelProto
    ) -> None:
        for model in (sequence_empty_model, control_flow_loop_model):
            report = _ANALYZER.analyze_model(model)
            assert report.issues
            for issue in report.issues:
                assert issue.rule_id, f"issue without rule_id: {issue.message}"
                assert issue.target_trt == "10.3"
