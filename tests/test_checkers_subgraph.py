"""Subgraph-coverage regression tests.

trtcheck's core promise is catching conversion blockers. Before these tests,
every checker except control_flow scanned only ``model.graph.node`` (and
control_flow recursed only via the ``body`` attribute), so an unsupported op
hidden inside an If branch produced a false "LIKELY TO CONVERT" verdict.
These pin the fix: checkers must descend into If/Loop/Scan subgraphs.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from trtcheck.analyzer import Analyzer, AnalyzerConfig
from trtcheck.types import Severity


def _undef_out(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def _if_model_with_branch_nodes(
    then_nodes: list[onnx.NodeProto], else_nodes: list[onnx.NodeProto]
) -> onnx.ModelProto:
    then_g = helper.make_graph(then_nodes, "then_body", [], [_undef_out("t_out")])
    else_g = helper.make_graph(else_nodes, "else_body", [], [_undef_out("e_out")])
    cond = helper.make_node(
        "Constant",
        [],
        ["cond"],
        name="cond",
        value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
    )
    ifnode = helper.make_node(
        "If", ["cond"], ["if_out"], name="top_if", then_branch=then_g, else_branch=else_g
    )
    g = helper.make_graph([cond, ifnode], "root", [], [_undef_out("if_out")])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def test_unsupported_op_in_if_then_branch_is_detected() -> None:
    seq_then = helper.make_node("SequenceEmpty", [], ["t_out"], name="hidden_seq")
    seq_else = helper.make_node("SequenceEmpty", [], ["e_out"], name="hidden_seq_else")
    model = _if_model_with_branch_nodes([seq_then], [seq_else])

    report = Analyzer(AnalyzerConfig()).analyze_model(model)

    seq_issues = [i for i in report.issues if i.operator == "SequenceEmpty"]
    assert seq_issues, "unsupported op hidden in an If branch must be flagged"
    assert any(i.severity is Severity.CRITICAL for i in seq_issues)
    assert report.conversion_likely is False


def test_int64_initializer_in_subgraph_is_flagged() -> None:
    big = helper.make_tensor("buried_idx", TensorProto.INT64, [2], [1, 2])
    then_g = helper.make_graph(
        [helper.make_node("Identity", ["buried_idx"], ["t_out"], name="id")],
        "then_body",
        [],
        [_undef_out("t_out")],
        initializer=[big],
    )
    else_g = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["e_out"],
                name="ce",
                value=helper.make_tensor("ev", TensorProto.FLOAT, [1], [0.0]),
            )
        ],
        "else_body",
        [],
        [_undef_out("e_out")],
    )
    cond = helper.make_node(
        "Constant",
        [],
        ["cond"],
        name="cond",
        value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
    )
    ifnode = helper.make_node(
        "If", ["cond"], ["if_out"], name="i", then_branch=then_g, else_branch=else_g
    )
    g = helper.make_graph([cond, ifnode], "root", [], [_undef_out("if_out")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    report = Analyzer(AnalyzerConfig()).analyze_model(model)

    int64 = [i for i in report.issues if "INT64" in i.message and i.node_name == "buried_idx"]
    assert int64, "INT64 weight buried in a subgraph must be flagged"
    assert int64[0].severity is Severity.WARNING


def test_nested_subgraphs_analyze_cleanly() -> None:
    """Analysis terminates cleanly on deep nesting (protobuf caps depth ~100)."""
    inner = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="leaf")],
        "leaf",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [_undef_out("y")],
    )
    for i in range(20):
        loop = helper.make_node("Loop", [], [f"o{i}"], name=f"loop{i}", body=inner)
        inner = helper.make_graph([loop], f"g{i}", [], [_undef_out(f"o{i}")])
    model = helper.make_model(inner, opset_imports=[helper.make_opsetid("", 17)])

    report = Analyzer(AnalyzerConfig()).analyze_model(model)  # must not raise
    assert report is not None
