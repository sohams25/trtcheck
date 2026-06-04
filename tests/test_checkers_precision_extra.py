"""Regression tests for precision false-negatives fixed in the 2026-06 audit.

Covers INT64 graph inputs (the most common real TRT input dtype problem) and
FLOAT64 introduced *inside* the graph via Cast(to=DOUBLE) or a DOUBLE Constant.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from trtcheck.checkers.precision import PrecisionChecker
from trtcheck.types import Severity


def _model(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    opset: int = 17,
) -> onnx.ModelProto:
    g = helper.make_graph(nodes, "g", inputs, outputs)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])


def test_int64_graph_input_is_flagged_warning() -> None:
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", 128])
    out = helper.make_tensor_value_info("y", TensorProto.INT64, ["batch", 128])
    model = _model([helper.make_node("Identity", ["input_ids"], ["y"])], [ids], [out])

    issues = PrecisionChecker().check(model)

    int64_inputs = [i for i in issues if i.node_name == "input_ids" and "INT64" in i.message]
    assert int64_inputs, "INT64 graph input must be flagged"
    assert int64_inputs[0].severity is Severity.WARNING
    assert int64_inputs[0].operator == "Input"


def test_cast_to_double_is_critical() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("xd", TensorProto.DOUBLE, [4])
    cast = helper.make_node("Cast", ["x"], ["xd"], name="to_double", to=TensorProto.DOUBLE)
    model = _model([cast], [x], [out])

    issues = PrecisionChecker().check(model)

    double = [i for i in issues if i.operator == "Cast" and "DOUBLE" in i.message]
    assert double, "Cast(to=DOUBLE) introduces FLOAT64 and must be flagged"
    assert double[0].severity is Severity.CRITICAL


def test_constant_double_tensor_is_critical() -> None:
    out = helper.make_tensor_value_info("c", TensorProto.DOUBLE, [2])
    const = helper.make_node(
        "Constant",
        [],
        ["c"],
        name="dbl_const",
        value=helper.make_tensor("v", TensorProto.DOUBLE, [2], [1.0, 2.0]),
    )
    model = _model([const], [], [out])

    issues = PrecisionChecker().check(model)

    double = [i for i in issues if i.operator == "Constant" and "DOUBLE" in i.message]
    assert double, "Constant holding a DOUBLE tensor must be flagged"
    assert double[0].severity is Severity.CRITICAL


def test_float32_cast_is_not_flagged() -> None:
    """A normal Cast to FLOAT must not be reported as a double-precision issue."""
    x = helper.make_tensor_value_info("x", TensorProto.DOUBLE, [4])
    out = helper.make_tensor_value_info("xf", TensorProto.FLOAT, [4])
    cast = helper.make_node("Cast", ["x"], ["xf"], name="to_float", to=TensorProto.FLOAT)
    model = _model([cast], [x], [out])

    issues = PrecisionChecker().check(model)

    assert not [i for i in issues if i.operator == "Cast"]
