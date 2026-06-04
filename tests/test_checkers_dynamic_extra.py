"""Regression test: dynamic dims encoded as a concrete -1 must count as symbolic.

Several exporters (and TensorRT itself) represent an unknown dimension as
dim_value=-1 rather than a named dim_param. The checker previously treated -1
as a concrete extent, so a fully dynamic [-1,-1,-1,-1] input emitted no warning.
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from trtcheck.checkers.dynamic_shapes import DynamicShapeChecker
from trtcheck.types import CheckCategory


def _input_with_dims(name: str, dims: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)


def test_negative_one_dims_are_treated_as_dynamic() -> None:
    inp = _input_with_dims("x", [-1, -1, -1, -1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [-1, -1, -1, -1])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [inp], [out])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    issues = DynamicShapeChecker().check(model)

    dyn = [i for i in issues if i.category is CheckCategory.DYNAMIC_SHAPES]
    assert dyn, "[-1,-1,-1,-1] is fully dynamic and must warn"
    assert "4 of 4" in dyn[0].message


def test_single_batch_dim_minus_one_is_not_warned() -> None:
    """One dynamic dim (typical batch) stays under the >=2 threshold."""
    inp = _input_with_dims("x", [-1, 3, 224, 224])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [-1, 3, 224, 224])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [inp], [out])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    issues = DynamicShapeChecker().check(model)

    assert not [i for i in issues if i.category is CheckCategory.DYNAMIC_SHAPES]
