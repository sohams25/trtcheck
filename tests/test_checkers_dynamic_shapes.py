"""Tests for DynamicShapeChecker."""

import onnx
from onnx import TensorProto, helper

from trtcheck.checkers.dynamic_shapes import DynamicShapeChecker
from trtcheck.types import CheckCategory, Severity


def _input_with_shape(shape: list) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph([ident], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestDynamicShapeChecker:
    def test_clean_static_input_no_issues(self, clean_model: onnx.ModelProto) -> None:
        assert DynamicShapeChecker().check(clean_model) == []

    def test_dynamic_batch_only_is_ok(self) -> None:
        # The healthy default -- only batch dim is symbolic.
        model = _input_with_shape(["batch", 3, 32, 32])
        assert DynamicShapeChecker().check(model) == []

    def test_fully_dynamic_input_is_warning(self, fully_dynamic_model: onnx.ModelProto) -> None:
        issues = DynamicShapeChecker().check(fully_dynamic_model)
        warnings = [i for i in issues if i.severity is Severity.WARNING]
        assert warnings, "fully dynamic input must emit at least one warning"
        for i in issues:
            assert i.category is CheckCategory.DYNAMIC_SHAPES

    def test_remediation_mentions_dynamic_axes(self, fully_dynamic_model: onnx.ModelProto) -> None:
        issues = DynamicShapeChecker().check(fully_dynamic_model)
        assert any("dynamic_axes" in i.remediation for i in issues)

    def test_two_of_four_dynamic_dims_is_warning(self) -> None:
        # Spatial dims dynamic but channel concrete -- still flag.
        model = _input_with_shape(["batch", 3, "h", "w"])
        issues = DynamicShapeChecker().check(model)
        assert any(i.severity is Severity.WARNING for i in issues)
