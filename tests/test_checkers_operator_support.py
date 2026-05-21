"""Tests for OperatorSupportChecker."""

import json
from pathlib import Path

import onnx
import pytest

from trtcheck.checkers.operator_support import OperatorSupportChecker
from trtcheck.types import CheckCategory, Severity

_MATRIX_PATH = Path(__file__).parent.parent / "trtcheck" / "data" / "operator_matrix.json"


@pytest.fixture(scope="module")
def matrix() -> dict:
    with open(_MATRIX_PATH) as f:
        return json.load(f)


class TestOperatorSupportChecker:
    def test_clean_model_produces_no_issues(self, clean_model: onnx.ModelProto) -> None:
        # Conv + Relu are both 'supported' on every target version.
        issues = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="10.3").check(
            clean_model
        )
        assert issues == []

    def test_sequence_empty_is_critical(self, sequence_empty_model: onnx.ModelProto) -> None:
        checker = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="10.3")
        issues = checker.check(sequence_empty_model)
        criticals = [i for i in issues if i.severity is Severity.CRITICAL]
        assert any(i.operator == "SequenceEmpty" for i in criticals)
        for i in issues:
            assert i.category is CheckCategory.OPERATOR_SUPPORT

    def test_partial_support_is_warning(self) -> None:
        # GroupNormalization is "not_supported" on 8.0/8.6 and "supported" on 10+.
        # Mish is partial in 8.6 and supported in 10+.
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        mish = helper.make_node("Mish", ["input"], ["output"], name="mish_1")
        graph = helper.make_graph([mish], "mish_model", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        checker = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="8.6")
        issues = checker.check(model)
        warnings = [i for i in issues if i.severity is Severity.WARNING and i.operator == "Mish"]
        assert warnings, "Mish should emit a WARNING on TRT 8.6"

    def test_not_supported_on_old_version_but_supported_on_new(self) -> None:
        # GroupNormalization: not_supported on 8.0, supported on 10.3
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
        scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4])
        bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4])
        gn = helper.make_node(
            "GroupNormalization",
            ["input", "scale", "bias"],
            ["output"],
            name="gn_1",
            num_groups=2,
        )
        graph = helper.make_graph([gn], "gn_model", [inp, scale, bias], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8

        old = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="8.0").check(model)
        new = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="10.3").check(model)
        assert any(i.severity is Severity.CRITICAL for i in old)
        assert all(i.operator != "GroupNormalization" for i in new)

    def test_unknown_operator_emits_info_not_critical(self) -> None:
        # Ops we don't know about should not panic users -- info only.
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        custom = helper.make_node(
            "MyCustomLayer", ["input"], ["output"], name="custom_1", domain="com.example"
        )
        graph = helper.make_graph([custom], "custom_model", [inp], [out])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 17),
                helper.make_opsetid("com.example", 1),
            ],
        )
        model.ir_version = 8
        issues = OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="10.3").check(model)
        assert all(i.severity is not Severity.CRITICAL for i in issues)

    def test_unknown_target_version_raises(self) -> None:
        with pytest.raises(ValueError):
            OperatorSupportChecker(matrix_path=_MATRIX_PATH, target_trt="99.9")
