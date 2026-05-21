"""Tests for PrecisionChecker (dtype-level findings)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.checkers.precision import PrecisionChecker
from trtcheck.types import CheckCategory, Severity


def _model_with_input_dtype(dtype: int) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", dtype, [1, 4])
    out = helper.make_tensor_value_info("output", dtype, [1, 4])
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph(nodes=[ident], name="g", inputs=[inp], outputs=[out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _model_with_initializer(dtype: int, arr: np.ndarray) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(arr.astype(_NP_FROM_TP[dtype]), name="w")
    # Just use the input -- the initializer is what we're testing.
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph(
        nodes=[ident], name="g", inputs=[inp], outputs=[out], initializer=[weight]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


_NP_FROM_TP = {
    TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32,
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.UINT8: np.uint8,
    TensorProto.BFLOAT16: np.uint16,  # bf16 has no native numpy dtype in older numpy
}


class TestPrecisionChecker:
    def test_clean_fp32_model_produces_no_issues(self, clean_model: onnx.ModelProto) -> None:
        assert PrecisionChecker().check(clean_model) == []

    def test_uint8_input_is_critical(self, uint8_input_model: onnx.ModelProto) -> None:
        issues = PrecisionChecker().check(uint8_input_model)
        criticals = [i for i in issues if i.severity is Severity.CRITICAL]
        assert any("uint8" in i.message.lower() for i in criticals)
        for i in issues:
            assert i.category is CheckCategory.PRECISION

    def test_int64_initializer_is_warning(self, int64_weights_model: onnx.ModelProto) -> None:
        issues = PrecisionChecker().check(int64_weights_model)
        warnings = [i for i in issues if i.severity is Severity.WARNING]
        assert any("int64" in i.message.lower() for i in warnings)

    def test_double_input_is_critical(self) -> None:
        model = _model_with_input_dtype(TensorProto.DOUBLE)
        issues = PrecisionChecker().check(model)
        assert any(
            i.severity is Severity.CRITICAL and "double" in i.message.lower() for i in issues
        )

    def test_string_input_is_critical(self) -> None:
        model = _model_with_input_dtype(TensorProto.STRING)
        issues = PrecisionChecker().check(model)
        assert any(
            i.severity is Severity.CRITICAL and "string" in i.message.lower() for i in issues
        )

    def test_int64_remediation_suggests_int32_cast(
        self, int64_weights_model: onnx.ModelProto
    ) -> None:
        issues = PrecisionChecker().check(int64_weights_model)
        int64_issues = [i for i in issues if "int64" in i.message.lower()]
        assert int64_issues, "expected at least one INT64 issue"
        assert any("int32" in i.remediation.lower() for i in int64_issues)


def test_double_initializer_in_input_list_is_not_duplicated() -> None:
    """Old-opset ONNX duplicates initializers into graph.input. Each tensor
    must only produce one finding, not two with conflicting operator labels."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    # A DOUBLE initializer
    init = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float64), name="dbl_weight")
    # Duplicate into inputs (opset < 9 behavior)
    init_as_input = helper.make_tensor_value_info("dbl_weight", TensorProto.DOUBLE, [2])
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph([ident], "old_opset", [inp, init_as_input], [out], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    issues = PrecisionChecker().check(model)
    double_issues = [i for i in issues if "DOUBLE" in i.message or "double" in i.message.lower()]
    # Exactly one finding for the dbl_weight tensor.
    dbl_findings = [i for i in double_issues if i.node_name == "dbl_weight"]
    assert len(dbl_findings) == 1, f"expected 1 finding for dbl_weight, got {len(dbl_findings)}"
