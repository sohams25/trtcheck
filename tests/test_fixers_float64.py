"""Tests for Float64ToFloat32Fixer."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.float64_to_float32 import Float64ToFloat32Fixer


def _model_with_double_initializer(values: np.ndarray) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, len(values)])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(values)])
    w = numpy_helper.from_array(values.astype(np.float64), name="w")
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph([ident], "m", [inp], [out], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestFloat64ToFloat32:
    def test_finite_double_values_are_cast(self) -> None:
        model = _model_with_double_initializer(np.array([1.5, -2.25, 0.0, 3.14]))
        new_model, applied = apply_all(model, [Float64ToFloat32Fixer()])
        assert len(applied) == 1
        assert applied[0].fixer == "float64_to_float32"
        assert applied[0].target == "w"
        new_init = next(i for i in new_model.graph.initializer if i.name == "w")
        assert new_init.data_type == TensorProto.FLOAT
        onnx.checker.check_model(new_model)

    def test_values_above_fp32_max_are_skipped(self) -> None:
        # 1e40 is finite as FP64 but overflows FP32.
        model = _model_with_double_initializer(np.array([1e40]))
        new_model, applied = apply_all(model, [Float64ToFloat32Fixer()])
        assert applied == []
        assert new_model.graph.initializer[0].data_type == TensorProto.DOUBLE

    def test_nan_and_inf_are_skipped(self) -> None:
        for bad in [np.nan, np.inf, -np.inf]:
            model = _model_with_double_initializer(np.array([1.0, bad, 2.0]))
            _, applied = apply_all(model, [Float64ToFloat32Fixer()])
            assert applied == [], f"value {bad} should block the fix"

    def test_original_model_is_untouched(self) -> None:
        model = _model_with_double_initializer(np.array([1.0, 2.0]))
        original_dtype = model.graph.initializer[0].data_type
        apply_all(model, [Float64ToFloat32Fixer()])
        assert model.graph.initializer[0].data_type == original_dtype

    def test_clean_model_emits_no_fixes(self, clean_model: onnx.ModelProto) -> None:
        _, applied = apply_all(clean_model, [Float64ToFloat32Fixer()])
        assert applied == []
