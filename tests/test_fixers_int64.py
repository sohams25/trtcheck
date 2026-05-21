"""Tests for Int64ToInt32Fixer."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.int64_to_int32 import Int64ToInt32Fixer


def _model_with_int64_indices(values: np.ndarray) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [len(values), 4])
    idx = numpy_helper.from_array(values.astype(np.int64), name="indices")
    gather = helper.make_node("Gather", ["input", "indices"], ["output"], name="g", axis=0)
    graph = helper.make_graph([gather], "m", [inp], [out], initializer=[idx])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestInt64ToInt32:
    def test_small_int64_values_cast_cleanly(self) -> None:
        model = _model_with_int64_indices(np.array([0, 1, 5, 9]))
        new_model, applied = apply_all(model, [Int64ToInt32Fixer()])
        # exactly one fix
        assert len(applied) == 1
        assert applied[0].fixer == "int64_to_int32"
        assert applied[0].target == "indices"
        # new model: initializer is INT32 and values preserved
        new_init = next(i for i in new_model.graph.initializer if i.name == "indices")
        assert new_init.data_type == TensorProto.INT32
        recovered = numpy_helper.to_array(new_init)
        np.testing.assert_array_equal(recovered, np.array([0, 1, 5, 9], dtype=np.int32))
        # result must still validate
        onnx.checker.check_model(new_model)

    def test_original_model_is_untouched(self) -> None:
        model = _model_with_int64_indices(np.array([0, 1, 2]))
        original_dtype = model.graph.initializer[0].data_type
        apply_all(model, [Int64ToInt32Fixer()])
        assert model.graph.initializer[0].data_type == original_dtype

    def test_oversized_int64_is_skipped(self) -> None:
        # 2**40 cannot be represented in INT32; fixer must refuse.
        model = _model_with_int64_indices(np.array([2**40]))
        new_model, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []
        assert new_model.graph.initializer[0].data_type == TensorProto.INT64

    def test_clean_model_emits_no_fixes(self, clean_model: onnx.ModelProto) -> None:
        new_model, applied = apply_all(clean_model, [Int64ToInt32Fixer()])
        assert applied == []

    def test_negative_int64_values_cast_correctly(self) -> None:
        # Negative ints within INT32 range should pass.
        model = _model_with_int64_indices(np.array([-1, -100, 0, 50]))
        new_model, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert len(applied) == 1
        new_init = next(i for i in new_model.graph.initializer if i.name == "indices")
        recovered = numpy_helper.to_array(new_init)
        np.testing.assert_array_equal(recovered, np.array([-1, -100, 0, 50], dtype=np.int32))
