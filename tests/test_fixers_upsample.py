"""Tests for UpsampleToResizeFixer."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.upsample_to_resize import UpsampleToResizeFixer


def _model_upsample(mode: str = "nearest") -> onnx.ModelProto:
    """Opset 9 form: Upsample(X, scales) with mode attribute."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )
    up = helper.make_node("Upsample", ["input", "scales"], ["output"], name="up_1", mode=mode)
    graph = helper.make_graph([up], "m", [inp], [out], initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
    model.ir_version = 8
    return model


class TestUpsampleToResize:
    def test_nearest_mode_is_rewritten(self) -> None:
        new_model, applied = apply_all(_model_upsample("nearest"), [UpsampleToResizeFixer()])
        assert len(applied) == 1
        assert applied[0].fixer == "upsample_to_resize"
        op_types = [n.op_type for n in new_model.graph.node]
        assert "Upsample" not in op_types
        assert "Resize" in op_types
        resize = next(n for n in new_model.graph.node if n.op_type == "Resize")
        mode_attr = next(a for a in resize.attribute if a.name == "mode")
        assert mode_attr.s.decode() == "nearest"

    def test_linear_mode_is_rewritten(self) -> None:
        new_model, applied = apply_all(_model_upsample("linear"), [UpsampleToResizeFixer()])
        assert len(applied) == 1
        resize = next(n for n in new_model.graph.node if n.op_type == "Resize")
        mode_attr = next(a for a in resize.attribute if a.name == "mode")
        assert mode_attr.s.decode() == "linear"

    def test_unsupported_mode_is_skipped(self) -> None:
        # Upsample technically only had nearest/linear, but be defensive.
        _, applied = apply_all(_model_upsample("cubic"), [UpsampleToResizeFixer()])
        assert applied == []

    def test_resize_inputs_have_roi_placeholder(self) -> None:
        """Resize expects (X, roi, scales, sizes). roi and sizes must be empty
        strings when scales is provided."""
        new_model, _ = apply_all(_model_upsample("nearest"), [UpsampleToResizeFixer()])
        resize = next(n for n in new_model.graph.node if n.op_type == "Resize")
        assert len(resize.input) == 4
        assert resize.input[0] == "input"
        assert resize.input[1] == ""  # empty roi
        assert resize.input[2] == "scales"
        assert resize.input[3] == ""  # empty sizes

    def test_original_model_is_untouched(self) -> None:
        model = _model_upsample("nearest")
        apply_all(model, [UpsampleToResizeFixer()])
        op_types = [n.op_type for n in model.graph.node]
        assert "Upsample" in op_types and "Resize" not in op_types

    def test_clean_model_emits_no_fixes(self, clean_model: onnx.ModelProto) -> None:
        _, applied = apply_all(clean_model, [UpsampleToResizeFixer()])
        assert applied == []
