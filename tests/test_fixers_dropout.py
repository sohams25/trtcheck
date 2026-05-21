"""Tests for DropDropoutFixer."""

import onnx
from onnx import TensorProto, helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.drop_dropout import DropDropoutFixer


def _model_with_dropout() -> onnx.ModelProto:
    """Input -> Identity -> Dropout -> Identity -> Output."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    a = helper.make_node("Identity", ["input"], ["mid1"], name="a")
    drop = helper.make_node("Dropout", ["mid1"], ["mid2"], name="drop", ratio=0.5)
    b = helper.make_node("Identity", ["mid2"], ["output"], name="b")
    graph = helper.make_graph([a, drop, b], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _model_with_dropout_two_outputs() -> onnx.ModelProto:
    """Dropout in opset 12+ can declare a mask output. If the mask is wired up,
    we cannot just drop the node -- skip it."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out_d = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    out_m = helper.make_tensor_value_info("mask", TensorProto.BOOL, [1, 4])
    drop = helper.make_node("Dropout", ["input"], ["output", "mask"], name="drop", ratio=0.5)
    graph = helper.make_graph([drop], "m", [inp], [out_d, out_m])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _model_dropout_at_input() -> onnx.ModelProto:
    """Dropout directly consuming a graph input. After dropping, the
    downstream node should read from the graph input directly."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    drop = helper.make_node("Dropout", ["input"], ["dropped"], name="drop", ratio=0.5)
    ident = helper.make_node("Identity", ["dropped"], ["output"], name="ident")
    graph = helper.make_graph([drop, ident], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestDropDropout:
    def test_single_output_dropout_is_removed_and_rewired(self) -> None:
        model = _model_with_dropout()
        new_model, applied = apply_all(model, [DropDropoutFixer()])
        assert len(applied) == 1
        assert applied[0].fixer == "drop_dropout"
        op_types = [n.op_type for n in new_model.graph.node]
        assert "Dropout" not in op_types
        # The downstream Identity should now read 'mid1' directly.
        b = next(n for n in new_model.graph.node if n.name == "b")
        assert b.input[0] == "mid1"
        onnx.checker.check_model(new_model)

    def test_dropout_with_used_mask_output_is_skipped(self) -> None:
        new_model, applied = apply_all(_model_with_dropout_two_outputs(), [DropDropoutFixer()])
        assert applied == []
        op_types = [n.op_type for n in new_model.graph.node]
        assert "Dropout" in op_types

    def test_dropout_directly_after_input(self) -> None:
        new_model, applied = apply_all(_model_dropout_at_input(), [DropDropoutFixer()])
        assert len(applied) == 1
        op_types = [n.op_type for n in new_model.graph.node]
        assert "Dropout" not in op_types
        ident = next(n for n in new_model.graph.node if n.op_type == "Identity")
        assert ident.input[0] == "input"
        onnx.checker.check_model(new_model)

    def test_dropout_feeding_graph_output(self) -> None:
        """Dropout whose output IS the graph output. After removal, the graph
        output must point at the Dropout's data input."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        ident = helper.make_node("Identity", ["input"], ["mid"], name="ident")
        drop = helper.make_node("Dropout", ["mid"], ["output"], name="drop")
        graph = helper.make_graph([ident, drop], "m", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        new_model, applied = apply_all(model, [DropDropoutFixer()])
        assert len(applied) == 1
        # The graph output is still named "output" -- the rewiring promotes
        # the Identity output to "output".
        out_names = [o.name for o in new_model.graph.output]
        assert "output" in out_names
        onnx.checker.check_model(new_model)

    def test_clean_model_emits_no_fixes(self, clean_model: onnx.ModelProto) -> None:
        _, applied = apply_all(clean_model, [DropDropoutFixer()])
        assert applied == []
