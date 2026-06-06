"""Tests for Uint8InputFixer."""

import onnx
from onnx import TensorProto, helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.uint8_input import Uint8InputFixer


def _uint8_then_cast(cast_to: int = TensorProto.FLOAT) -> onnx.ModelProto:
    """Input(UINT8) -> Cast(to=FLOAT) -> Identity."""
    inp = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
    cast = helper.make_node("Cast", ["input"], ["casted"], name="cast_1", to=cast_to)
    ident = helper.make_node("Identity", ["casted"], ["output"], name="ident")
    graph = helper.make_graph([cast, ident], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _uint8_no_cast() -> onnx.ModelProto:
    """Input(UINT8) consumed directly by some op (no leading Cast to FLOAT).
    Fixer should refuse -- it can't tell what the user wants."""
    inp = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 4])
    ident = helper.make_node("Identity", ["input"], ["output"], name="ident")
    graph = helper.make_graph([ident], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestUint8Input:
    def test_uint8_followed_by_cast_to_float_is_fixed(self) -> None:
        model = _uint8_then_cast()
        new_model, applied = apply_all(model, [Uint8InputFixer()])
        assert len(applied) == 1
        assert applied[0].fixer == "uint8_input"
        assert applied[0].target == "input"
        # Input is now FLOAT
        new_inp = new_model.graph.input[0]
        assert new_inp.type.tensor_type.elem_type == TensorProto.FLOAT
        # Cast node is gone
        op_types = [n.op_type for n in new_model.graph.node]
        assert "Cast" not in op_types
        # Downstream Identity now consumes 'input' directly
        ident = next(n for n in new_model.graph.node if n.op_type == "Identity")
        assert ident.input[0] == "input"
        onnx.checker.check_model(new_model)

    def test_original_model_is_untouched(self) -> None:
        model = _uint8_then_cast()
        original_dtype = model.graph.input[0].type.tensor_type.elem_type
        original_node_count = len(model.graph.node)
        apply_all(model, [Uint8InputFixer()])
        assert model.graph.input[0].type.tensor_type.elem_type == original_dtype
        assert len(model.graph.node) == original_node_count

    def test_uint8_without_immediate_cast_is_skipped(self) -> None:
        new_model, applied = apply_all(_uint8_no_cast(), [Uint8InputFixer()])
        assert applied == []
        assert new_model.graph.input[0].type.tensor_type.elem_type == TensorProto.UINT8

    def test_uint8_cast_to_non_float_is_skipped(self) -> None:
        # If the Cast targets INT32 the safe rewrite is not the same.
        new_model, applied = apply_all(
            _uint8_then_cast(cast_to=TensorProto.INT32), [Uint8InputFixer()]
        )
        assert applied == []

    def test_clean_model_emits_no_fixes(self, clean_model: onnx.ModelProto) -> None:
        new_model, applied = apply_all(clean_model, [Uint8InputFixer()])
        assert applied == []

    def test_uint8_input_is_also_graph_output_is_refused(self) -> None:
        """The UINT8 input is forwarded directly to a graph output (a legal
        passthrough) *and* feeds a single Cast. Promoting the input to FLOAT
        would leave the same-named output still declaring UINT8 -- a model that
        fails full type inference. The fixer must refuse rather than emit it.
        """
        inp = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, 3, 8, 8])
        out_pass = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, 3, 8, 8])
        out_main = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 8, 8])
        cast = helper.make_node("Cast", ["img"], ["casted"], name="cast_1", to=TensorProto.FLOAT)
        ident = helper.make_node("Identity", ["casted"], ["out"], name="ident")
        graph = helper.make_graph([cast, ident], "m", [inp], [out_pass, out_main])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        # Premise: the input model is valid under full type inference.
        onnx.checker.check_model(model, full_check=True)

        new_model, applied = apply_all(model, [Uint8InputFixer()])
        assert applied == [], "fixer must refuse promoting an input that is also a graph output"
        assert new_model.graph.input[0].type.tensor_type.elem_type == TensorProto.UINT8
        onnx.checker.check_model(new_model, full_check=True)

    def test_uint8_cast_output_is_graph_output_is_refused(self) -> None:
        """When the redundant Cast's output is itself the graph output, the naive
        rewrite renames the output to the input name and removes the node,
        collapsing the graph into a degenerate node-less input==output identity.
        The fixer must refuse instead of emitting that silently-wrong rewrite.
        """
        inp = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, 3, 8, 8])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 8, 8])
        cast = helper.make_node("Cast", ["img"], ["out"], name="cast_1", to=TensorProto.FLOAT)
        graph = helper.make_graph([cast], "m", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8

        new_model, applied = apply_all(model, [Uint8InputFixer()])
        assert applied == [], "fixer must refuse a rewrite that yields input==output identity"
        # Untouched: input stays UINT8, the Cast is still present and producing a
        # distinctly-named output.
        assert new_model.graph.input[0].type.tensor_type.elem_type == TensorProto.UINT8
        assert [n.op_type for n in new_model.graph.node] == ["Cast"]
        input_names = {i.name for i in new_model.graph.input}
        output_names = {o.name for o in new_model.graph.output}
        assert input_names.isdisjoint(output_names)
