"""Schema-awareness regression tests for Int64ToInt32Fixer.

The headline regression: ONNX ``Reshape`` requires its ``shape`` input to be
INT64. The old fixer converted every in-range INT64 initializer, producing a
model that passes the shallow ``onnx.checker.check_model()`` but fails
``full_check=True`` (strict type inference). These tests prove the failure
mode exists and that the use-aware fixer refuses it.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import apply_all
from trtcheck.fixers.int64_to_int32 import Int64ToInt32Fixer


def _reshape_model() -> onnx.ModelProto:
    """A valid model whose INT64 initializer is Reshape's shape input."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 6])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    shape = numpy_helper.from_array(np.array([3, 4], dtype=np.int64), name="new_shape")
    reshape = helper.make_node("Reshape", ["x", "new_shape"], ["y"], name="r")
    graph = helper.make_graph([reshape], "m", [inp], [out], initializer=[shape])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestReshapeRegression:
    def test_original_model_is_fully_valid(self) -> None:
        onnx.checker.check_model(_reshape_model(), full_check=True)

    def test_blind_conversion_breaks_full_validation(self) -> None:
        """Proof of the failure mode: manually cast the shape tensor to INT32
        (what the old fixer did). The shallow check passes; full type
        inference rejects the model."""
        model = _reshape_model()
        init = model.graph.initializer[0]
        arr = numpy_helper.to_array(init)
        init.CopyFrom(numpy_helper.from_array(arr.astype(np.int32), name=init.name))

        onnx.checker.check_model(model)  # shallow check: no complaint
        with pytest.raises(Exception):
            onnx.checker.check_model(model, full_check=True)

    def test_fixer_refuses_reshape_shape_input(self) -> None:
        model = _reshape_model()
        fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []
        init = next(i for i in fixed.graph.initializer if i.name == "new_shape")
        assert init.data_type == TensorProto.INT64
        onnx.checker.check_model(fixed, full_check=True)


class TestMixedAndNestedUses:
    def test_shared_initializer_with_mixed_consumers_is_refused(self) -> None:
        """One initializer feeding both a safe position (Gather indices) and an
        unsafe one (Reshape shape): every use must be safe, so refuse."""
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
        out_g = helper.make_tensor_value_info("g_out", TensorProto.FLOAT, [2, 4])
        out_r = helper.make_tensor_value_info("r_out", TensorProto.FLOAT, [2, 8])
        shared = numpy_helper.from_array(np.array([2, 8], dtype=np.int64), name="shared")
        gather = helper.make_node("Gather", ["x", "shared"], ["g_out"], name="g", axis=0)
        reshape = helper.make_node("Reshape", ["x", "shared"], ["r_out"], name="r")
        graph = helper.make_graph(
            [gather, reshape], "m", [inp], [out_g, out_r], initializer=[shared]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model, full_check=True)

        fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []
        assert fixed.graph.initializer[0].data_type == TensorProto.INT64

    def test_outer_initializer_used_unsafely_inside_subgraph_is_refused(self) -> None:
        """A top-level INT64 initializer captured by a Reshape inside an If
        branch: the nested use must be seen and must veto the conversion."""
        shape64 = numpy_helper.from_array(np.array([4], dtype=np.int64), name="cap_shape")
        then_g = helper.make_graph(
            [helper.make_node("Reshape", ["data", "cap_shape"], ["t_out"], name="rs")],
            "then_body",
            [],
            [helper.make_tensor_value_info("t_out", TensorProto.FLOAT, [4])],
        )
        else_g = helper.make_graph(
            [helper.make_node("Identity", ["data"], ["e_out"], name="eid")],
            "else_body",
            [],
            [helper.make_tensor_value_info("e_out", TensorProto.FLOAT, [2, 2])],
        )
        cond = helper.make_node(
            "Constant",
            [],
            ["cond"],
            name="cond",
            value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
        )
        ifnode = helper.make_node(
            "If", ["cond"], ["if_out"], name="i", then_branch=then_g, else_branch=else_g
        )
        vi = onnx.ValueInfoProto()
        vi.name = "if_out"
        g = helper.make_graph(
            [cond, ifnode],
            "root",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 2])],
            [vi],
            initializer=[shape64],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

        _fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == [], "nested Reshape use must veto the conversion"

    def test_custom_domain_consumer_is_refused(self) -> None:
        idx = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="idx")
        node = helper.make_node("MyOp", ["idx"], ["y"], name="c", domain="com.example")
        vi = onnx.ValueInfoProto()
        vi.name = "y"
        g = helper.make_graph([node], "m", [], [vi], initializer=[idx])
        model = helper.make_model(
            g,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.example", 1)],
        )
        _fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []

    def test_shadowed_name_across_scopes_is_refused(self) -> None:
        """The same name defined as an initializer at two scopes: which one a
        nested consumer sees is scope-dependent, so the fixer must refuse."""
        outer = numpy_helper.from_array(np.array([0], dtype=np.int64), name="dup")
        inner = numpy_helper.from_array(np.array([1], dtype=np.int64), name="dup")
        then_g = helper.make_graph(
            [helper.make_node("Gather", ["data", "dup"], ["t_out"], name="g2", axis=0)],
            "then_body",
            [],
            [helper.make_tensor_value_info("t_out", TensorProto.FLOAT, [1])],
            initializer=[inner],
        )
        else_g = helper.make_graph(
            [helper.make_node("Identity", ["data"], ["e_out"], name="eid")],
            "else_body",
            [],
            [helper.make_tensor_value_info("e_out", TensorProto.FLOAT, [2])],
        )
        cond = helper.make_node(
            "Constant",
            [],
            ["cond"],
            name="cond",
            value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
        )
        ifnode = helper.make_node(
            "If", ["cond"], ["if_out"], name="i", then_branch=then_g, else_branch=else_g
        )
        gather = helper.make_node("Gather", ["data", "dup"], ["g_out"], name="g1", axis=0)
        vi = onnx.ValueInfoProto()
        vi.name = "if_out"
        vo = onnx.ValueInfoProto()
        vo.name = "g_out"
        g = helper.make_graph(
            [gather, cond, ifnode],
            "root",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2])],
            [vi, vo],
            initializer=[outer],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

        _fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []

    def test_unconsumed_initializer_is_refused(self) -> None:
        dead = numpy_helper.from_array(np.array([1], dtype=np.int64), name="dead")
        ident = helper.make_node("Identity", ["x"], ["y"], name="id")
        g = helper.make_graph(
            [ident],
            "m",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            initializer=[dead],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        _fixed, applied = apply_all(model, [Int64ToInt32Fixer()])
        assert applied == []
