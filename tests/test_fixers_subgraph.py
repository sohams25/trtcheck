"""Regression tests for fixer subgraph descent (FIX-2) and the empty-INT64 crash (FIX-1)."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import apply_all, default_fixers
from trtcheck.fixers.drop_dropout import DropDropoutFixer
from trtcheck.fixers.int64_to_int32 import Int64ToInt32Fixer


def _undef_out(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def test_empty_int64_initializer_does_not_crash() -> None:
    """An empty INT64 initializer is legal ONNX; the fixer must skip it, not raise."""
    empty = numpy_helper.from_array(np.array([], dtype=np.int64), name="empty_idx")
    g = helper.make_graph(
        [helper.make_node("Identity", ["empty_idx"], ["y"], name="id")],
        "g",
        [],
        [_undef_out("y")],
        initializer=[empty],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    _new, applied = apply_all(model, [Int64ToInt32Fixer()])  # must not raise

    assert applied == []  # nothing to do, but no crash


def test_int64_initializer_inside_if_branch_is_fixed() -> None:
    """Descent proof: an INT64 initializer buried in an If branch, used only at
    an INT32-compatible position (Gather indices), is converted."""
    buried = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), name="buried_idx")
    then_g = helper.make_graph(
        [helper.make_node("Gather", ["data", "buried_idx"], ["t_out"], name="id", axis=0)],
        "then_body",
        [],
        [_undef_out("t_out")],
        initializer=[buried],
    )
    else_g = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["e_out"],
                name="ce",
                value=helper.make_tensor("ev", TensorProto.FLOAT, [1], [0.0]),
            )
        ],
        "else_body",
        [],
        [_undef_out("e_out")],
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
    g = helper.make_graph(
        [cond, ifnode],
        "root",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [10])],
        [_undef_out("if_out")],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    new_model, applied = apply_all(model, [Int64ToInt32Fixer()])

    assert applied, "fixer must descend into the If branch and cast the buried INT64 indices"
    # locate the buried initializer in the rewritten model's then_branch
    if_node = next(n for n in new_model.graph.node if n.op_type == "If")
    then_branch = next(a.g for a in if_node.attribute if a.name == "then_branch")
    buried_after = next(i for i in then_branch.initializer if i.name == "buried_idx")
    assert buried_after.data_type == TensorProto.INT32


def test_default_fixer_pipeline_handles_subgraphs_without_error() -> None:
    """The full --fix pipeline must run cleanly over a model with subgraphs."""
    buried = numpy_helper.from_array(np.array([4, 5], dtype=np.int64), name="bidx")
    then_g = helper.make_graph(
        [helper.make_node("Gather", ["data", "bidx"], ["t_out"], name="id", axis=0)],
        "then_body",
        [],
        [_undef_out("t_out")],
        initializer=[buried],
    )
    else_g = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["e_out"],
                name="ce",
                value=helper.make_tensor("ev", TensorProto.FLOAT, [1], [0.0]),
            )
        ],
        "else_body",
        [],
        [_undef_out("e_out")],
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
    g = helper.make_graph(
        [cond, ifnode],
        "root",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [10])],
        [_undef_out("if_out")],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    _new, applied = apply_all(model, default_fixers())  # must not raise

    assert any(fa.fixer == "int64_to_int32" for fa in applied)


def test_dropout_inside_subgraph_with_local_consumer_is_removed() -> None:
    """A Dropout whose output is consumed within the same subgraph is removable."""
    then_g = helper.make_graph(
        [
            helper.make_node("Dropout", ["t_in"], ["d_out"], name="drop"),
            helper.make_node("Relu", ["d_out"], ["t_out"], name="relu"),
        ],
        "then_body",
        [helper.make_tensor_value_info("t_in", TensorProto.FLOAT, [1])],
        [_undef_out("t_out")],
    )
    else_g = helper.make_graph(
        [helper.make_node("Identity", ["t_in"], ["e_out"], name="eid")],
        "else_body",
        [helper.make_tensor_value_info("t_in", TensorProto.FLOAT, [1])],
        [_undef_out("e_out")],
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
    g = helper.make_graph([cond, ifnode], "root", [], [_undef_out("if_out")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    new_model, applied = apply_all(model, [DropDropoutFixer()])

    assert any(fa.fixer == "drop_dropout" for fa in applied)
    if_node = next(n for n in new_model.graph.node if n.op_type == "If")
    then_branch = next(a.g for a in if_node.attribute if a.name == "then_branch")
    assert not any(n.op_type == "Dropout" for n in then_branch.node)
    relu = next(n for n in then_branch.node if n.op_type == "Relu")
    assert list(relu.input) == ["t_in"]  # rewired past the removed Dropout


def test_dropout_output_captured_by_other_scope_is_not_removed() -> None:
    """A Dropout whose output is read by another subgraph must be left in place
    -- within-graph rewiring could not fix that cross-scope edge."""
    captured = helper.make_node("Identity", ["d_out"], ["t_out"], name="capture")
    then_g = helper.make_graph([captured], "then_body", [], [_undef_out("t_out")])
    else_g = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["e_out"],
                name="ce",
                value=helper.make_tensor("ev", TensorProto.FLOAT, [1], [0.0]),
            )
        ],
        "else_body",
        [],
        [_undef_out("e_out")],
    )
    cond = helper.make_node(
        "Constant",
        [],
        ["cond"],
        name="cond",
        value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
    )
    drop = helper.make_node("Dropout", ["x"], ["d_out"], name="drop")
    ifnode = helper.make_node(
        "If", ["cond"], ["if_out"], name="i", then_branch=then_g, else_branch=else_g
    )
    g = helper.make_graph(
        [drop, cond, ifnode],
        "root",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [_undef_out("if_out")],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    new_model, applied = apply_all(model, [DropDropoutFixer()])

    assert not applied, "Dropout feeding another scope must not be removed"
    assert any(n.op_type == "Dropout" for n in new_model.graph.node)
