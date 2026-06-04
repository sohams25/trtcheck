"""Unit tests for the shared subgraph traversal helpers in trtcheck._graph."""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from trtcheck._graph import count_nodes, iter_initializers, iter_nodes, iter_subgraphs


def _const(name: str, out: str) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [out],
        name=name,
        value=helper.make_tensor(name + "_v", TensorProto.FLOAT, [1], [1.0]),
    )


def _if_with_bodies(
    then_nodes: list[onnx.NodeProto], else_nodes: list[onnx.NodeProto], name: str = "top_if"
) -> tuple[onnx.NodeProto, onnx.NodeProto]:
    then_g = helper.make_graph(
        then_nodes,
        "then_body",
        [],
        [helper.make_tensor_value_info("t_out", TensorProto.FLOAT, [1])],
    )
    else_g = helper.make_graph(
        else_nodes,
        "else_body",
        [],
        [helper.make_tensor_value_info("e_out", TensorProto.FLOAT, [1])],
    )
    cond = helper.make_node(
        "Constant",
        [],
        ["cond"],
        name="c",
        value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
    )
    return cond, helper.make_node(
        "If", ["cond"], ["if_out"], name=name, then_branch=then_g, else_branch=else_g
    )


def test_iter_nodes_descends_into_if_branches() -> None:
    cond, ifnode = _if_with_bodies(
        then_nodes=[_const("inner_then", "t_out")],
        else_nodes=[_const("inner_else", "e_out")],
    )
    g = helper.make_graph(
        [cond, ifnode], "g", [], [helper.make_tensor_value_info("if_out", TensorProto.FLOAT, [1])]
    )
    op_types = [n.op_type for n, _owner in iter_nodes(g)]
    # top-level cond + If, plus the two Constants buried in the branches
    assert op_types.count("Constant") == 3
    assert "If" in op_types


def test_iter_nodes_reports_owning_graph() -> None:
    cond, ifnode = _if_with_bodies([_const("a", "t_out")], [_const("b", "e_out")])
    g = helper.make_graph(
        [cond, ifnode],
        "root",
        [],
        [helper.make_tensor_value_info("if_out", TensorProto.FLOAT, [1])],
    )
    owners = {n.name: owner.name for n, owner in iter_nodes(g)}
    assert owners["a"] == "then_body"
    assert owners["b"] == "else_body"
    assert owners["top_if"] == "root"


def test_iter_subgraphs_yields_root_first_then_nested() -> None:
    cond, ifnode = _if_with_bodies([_const("a", "t_out")], [_const("b", "e_out")])
    g = helper.make_graph(
        [cond, ifnode],
        "root",
        [],
        [helper.make_tensor_value_info("if_out", TensorProto.FLOAT, [1])],
    )
    names = [sub.name for sub in iter_subgraphs(g)]
    assert names[0] == "root"
    assert set(names) == {"root", "then_body", "else_body"}


def test_iter_initializers_includes_subgraph_initializers() -> None:
    init = helper.make_tensor("buried", TensorProto.INT64, [2], [1, 2])
    body = helper.make_graph(
        [_const("x", "t_out")],
        "then_body",
        [],
        [helper.make_tensor_value_info("t_out", TensorProto.FLOAT, [1])],
        initializer=[init],
    )
    else_g = helper.make_graph(
        [_const("y", "e_out")],
        "else_body",
        [],
        [helper.make_tensor_value_info("e_out", TensorProto.FLOAT, [1])],
    )
    cond = helper.make_node(
        "Constant",
        [],
        ["cond"],
        name="c",
        value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
    )
    ifnode = helper.make_node(
        "If", ["cond"], ["if_out"], name="i", then_branch=body, else_branch=else_g
    )
    g = helper.make_graph(
        [cond, ifnode],
        "root",
        [],
        [helper.make_tensor_value_info("if_out", TensorProto.FLOAT, [1])],
    )
    init_names = {init.name for init, _ in iter_initializers(g)}
    assert "buried" in init_names


def test_walker_descends_deep_nesting_and_terminates() -> None:
    """Nested subgraphs are walked to the leaf and traversal terminates cleanly.

    protobuf itself caps message-construction/parse depth (~100), so a model
    deep enough to overflow Python's recursion limit cannot even be built or
    onnx.load-ed; the depth bound in iter_subgraphs is defence-in-depth. Here we
    use a depth protobuf accepts and assert the walker reaches the leaf.
    """
    inner = helper.make_graph(
        [_const("leaf", "leaf_out")],
        "leaf",
        [],
        [helper.make_tensor_value_info("leaf_out", TensorProto.FLOAT, [1])],
    )
    depth = 20
    for i in range(depth):
        loop = helper.make_node("Loop", [], [f"o{i}"], name=f"loop{i}", body=inner)
        inner = helper.make_graph(
            [loop], f"g{i}", [], [helper.make_tensor_value_info(f"o{i}", TensorProto.FLOAT, [1])]
        )
    op_types = [n.op_type for n, _ in iter_nodes(inner)]
    assert "Constant" in op_types  # reached the buried leaf
    assert op_types.count("Loop") == depth
    assert count_nodes(inner) == depth + 1
