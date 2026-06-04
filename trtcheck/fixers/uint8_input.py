"""Promote a UINT8 graph input to FLOAT when its only consumer is a Cast.

The safe pattern is:

    Input(UINT8) -> Cast(to=FLOAT) -> rest_of_graph

becoming:

    Input(FLOAT) -> rest_of_graph

If the UINT8 input is consumed by anything other than a single Cast(to=FLOAT),
the fixer refuses -- the right rewrite depends on what the rest of the
graph expected the UINT8 value to mean (raw bytes, normalized image, etc.).
"""

from __future__ import annotations

import onnx
from onnx import TensorProto

from trtcheck._graph import iter_subgraphs
from trtcheck.fixers import FixApplied


class Uint8InputFixer:
    name = "uint8_input"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        for graph in iter_subgraphs(model.graph):
            applied.extend(self._fix_graph(graph))
        return applied

    def _fix_graph(self, graph: onnx.GraphProto) -> list[FixApplied]:
        applied: list[FixApplied] = []

        # Map: input name -> list of (node_index, input_position) that consume it
        for inp in list(graph.input):
            if inp.type.tensor_type.elem_type != TensorProto.UINT8:
                continue

            consumers = _consumers_of(graph, inp.name)
            if len(consumers) != 1:
                continue  # ambiguous

            node, _ = consumers[0]
            if node.op_type != "Cast":
                continue
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr is None or to_attr.i != TensorProto.FLOAT:
                continue

            # Safe to rewrite:
            #   - promote input dtype to FLOAT
            #   - delete the Cast node, rewire its output to point at the input
            cast_output = node.output[0]
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            for other in graph.node:
                if other is node:
                    continue
                for i, name in enumerate(other.input):
                    if name == cast_output:
                        other.input[i] = inp.name
            # Also update graph outputs that may reference cast_output
            for out in graph.output:
                if out.name == cast_output:
                    out.name = inp.name
            graph.node.remove(node)

            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=inp.name,
                    description=(
                        f"promote input '{inp.name}' from UINT8 to FLOAT and "
                        f"drop the redundant Cast node '{node.name}'"
                    ),
                )
            )
        return applied


def _consumers_of(graph: onnx.GraphProto, name: str) -> list[tuple[onnx.NodeProto, int]]:
    """Return (node, input_position) pairs that consume `name`."""
    out: list[tuple[onnx.NodeProto, int]] = []
    for node in graph.node:
        for i, inp_name in enumerate(node.input):
            if inp_name == name:
                out.append((node, i))
    return out
