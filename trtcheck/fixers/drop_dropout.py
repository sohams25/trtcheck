"""Remove Dropout nodes from the graph.

TensorRT folds Dropout out of inference engines anyway. Removing the nodes
at fix time keeps the diagnostic report cleaner and makes downstream
analyses (visualization, latency estimates) more accurate.

The fixer refuses when the Dropout emits a mask output that is referenced
elsewhere -- the mask is a real value some models use, not just training
noise.
"""

from __future__ import annotations

import onnx

from trtcheck._graph import iter_nodes, iter_subgraphs
from trtcheck.fixers import FixApplied


class DropDropoutFixer:
    name = "drop_dropout"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        # Rewiring a removed Dropout happens within the graph that owns it, so a
        # Dropout whose output is captured by another scope (an If/Loop/Scan body
        # reading it as an outer-scope value) cannot be removed safely from here.
        # We refuse those, and check the mask model-wide, so descent is sound.
        for graph in iter_subgraphs(model.graph):
            applied.extend(self._fix_graph(model, graph))
        return applied

    def _fix_graph(self, model: onnx.ModelProto, graph: onnx.GraphProto) -> list[FixApplied]:
        applied: list[FixApplied] = []

        # Walk a copy of the node list because we mutate graph.node mid-loop.
        for node in list(graph.node):
            if node.op_type != "Dropout":
                continue

            # The data output is always output[0]. Outputs[1:] are the mask
            # (opset 12+); if any of them is referenced anywhere in the model
            # (including other scopes), refuse.
            if len(node.output) > 1:
                extras = [name for name in node.output[1:] if name]
                if any(_referenced_in_model(model, name) for name in extras):
                    continue

            data_in = node.input[0]
            data_out = node.output[0]

            # If the data output is consumed outside this graph (captured by a
            # sibling/child subgraph), within-graph rewiring would leave that
            # edge dangling. Refuse rather than corrupt the model.
            if _consumed_outside(model, graph, data_out):
                continue

            # Case A: Dropout's data output is a graph output. Promote the
            # producer of data_in to emit data_out directly.
            if _name_is_graph_output(graph, data_out):
                producer = _find_producer(graph, data_in)
                if producer is None:
                    # data_in is a graph input -- renaming would change the
                    # graph signature. Refuse.
                    continue
                # Replace producer's output reference.
                for i, name in enumerate(producer.output):
                    if name == data_in:
                        producer.output[i] = data_out
                # Any other consumer of data_in now needs to read data_out.
                for other in graph.node:
                    if other is node or other is producer:
                        continue
                    for i, name in enumerate(other.input):
                        if name == data_in:
                            other.input[i] = data_out
            else:
                # Case B: rewire downstream consumers to skip the Dropout.
                for other in graph.node:
                    if other is node:
                        continue
                    for i, name in enumerate(other.input):
                        if name == data_out:
                            other.input[i] = data_in

            graph.node.remove(node)
            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=node.name or "<Dropout>",
                    description=(
                        f"removed Dropout node '{node.name}'; downstream "
                        f"consumers now read '{data_in}' directly"
                    ),
                )
            )
        return applied


def _referenced_in_model(model: onnx.ModelProto, name: str) -> bool:
    """True if `name` is a graph output of any scope or consumed as an input by
    any node anywhere in the model. (A Dropout never feeds itself, so there is
    no need to exclude the producing node.)"""
    for graph in iter_subgraphs(model.graph):
        if any(out.name == name for out in graph.output):
            return True
    for node, _owner in iter_nodes(model.graph):
        if name in node.input:
            return True
    return False


def _consumed_outside(model: onnx.ModelProto, graph: onnx.GraphProto, name: str) -> bool:
    """True if `name` is consumed (or exposed as a graph output) anywhere in the
    model beyond the owning ``graph``.

    Uses occurrence counting rather than node identity: protobuf's C/upb backend
    hands back a fresh Python wrapper on every field access, so `is`/`id()`
    comparisons across two iterations are unreliable. Counting references is
    identity-free and correct.
    """
    in_graph_inputs = sum(inp == name for n in graph.node for inp in n.input)
    in_model_inputs = sum(inp == name for n, _ in iter_nodes(model.graph) for inp in n.input)
    if in_model_inputs > in_graph_inputs:
        return True
    in_graph_outputs = sum(o.name == name for o in graph.output)
    in_model_outputs = sum(o.name == name for g in iter_subgraphs(model.graph) for o in g.output)
    return bool(in_model_outputs > in_graph_outputs)


def _name_is_graph_output(graph: onnx.GraphProto, name: str) -> bool:
    return any(out.name == name for out in graph.output)


def _find_producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    for node in graph.node:
        if name in node.output:
            return node  # type: ignore[no-any-return]
    return None
