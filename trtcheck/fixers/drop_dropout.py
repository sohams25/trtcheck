"""Remove Dropout nodes from the graph.

TensorRT folds Dropout out of inference engines anyway. Removing the nodes
at fix time keeps the diagnostic report cleaner and makes downstream
analyses (visualization, latency estimates) more accurate.

The fixer refuses when the Dropout emits a mask output that is referenced
elsewhere -- the mask is a real value some models use, not just training
noise.

Removal is inference-semantics-preserving ONLY when the node is provably in
inference mode. Opset >= 12 Dropout takes an optional third ``training_mode``
input: absent or statically-false means inference (identity); true, dynamic,
or unresolvable means the node's behavior is not the identity and it must be
left alone. Opset <= 6 Dropout carries an ``is_test`` attribute with the same
role (default 0 = training).
"""

from __future__ import annotations

import logging

import onnx
from onnx import numpy_helper

from trtcheck._graph import iter_nodes, iter_subgraphs
from trtcheck.fixers import FixApplied

_logger = logging.getLogger("trtcheck.fixers")


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
        default_opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            default=0,
        )

        # Walk a copy of the node list because we mutate graph.node mid-loop.
        for node in list(graph.node):
            if node.op_type != "Dropout":
                continue

            # Only remove a Dropout that is provably in inference mode.
            mode = _resolve_inference_mode(model, node, default_opset)
            if mode != "inference":
                _logger.info(
                    "drop_dropout: skipping '%s': training mode is %s",
                    node.name or "<Dropout>",
                    mode,
                )
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


def _resolve_inference_mode(
    model: onnx.ModelProto, node: onnx.NodeProto, default_opset: int
) -> str:
    """Classify a Dropout node's training mode.

    Returns ``"inference"`` when removal preserves inference semantics,
    otherwise a short reason string ("training", "dynamic", "ambiguous", ...)
    used for the skip log.
    """
    if default_opset != 0 and default_opset <= 6:
        # Opset <= 6: is_test attribute, default 0 (training behavior).
        is_test = next((a.i for a in node.attribute if a.name == "is_test"), 0)
        return "inference" if is_test == 1 else "training (is_test != 1)"

    # Opset 7-11 Dropout has no training switch: inference semantics are the
    # identity. Opset >= 12 adds the optional training_mode input.
    if len(node.input) < 3 or not node.input[2]:
        return "inference"

    name = node.input[2]
    value, reason = _resolve_static_bool(model, name)
    if value is None:
        return reason
    return "inference" if value is False else "training (training_mode=true)"


def _resolve_static_bool(model: onnx.ModelProto, name: str) -> tuple[bool | None, str]:
    """Resolve ``name`` to a static scalar bool if it is an initializer or a
    Constant node output. Returns (value, reason-if-unresolvable).

    A name produced in more than one scope is ambiguous (shadowing) and is
    refused rather than guessed at.
    """
    initializers: list[onnx.TensorProto] = []
    producer_nodes: list[onnx.NodeProto] = []
    for graph in iter_subgraphs(model.graph):
        for init in graph.initializer:
            if init.name == name:
                initializers.append(init)
        for n in graph.node:
            if name in n.output:
                producer_nodes.append(n)

    total_defs = len(initializers) + len(producer_nodes)
    if total_defs > 1:
        return None, "ambiguous (name defined in multiple scopes)"
    if total_defs == 0:
        return None, "dynamic (fed from a graph input or unresolved name)"

    tensor: onnx.TensorProto | None
    if initializers:
        tensor = initializers[0]
    else:
        producer = producer_nodes[0]
        if producer.op_type != "Constant":
            return None, "dynamic (produced by a non-Constant node)"
        tensor = next(
            (a.t for a in producer.attribute if a.name == "value" and a.HasField("t")),
            None,
        )
        if tensor is None:
            return None, "Constant producer carries no tensor value"
    try:
        arr = numpy_helper.to_array(tensor)
    except Exception:
        return None, "unreadable training_mode tensor"
    if arr.size != 1:
        return None, "training_mode is not a scalar"
    return bool(arr.reshape(())), ""


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
