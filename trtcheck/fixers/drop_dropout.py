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

from trtcheck.fixers import FixApplied


class DropDropoutFixer:
    name = "drop_dropout"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        graph = model.graph
        applied: list[FixApplied] = []

        # Walk a copy of the node list because we mutate graph.node mid-loop.
        for node in list(graph.node):
            if node.op_type != "Dropout":
                continue

            # The data output is always output[0]. Outputs[1:] are the mask
            # (opset 12+); if any of them is referenced, refuse.
            if len(node.output) > 1:
                extras = [name for name in node.output[1:] if name]
                if any(_is_referenced(graph, name, exclude=node) for name in extras):
                    continue

            data_in = node.input[0]
            data_out = node.output[0]

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


def _is_referenced(graph: onnx.GraphProto, name: str, *, exclude: onnx.NodeProto) -> bool:
    if any(out.name == name for out in graph.output):
        return True
    for node in graph.node:
        if node is exclude:
            continue
        if name in node.input:
            return True
    return False


def _name_is_graph_output(graph: onnx.GraphProto, name: str) -> bool:
    return any(out.name == name for out in graph.output)


def _find_producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    for node in graph.node:
        if name in node.output:
            return node  # type: ignore[no-any-return]
    return None
