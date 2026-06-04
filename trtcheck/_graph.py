"""Shared ONNX graph traversal helpers.

Several checkers and fixers need to see *every* node in a model, not just
the ones at the top level. TensorRT validates the bodies of ``If`` / ``Loop``
/ ``Scan`` subgraphs too, so an unsupported operator (or a fixable pattern)
buried in a branch is still a conversion blocker. Iterating only
``model.graph.node`` silently misses those -- a model can be reported
"likely to convert" while a conversion-stopping op hides inside an ``If``
branch.

These helpers walk the top-level graph plus every nested subgraph carried on
a node attribute (``then_branch``, ``else_branch``, ``body``, and the
``GRAPHS``-valued attributes some ops use). Traversal is depth-bounded so a
pathologically nested model cannot exhaust Python's recursion limit -- a
hand-crafted ONNX file with thousands of nested subgraphs is exactly the kind
of untrusted input trtcheck is meant to survive.
"""

from __future__ import annotations

from typing import Iterator

import onnx

# Real models nest a handful of subgraphs deep at most (a Loop inside an If
# inside a Loop is already exotic). 256 is far above anything legitimate while
# still bounding adversarial inputs well under Python's default recursion limit.
_MAX_SUBGRAPH_DEPTH = 256


def iter_subgraphs(
    graph: onnx.GraphProto, *, max_depth: int = _MAX_SUBGRAPH_DEPTH
) -> Iterator[onnx.GraphProto]:
    """Yield ``graph`` and every nested subgraph, depth-first.

    Recursion is bounded by ``max_depth``; subgraphs deeper than that are not
    visited (a deliberate trade-off: no real model nests that deep, and the
    bound is what keeps a malicious file from blowing the stack).
    """
    yield from _iter_subgraphs(graph, 0, max_depth)


def _iter_subgraphs(
    graph: onnx.GraphProto, depth: int, max_depth: int
) -> Iterator[onnx.GraphProto]:
    yield graph
    if depth >= max_depth:
        return
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _iter_subgraphs(attr.g, depth + 1, max_depth)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _iter_subgraphs(sub, depth + 1, max_depth)


def iter_nodes(
    graph: onnx.GraphProto, *, max_depth: int = _MAX_SUBGRAPH_DEPTH
) -> Iterator[tuple[onnx.NodeProto, onnx.GraphProto]]:
    """Yield ``(node, owning_graph)`` for every node in ``graph`` and its subgraphs.

    The owning graph is handed back so callers that need scope-local context
    (e.g. a subgraph's own initializers) do not have to re-derive it.
    """
    for sub in iter_subgraphs(graph, max_depth=max_depth):
        for node in sub.node:
            yield node, sub


def iter_initializers(
    graph: onnx.GraphProto, *, max_depth: int = _MAX_SUBGRAPH_DEPTH
) -> Iterator[tuple[onnx.TensorProto, onnx.GraphProto]]:
    """Yield ``(initializer, owning_graph)`` across ``graph`` and its subgraphs."""
    for sub in iter_subgraphs(graph, max_depth=max_depth):
        for init in sub.initializer:
            yield init, sub


def count_nodes(graph: onnx.GraphProto, *, max_depth: int = _MAX_SUBGRAPH_DEPTH) -> int:
    """Total node count including every nested subgraph."""
    return sum(1 for _ in iter_nodes(graph, max_depth=max_depth))
