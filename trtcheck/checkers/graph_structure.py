"""Structural sanity checks for an ONNX graph.

These checks don't require any operator-support knowledge -- they look at
the shape of the graph itself: do outputs exist, are node names unique, are
there unusually large constants that might inflate the engine.

Remediation/explanation/severity live in remediation_db.json (via
:mod:`trtcheck.remediation`); this checker supplies the per-node prefix.
"""

from __future__ import annotations

from collections import Counter

import onnx

from trtcheck import remediation
from trtcheck._graph import iter_nodes, iter_subgraphs
from trtcheck.types import Issue

_LARGE_CONSTANT_BYTES = 10 * 1024 * 1024  # 10 MiB
# TRT 10.x has its best/most-direct operator coverage at ONNX opset 17+; older
# opsets force conservative decompositions. Bump here when the matrix advances.
_RECOMMENDED_OPSET_FLOOR = 17

# Remediation-DB keys this checker can emit (guarded by tests/test_remediation_wiring.py).
EMITS = frozenset(
    {
        "missing_output",
        "duplicate_node_name",
        "large_constant",
        "opset_too_old",
        "input_with_no_type",
        "isolated_node",
    }
)


class GraphStructureChecker:
    name = "graph_structure"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        graph = model.graph
        issues: list[Issue] = []
        issues.extend(self._check_outputs(graph))
        issues.extend(self._check_input_types(graph))
        issues.extend(self._check_duplicate_names(graph))
        issues.extend(self._check_large_constants(graph))
        issues.extend(self._check_opset(model))
        issues.extend(self._check_isolated_nodes(model))
        return issues

    # -- individual rules -------------------------------------------------

    def _check_outputs(self, graph: onnx.GraphProto) -> list[Issue]:
        if len(graph.output) == 0:
            return [
                remediation.make_issue(
                    "missing_output",
                    node_name=graph.name or "<graph>",
                    operator="Graph",
                    prefix="Graph declares zero outputs",
                )
            ]
        return []

    def _check_input_types(self, graph: onnx.GraphProto) -> list[Issue]:
        """Flag a graph input that declares no tensor element type.

        TensorRT's parser cannot bind an input without an element type; this
        usually means a corrupted/odd export. We gate on the TypeProto ``value``
        oneof so that legitimately non-tensor inputs (sequence/optional/map) are
        skipped, not false-flagged, and skip names that are also initializers
        (opset<9 duplication) since those carry a real dtype.
        """
        initializer_names = {init.name for init in graph.initializer}
        issues: list[Issue] = []
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            kind = inp.type.WhichOneof("value")
            typeless = kind is None or (
                kind == "tensor_type"
                and inp.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
            )
            if typeless:
                issues.append(
                    remediation.make_issue(
                        "input_with_no_type",
                        node_name=inp.name or "<input>",
                        operator="Input",
                        prefix=f"Input '{inp.name}' declares no tensor element type",
                    )
                )
        return issues

    def _check_duplicate_names(self, graph: onnx.GraphProto) -> list[Issue]:
        # Anonymous nodes (empty name) are common and should not trip the dup check.
        named = [n.name for n in graph.node if n.name]
        counts = Counter(named)
        duplicates = {name for name, count in counts.items() if count > 1}
        return [
            remediation.make_issue(
                "duplicate_node_name",
                node_name=name,
                operator="Graph",
                prefix=f"Node name '{name}' is used by {counts[name]} nodes (duplicate)",
            )
            for name in sorted(duplicates)
        ]

    def _check_large_constants(self, graph: onnx.GraphProto) -> list[Issue]:
        issues: list[Issue] = []
        # Initializers
        for init in graph.initializer:
            size = _initializer_size_bytes(init)
            if size > _LARGE_CONSTANT_BYTES:
                issues.append(self._large_const_issue(init.name or "<initializer>", size))
        # Constant nodes (rare in modern exports but seen in older ONNX files)
        for node in graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    size = _initializer_size_bytes(attr.t)
                    if size > _LARGE_CONSTANT_BYTES:
                        issues.append(self._large_const_issue(node.name or "<constant>", size))
        return issues

    def _check_opset(self, model: onnx.ModelProto) -> list[Issue]:
        # Same default-domain resolution analyzer.py uses for report.opset_version,
        # so header and finding agree. default=0 means the model declares no
        # ai.onnx opset (pure custom-domain) -- that is NOT "ancient opset", so
        # the != 0 guard prevents a bogus "opset 0" report.
        default_opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            default=0,
        )
        if default_opset != 0 and default_opset < _RECOMMENDED_OPSET_FLOOR:
            return [
                remediation.make_issue(
                    "opset_too_old",
                    node_name="<model>",
                    operator="Model",
                    prefix=(
                        f"Model default-domain opset is {default_opset}, below the "
                        f"recommended floor of {_RECOMMENDED_OPSET_FLOOR}"
                    ),
                )
            ]
        return []

    def _check_isolated_nodes(self, model: onnx.ModelProto) -> list[Issue]:
        """Flag a node whose outputs are consumed by nothing (dead code).

        TRT silently drops disconnected nodes, which can mask a lost in-place
        side effect. The consumed-name set is built across ALL subgraphs (node
        inputs + every graph's outputs) so an outer-scope capture -- a top-level
        tensor used by name inside a branch body without a subgraph input -- is
        not mistaken for dead. A node fires only if EVERY non-empty output is
        unconsumed (a multi-output op with one live output is not isolated).
        """
        consumed: set[str] = set()
        for graph in iter_subgraphs(model.graph):
            for node in graph.node:
                consumed.update(name for name in node.input if name)
            consumed.update(out.name for out in graph.output if out.name)

        issues: list[Issue] = []
        for node, _owner in iter_nodes(model.graph):
            outputs = [name for name in node.output if name]
            if outputs and not any(name in consumed for name in outputs):
                issues.append(
                    remediation.make_issue(
                        "isolated_node",
                        node_name=node.name or f"<{node.op_type}>",
                        operator=node.op_type,
                        prefix=(
                            f"Node '{node.name}' ({node.op_type}) is isolated: "
                            "none of its outputs are consumed"
                        ),
                    )
                )
        return issues

    @staticmethod
    def _large_const_issue(name: str, size: int) -> Issue:
        mb = size / (1024 * 1024)
        return remediation.make_issue(
            "large_constant",
            node_name=name,
            operator="Constant",
            prefix=f"Large constant '{name}' is {mb:.1f} MiB",
        )


def _initializer_size_bytes(tensor: onnx.TensorProto) -> int:
    # element count * bytes-per-element (approximate; good enough to flag oversized tensors)
    element_count = 1
    for dim in tensor.dims:
        element_count *= dim
    bytes_per = _BYTES_PER_DTYPE.get(tensor.data_type, 4)
    return element_count * bytes_per


_BYTES_PER_DTYPE: dict[int, int] = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.STRING: 1,
    onnx.TensorProto.BOOL: 1,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.BFLOAT16: 2,
}
