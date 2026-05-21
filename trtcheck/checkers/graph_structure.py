"""Structural sanity checks for an ONNX graph.

These checks don't require any operator-support knowledge -- they look at
the shape of the graph itself: do outputs exist, are node names unique, are
there unusually large constants that might inflate the engine.
"""

from __future__ import annotations

from collections import Counter

import onnx

from trtcheck.types import CheckCategory, Issue, Severity

_LARGE_CONSTANT_BYTES = 10 * 1024 * 1024  # 10 MiB


class GraphStructureChecker:
    name = "graph_structure"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        graph = model.graph
        issues: list[Issue] = []
        issues.extend(self._check_outputs(graph))
        issues.extend(self._check_duplicate_names(graph))
        issues.extend(self._check_large_constants(graph))
        return issues

    # -- individual rules -------------------------------------------------

    def _check_outputs(self, graph: onnx.GraphProto) -> list[Issue]:
        if len(graph.output) == 0:
            return [
                Issue(
                    severity=Severity.CRITICAL,
                    category=CheckCategory.GRAPH_STRUCTURE,
                    node_name=graph.name or "<graph>",
                    operator="Graph",
                    message="Graph declares zero outputs; TensorRT requires at least one.",
                    remediation=(
                        "Re-export the model with do_constant_folding=False to localize "
                        "the issue, or verify that forward() returns at least one tensor."
                    ),
                    docs_link=None,
                )
            ]
        return []

    def _check_duplicate_names(self, graph: onnx.GraphProto) -> list[Issue]:
        # Anonymous nodes (empty name) are common and should not trip the dup check.
        named = [n.name for n in graph.node if n.name]
        counts = Counter(named)
        duplicates = {name for name, count in counts.items() if count > 1}
        return [
            Issue(
                severity=Severity.WARNING,
                category=CheckCategory.GRAPH_STRUCTURE,
                node_name=name,
                operator="Graph",
                message=f"Node name '{name}' is used by {counts[name]} nodes (duplicate).",
                remediation=(
                    "Set unique node names during export or run an ONNX simplifier "
                    "pass before TensorRT conversion."
                ),
                docs_link=None,
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

    @staticmethod
    def _large_const_issue(name: str, size: int) -> Issue:
        mb = size / (1024 * 1024)
        return Issue(
            severity=Severity.INFO,
            category=CheckCategory.GRAPH_STRUCTURE,
            node_name=name,
            operator="Constant",
            message=f"Large constant '{name}' is {mb:.1f} MiB.",
            remediation=(
                "Verify the constant is a learned weight, not e.g. a baked-in image. "
                "Baked input data inflates engine size unnecessarily."
            ),
            docs_link=None,
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
