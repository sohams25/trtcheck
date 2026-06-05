"""Precision and dtype checks.

The classics TensorRT rejects or coerces:
  - UINT8 graph inputs (TRT accepts FP32/FP16/INT32/INT8 only)
  - FLOAT64 anywhere (TRT has no double precision)
  - String tensors (no TRT type)
  - BFLOAT16 (Ampere + TRT 8.6+ only)
  - INT64 inputs/weights/initializers (TRT casts to INT32, may overflow)

Remediation/explanation/severity for each finding live in remediation_db.json
(via :mod:`trtcheck.remediation`); this checker only supplies the per-node
prefix (which tensor, which dtype).
"""

from __future__ import annotations

import onnx
from onnx import TensorProto

from trtcheck import remediation
from trtcheck._graph import iter_initializers, iter_nodes
from trtcheck.types import Issue

# Remediation-DB keys this checker can emit (guarded by tests/test_data_files.py).
EMITS = frozenset(
    {
        "uint8_input",
        "int64_input",
        "float64_tensors",
        "string_tensors",
        "bf16_unsupported",
        "int64_weights",
    }
)

# Graph-input dtype -> (issue id, dtype token for the per-node prefix). Keyed by
# the raw int dtype code so lookups with onnx's int-typed elem_type/data_type
# fields type-check cleanly.
_INPUT_DTYPES: dict[int, tuple[str, str]] = {
    TensorProto.UINT8: ("uint8_input", "UINT8"),
    TensorProto.DOUBLE: ("float64_tensors", "DOUBLE"),
    TensorProto.STRING: ("string_tensors", "STRING"),
    TensorProto.INT64: ("int64_input", "INT64"),
    TensorProto.BFLOAT16: ("bf16_unsupported", "BFLOAT16"),
}

# Initializer dtype -> (issue id, dtype token).
_INIT_DTYPES: dict[int, tuple[str, str]] = {
    TensorProto.INT64: ("int64_weights", "INT64"),
    TensorProto.DOUBLE: ("float64_tensors", "DOUBLE"),
    TensorProto.BFLOAT16: ("bf16_unsupported", "BFLOAT16"),
}


class PrecisionChecker:
    name = "precision"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        issues.extend(self._check_inputs(model.graph))
        issues.extend(self._check_initializers(model.graph))
        issues.extend(self._check_internal_double(model.graph))
        return issues

    def _check_inputs(self, graph: onnx.GraphProto) -> list[Issue]:
        # In ONNX opset < 9 initializers are duplicated into graph.input. Skip
        # those so we don't emit the same finding from both _check_inputs and
        # _check_initializers (would surface as conflicting Input/Initializer
        # operator labels for the identical tensor).
        initializer_names = {init.name for init in graph.initializer}
        issues: list[Issue] = []
        for inp in graph.input:
            if inp.name in initializer_names:
                continue
            mapping = _INPUT_DTYPES.get(inp.type.tensor_type.elem_type)
            if mapping is None:
                continue
            issue_id, token = mapping
            issues.append(
                remediation.make_issue(
                    issue_id,
                    node_name=inp.name,
                    operator="Input",
                    prefix=f"Input '{inp.name}' has dtype {token}",
                )
            )
        return issues

    def _check_initializers(self, graph: onnx.GraphProto) -> list[Issue]:
        issues: list[Issue] = []
        # Walk subgraph initializers too -- an INT64/DOUBLE weight buried in an
        # If/Loop/Scan body is just as much of a conversion problem.
        for init, _owner in iter_initializers(graph):
            mapping = _INIT_DTYPES.get(init.data_type)
            if mapping is None:
                continue
            issue_id, token = mapping
            issues.append(
                remediation.make_issue(
                    issue_id,
                    node_name=init.name,
                    operator="Initializer",
                    prefix=f"Initializer '{init.name}' has dtype {token}",
                )
            )
        return issues

    def _check_internal_double(self, graph: onnx.GraphProto) -> list[Issue]:
        """Catch FLOAT64 introduced *inside* the graph, not just at the boundary.

        ``_check_inputs`` / ``_check_initializers`` only see the graph's edges.
        A ``Cast(to=DOUBLE)`` or a ``Constant`` holding a DOUBLE tensor injects
        double precision into intermediate values, which TensorRT cannot
        represent anywhere. Scanned across subgraphs too.
        """
        issues: list[Issue] = []
        for node, _owner in iter_nodes(graph):
            if node.op_type == "Cast":
                to_attr = next((a for a in node.attribute if a.name == "to"), None)
                if to_attr is not None and to_attr.i == TensorProto.DOUBLE:
                    issues.append(
                        remediation.make_issue(
                            "float64_tensors",
                            node_name=node.name or "<unnamed Cast>",
                            operator="Cast",
                            prefix="Cast targets dtype DOUBLE",
                        )
                    )
            elif node.op_type == "Constant":
                for attr in node.attribute:
                    if (
                        attr.name == "value"
                        and attr.type == onnx.AttributeProto.TENSOR
                        and attr.t.data_type == TensorProto.DOUBLE
                    ):
                        issues.append(
                            remediation.make_issue(
                                "float64_tensors",
                                node_name=node.name or "<unnamed Constant>",
                                operator="Constant",
                                prefix="Constant holds a DOUBLE tensor",
                            )
                        )
        return issues
