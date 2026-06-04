"""Precision and dtype checks.

The classics TensorRT rejects or coerces:
  - UINT8 graph inputs (TRT accepts FP32/FP16/INT32/INT8 only)
  - FLOAT64 anywhere (TRT has no double precision)
  - String tensors (no TRT type)
  - BFLOAT16 (Ampere + TRT 8.6+ only)
  - INT64 weights/initializers (TRT casts to INT32, may overflow)
"""

from __future__ import annotations

import onnx
from onnx import TensorProto

from trtcheck._graph import iter_initializers, iter_nodes
from trtcheck.types import CheckCategory, Issue, Severity

_REMEDIATION_INT64 = (
    "Cast indices to int32 before exporting: idx = idx.to(torch.int32). "
    "For embedding lookups, ensure the indices tensor is int32."
)
_REMEDIATION_UINT8 = (
    "Move the UINT8 -> FLOAT32 conversion (and normalization) into your "
    "preprocessing pipeline rather than the model body."
)
_REMEDIATION_DOUBLE = (
    "Call model = model.float() before torch.onnx.export, then verify with "
    "onnx.checker that no FLOAT64 tensors remain."
)
_REMEDIATION_STRING = (
    "Move string preprocessing (tokenization, label encoding) out of the "
    "model and pass integer IDs at the boundary instead."
)
_REMEDIATION_BF16 = (
    "If targeting TRT < 8.6 or pre-Ampere GPUs, export the model in FP16 " "rather than BF16."
)


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
            dtype = inp.type.tensor_type.elem_type
            if dtype == TensorProto.UINT8:
                issues.append(
                    self._issue(
                        Severity.CRITICAL,
                        inp.name,
                        "Input",
                        f"Input '{inp.name}' has dtype UINT8; TensorRT accepts only "
                        "FP32, FP16, INT32, or INT8 as graph inputs.",
                        _REMEDIATION_UINT8,
                    )
                )
            elif dtype == TensorProto.DOUBLE:
                issues.append(
                    self._issue(
                        Severity.CRITICAL,
                        inp.name,
                        "Input",
                        f"Input '{inp.name}' has dtype DOUBLE (FLOAT64); TensorRT "
                        "has no double-precision support.",
                        _REMEDIATION_DOUBLE,
                    )
                )
            elif dtype == TensorProto.STRING:
                issues.append(
                    self._issue(
                        Severity.CRITICAL,
                        inp.name,
                        "Input",
                        f"Input '{inp.name}' has dtype STRING; TensorRT has no " "string type.",
                        _REMEDIATION_STRING,
                    )
                )
            elif dtype == TensorProto.INT64:
                # INT64 network inputs are the single most common real TRT input
                # problem (token-id / index inputs from NLP & embedding models).
                # TRT inputs cannot be INT64; the parser casts to INT32, which can
                # overflow. WARNING (not CRITICAL) to match the INT64-initializer
                # severity and avoid flipping the verdict on the very common,
                # usually-convertible token-id pattern.
                issues.append(
                    self._issue(
                        Severity.WARNING,
                        inp.name,
                        "Input",
                        f"Input '{inp.name}' has dtype INT64; TensorRT casts graph "
                        "inputs to INT32, which can overflow for large index values.",
                        _REMEDIATION_INT64,
                    )
                )
            elif dtype == TensorProto.BFLOAT16:
                issues.append(
                    self._issue(
                        Severity.WARNING,
                        inp.name,
                        "Input",
                        f"Input '{inp.name}' has dtype BFLOAT16; supported only "
                        "on TRT 8.6+ with Ampere or newer GPUs.",
                        _REMEDIATION_BF16,
                    )
                )
        return issues

    def _check_initializers(self, graph: onnx.GraphProto) -> list[Issue]:
        issues: list[Issue] = []
        # Walk subgraph initializers too -- an INT64/DOUBLE weight buried in an
        # If/Loop/Scan body is just as much of a conversion problem.
        for init, _owner in iter_initializers(graph):
            dtype = init.data_type
            if dtype == TensorProto.INT64:
                issues.append(
                    self._issue(
                        Severity.WARNING,
                        init.name,
                        "Initializer",
                        f"Initializer '{init.name}' has dtype INT64; TensorRT will "
                        "cast to INT32, which can overflow for large indices.",
                        _REMEDIATION_INT64,
                    )
                )
            elif dtype == TensorProto.DOUBLE:
                issues.append(
                    self._issue(
                        Severity.CRITICAL,
                        init.name,
                        "Initializer",
                        f"Initializer '{init.name}' has dtype DOUBLE (FLOAT64); "
                        "TensorRT has no double-precision support.",
                        _REMEDIATION_DOUBLE,
                    )
                )
            elif dtype == TensorProto.BFLOAT16:
                issues.append(
                    self._issue(
                        Severity.WARNING,
                        init.name,
                        "Initializer",
                        f"Initializer '{init.name}' has dtype BFLOAT16; supported "
                        "only on TRT 8.6+ with Ampere or newer GPUs.",
                        _REMEDIATION_BF16,
                    )
                )
        return issues

    def _check_internal_double(self, graph: onnx.GraphProto) -> list[Issue]:
        """Catch FLOAT64 introduced *inside* the graph, not just at the boundary.

        ``_check_inputs`` / ``_check_initializers`` only see the graph's edges.
        A ``Cast(to=DOUBLE)`` or a ``Constant`` holding a DOUBLE tensor injects
        double precision into intermediate values, which TensorRT cannot
        represent anywhere -- exactly the failure mode the module docstring
        names ("FLOAT64 anywhere"). Scanned across subgraphs too.
        """
        issues: list[Issue] = []
        for node, _owner in iter_nodes(graph):
            if node.op_type == "Cast":
                to_attr = next((a for a in node.attribute if a.name == "to"), None)
                if to_attr is not None and to_attr.i == TensorProto.DOUBLE:
                    issues.append(
                        self._issue(
                            Severity.CRITICAL,
                            node.name or "<unnamed Cast>",
                            "Cast",
                            "Cast targets dtype DOUBLE (FLOAT64); TensorRT has no "
                            "double-precision support for intermediate tensors.",
                            _REMEDIATION_DOUBLE,
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
                            self._issue(
                                Severity.CRITICAL,
                                node.name or "<unnamed Constant>",
                                "Constant",
                                "Constant holds a DOUBLE (FLOAT64) tensor; TensorRT "
                                "has no double-precision support.",
                                _REMEDIATION_DOUBLE,
                            )
                        )
        return issues

    @staticmethod
    def _issue(
        severity: Severity, node_name: str, operator: str, message: str, remediation: str
    ) -> Issue:
        return Issue(
            severity=severity,
            category=CheckCategory.PRECISION,
            node_name=node_name,
            operator=operator,
            message=message,
            remediation=remediation,
            docs_link=None,
        )
