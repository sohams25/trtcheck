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
    "If targeting TRT < 8.6 or pre-Ampere GPUs, export the model in FP16 "
    "rather than BF16."
)


class PrecisionChecker:
    name = "precision"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        issues.extend(self._check_inputs(model.graph))
        issues.extend(self._check_initializers(model.graph))
        return issues

    def _check_inputs(self, graph: onnx.GraphProto) -> list[Issue]:
        issues: list[Issue] = []
        for inp in graph.input:
            dtype = inp.type.tensor_type.elem_type
            if dtype == TensorProto.UINT8:
                issues.append(self._issue(
                    Severity.CRITICAL, inp.name, "Input",
                    f"Input '{inp.name}' has dtype UINT8; TensorRT accepts only "
                    "FP32, FP16, INT32, or INT8 as graph inputs.",
                    _REMEDIATION_UINT8,
                ))
            elif dtype == TensorProto.DOUBLE:
                issues.append(self._issue(
                    Severity.CRITICAL, inp.name, "Input",
                    f"Input '{inp.name}' has dtype DOUBLE (FLOAT64); TensorRT "
                    "has no double-precision support.",
                    _REMEDIATION_DOUBLE,
                ))
            elif dtype == TensorProto.STRING:
                issues.append(self._issue(
                    Severity.CRITICAL, inp.name, "Input",
                    f"Input '{inp.name}' has dtype STRING; TensorRT has no "
                    "string type.",
                    _REMEDIATION_STRING,
                ))
            elif dtype == TensorProto.BFLOAT16:
                issues.append(self._issue(
                    Severity.WARNING, inp.name, "Input",
                    f"Input '{inp.name}' has dtype BFLOAT16; supported only "
                    "on TRT 8.6+ with Ampere or newer GPUs.",
                    _REMEDIATION_BF16,
                ))
        return issues

    def _check_initializers(self, graph: onnx.GraphProto) -> list[Issue]:
        issues: list[Issue] = []
        for init in graph.initializer:
            dtype = init.data_type
            if dtype == TensorProto.INT64:
                issues.append(self._issue(
                    Severity.WARNING, init.name, "Initializer",
                    f"Initializer '{init.name}' has dtype INT64; TensorRT will "
                    "cast to INT32, which can overflow for large indices.",
                    _REMEDIATION_INT64,
                ))
            elif dtype == TensorProto.DOUBLE:
                issues.append(self._issue(
                    Severity.CRITICAL, init.name, "Initializer",
                    f"Initializer '{init.name}' has dtype DOUBLE (FLOAT64); "
                    "TensorRT has no double-precision support.",
                    _REMEDIATION_DOUBLE,
                ))
            elif dtype == TensorProto.BFLOAT16:
                issues.append(self._issue(
                    Severity.WARNING, init.name, "Initializer",
                    f"Initializer '{init.name}' has dtype BFLOAT16; supported "
                    "only on TRT 8.6+ with Ampere or newer GPUs.",
                    _REMEDIATION_BF16,
                ))
        return issues

    @staticmethod
    def _issue(severity: Severity, node_name: str, operator: str,
               message: str, remediation: str) -> Issue:
        return Issue(
            severity=severity,
            category=CheckCategory.PRECISION,
            node_name=node_name,
            operator=operator,
            message=message,
            remediation=remediation,
            docs_link=None,
        )
