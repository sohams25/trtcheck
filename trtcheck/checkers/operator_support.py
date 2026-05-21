"""Operator-vs-TRT support matrix lookup.

For each node in the graph, find the operator in operator_matrix.json and
emit an Issue at the appropriate severity:

  - not_supported -> CRITICAL
  - partial       -> WARNING
  - supported     -> nothing
  - unknown       -> nothing (info-level, never blocking)

Custom-domain ops are silently skipped: the matrix only describes the
default ONNX domain, and emitting a finding for every custom plugin would
be noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx

from trtcheck.types import CheckCategory, Issue, Severity


def _default_matrix_path() -> Path:
    return Path(__file__).parent.parent / "data" / "operator_matrix.json"


class OperatorSupportChecker:
    name = "operator_support"

    def __init__(
        self,
        matrix_path: Path | str | None = None,
        target_trt: str = "10.3",
    ) -> None:
        path = Path(matrix_path) if matrix_path else _default_matrix_path()
        with open(path) as f:
            self._matrix: dict = json.load(f)
        valid = set(self._matrix.get("target_trt_versions", []))
        if target_trt not in valid:
            raise ValueError(
                f"target_trt={target_trt!r} not in matrix versions {sorted(valid)}"
            )
        self._target = target_trt
        self._ops: dict = self._matrix["operators"]

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        for node in model.graph.node:
            # Skip custom domains -- the matrix only describes "" / "ai.onnx".
            if node.domain and node.domain not in ("", "ai.onnx"):
                continue
            entry = self._ops.get(node.op_type)
            if entry is None:
                continue  # Unknown op -- don't blow up on third-party ops
            status = entry["support"].get(self._target, "unknown")
            if status == "not_supported":
                issues.append(self._make_issue(node, entry, Severity.CRITICAL))
            elif status == "partial":
                issues.append(self._make_issue(node, entry, Severity.WARNING))
        return issues

    def _make_issue(
        self,
        node: onnx.NodeProto,
        entry: dict,
        severity: Severity,
    ) -> Issue:
        op = node.op_type
        notes = entry.get("notes", "")
        if severity is Severity.CRITICAL:
            message = (
                f"Operator '{op}' is not supported in TensorRT {self._target}."
                + (f" {notes}" if notes else "")
            )
            remediation = entry.get("remediation") or (
                "Replace with an equivalent supported op, write a TRT plugin, "
                "or remove from the graph if it is dead."
            )
        else:
            limitations = entry.get("limitations", [])
            lim_str = f" Limitations: {'; '.join(limitations)}." if limitations else ""
            message = (
                f"Operator '{op}' has partial support in TensorRT {self._target}."
                + (f" {notes}" if notes else "")
                + lim_str
            )
            remediation = entry.get("remediation") or (
                "Check the operator-specific limitations and validate against your "
                "exported attribute set."
            )
        return Issue(
            severity=severity,
            category=CheckCategory.OPERATOR_SUPPORT,
            node_name=node.name or f"<unnamed {op}>",
            operator=op,
            message=message,
            remediation=remediation,
            docs_link=entry.get("github_issue"),
        )
