"""Operator-vs-TRT support matrix lookup.

For each node in the graph, find the operator in operator_matrix.json and
emit an Issue at the appropriate severity:

  - not_supported -> CRITICAL  (rule TRT-OP-UNSUPPORTED)
  - partial       -> WARNING   (rule TRT-OP-PARTIAL, needs verification)
  - supported     -> nothing, unless a conditional-support rule fires
  - unknown / absent from the matrix -> INFO (rule TRT-OP-UNCLASSIFIED,
    needs verification). An operator trtcheck has no evidence about must
    not silently pass as clean.

Custom-domain operators (rule TRT-OP-CUSTOM-DOMAIN) always need a TensorRT
plugin; they are reported as unverified findings unless the caller
explicitly declares the domain as plugin-backed via ``plugin_domains``.

Conditional support (schema 2.x matrices) is expressed per operator as a
``conditions`` list. Each condition either passes, fires a violation
(rule TRT-OP-CONDITION), or cannot be resolved statically and produces an
unverified finding (rule TRT-OP-CONDITION-UNRESOLVED).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import onnx

from trtcheck._graph import iter_nodes, iter_subgraphs
from trtcheck.types import CheckCategory, Confidence, Issue, Severity

# Stable rule ids this checker owns (the remediation DB owns the rest).
RULE_UNSUPPORTED = "TRT-OP-UNSUPPORTED"
RULE_PARTIAL = "TRT-OP-PARTIAL"
RULE_UNCLASSIFIED = "TRT-OP-UNCLASSIFIED"
RULE_CUSTOM_DOMAIN = "TRT-OP-CUSTOM-DOMAIN"
RULE_CONDITION = "TRT-OP-CONDITION"
RULE_CONDITION_UNRESOLVED = "TRT-OP-CONDITION-UNRESOLVED"

_DEFAULT_DOMAINS = ("", "ai.onnx")


def _default_matrix_path() -> Path:
    return Path(__file__).parent.parent / "data" / "operator_matrix.json"


class OperatorSupportChecker:
    name = "operator_support"

    def __init__(
        self,
        matrix_path: Path | str | None = None,
        target_trt: str = "10.3",
        plugin_domains: Iterable[str] = (),
    ) -> None:
        path = Path(matrix_path) if matrix_path else _default_matrix_path()
        with open(path) as f:
            self._matrix: dict[str, Any] = json.load(f)
        valid = set(self._matrix.get("target_trt_versions", []))
        if target_trt not in valid:
            raise ValueError(f"target_trt={target_trt!r} not in matrix versions {sorted(valid)}")
        self._target = target_trt
        self._ops: dict[str, dict[str, Any]] = self._matrix["operators"]
        self._plugin_domains = set(plugin_domains)

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        # Scalar-int constants (initializers + Constant nodes) across every
        # scope, used to evaluate constant-input conditions statically.
        constants = _collect_scalar_constants(model)
        # Aggregated uncertainty findings: one per distinct op_type (or
        # domain/op_type pair), not one per node -- a transformer with 400
        # unclassified nodes of the same op should read as one finding.
        unclassified: Counter[str] = Counter()
        custom: Counter[tuple[str, str]] = Counter()

        # Walk the top-level graph AND every If/Loop/Scan subgraph body: an
        # unsupported op inside a branch still blocks the TensorRT build.
        for node, graph in iter_nodes(model.graph):
            if node.domain and node.domain not in _DEFAULT_DOMAINS:
                if node.domain not in self._plugin_domains:
                    custom[(node.domain, node.op_type)] += 1
                continue
            entry = self._ops.get(node.op_type)
            if entry is None:
                unclassified[node.op_type] += 1
                continue
            status = entry["support"].get(self._target, "unknown")
            if status == "not_supported":
                issues.append(self._make_issue(node, entry, Severity.CRITICAL, graph.name))
            elif status == "partial":
                issues.append(self._make_issue(node, entry, Severity.WARNING, graph.name))
            elif status == "unknown":
                unclassified[node.op_type] += 1
            if status in ("supported", "partial"):
                issues.extend(self._check_conditions(node, entry, graph.name, constants))

        issues.extend(
            self._unclassified_issue(op, count) for op, count in sorted(unclassified.items())
        )
        issues.extend(
            self._custom_domain_issue(domain, op, count)
            for (domain, op), count in sorted(custom.items())
        )
        return issues

    # -- support-status findings ------------------------------------------

    def _make_issue(
        self,
        node: onnx.NodeProto,
        entry: dict[str, Any],
        severity: Severity,
        graph_scope: str,
    ) -> Issue:
        op = node.op_type
        notes = entry.get("notes", "")
        if severity is Severity.CRITICAL:
            message = f"Operator '{op}' is not supported in TensorRT {self._target}." + (
                f" {notes}" if notes else ""
            )
            remediation = entry.get("remediation") or (
                "Replace with an equivalent supported op, write a TRT plugin, "
                "or remove from the graph if it is dead."
            )
            rule_id, confidence, verify = RULE_UNSUPPORTED, Confidence.HIGH, False
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
            rule_id, confidence, verify = RULE_PARTIAL, Confidence.MEDIUM, True
        return Issue(
            severity=severity,
            category=CheckCategory.OPERATOR_SUPPORT,
            node_name=node.name or f"<unnamed {op}>",
            operator=op,
            message=message,
            remediation=remediation,
            docs_link=entry.get("github_issue"),
            rule_id=rule_id,
            confidence=confidence,
            verify_required=verify,
            target_trt=self._target,
            graph_scope=graph_scope,
        )

    def _unclassified_issue(self, op: str, count: int) -> Issue:
        plural = "s" if count != 1 else ""
        return Issue(
            severity=Severity.INFO,
            category=CheckCategory.OPERATOR_SUPPORT,
            node_name=f"<{count} node{plural}>",
            operator=op,
            message=(
                f"Operator '{op}' ({count} node{plural}) is not classified in "
                f"trtcheck's support matrix for TensorRT {self._target}. It may "
                "convert fine, but static analysis has no evidence either way."
            ),
            remediation=(
                "Verify with a real TensorRT parse (trtexec --onnx=model.onnx) "
                "or check the onnx-tensorrt supported-operators list for this op."
            ),
            docs_link="https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md",
            rule_id=RULE_UNCLASSIFIED,
            confidence=Confidence.LOW,
            verify_required=True,
            target_trt=self._target,
        )

    def _custom_domain_issue(self, domain: str, op: str, count: int) -> Issue:
        plural = "s" if count != 1 else ""
        return Issue(
            severity=Severity.INFO,
            category=CheckCategory.OPERATOR_SUPPORT,
            node_name=f"<{count} node{plural}>",
            operator=f"{domain}::{op}",
            message=(
                f"Operator '{op}' ({count} node{plural}) lives in custom domain "
                f"'{domain}'. TensorRT needs a plugin that implements it; trtcheck "
                "cannot verify plugin availability statically."
            ),
            remediation=(
                "If a TensorRT plugin for this domain is installed, declare it "
                "with --plugin-domain to suppress this finding; otherwise "
                "implement/register the plugin before converting."
            ),
            docs_link=(
                "https://docs.nvidia.com/deeplearning/tensorrt/latest/"
                "inference-library/extending-custom-layers.html"
            ),
            rule_id=RULE_CUSTOM_DOMAIN,
            confidence=Confidence.LOW,
            verify_required=True,
            target_trt=self._target,
        )

    # -- conditional support ----------------------------------------------

    def _check_conditions(
        self,
        node: onnx.NodeProto,
        entry: dict[str, Any],
        graph_scope: str,
        constants: dict[str, int | None],
    ) -> list[Issue]:
        issues: list[Issue] = []
        for cond in entry.get("conditions", []):
            applies = cond.get("applies_to")
            if applies is not None and self._target not in applies:
                continue
            kind = cond.get("kind")
            if kind == "attribute_allowed":
                verdict = _eval_attribute_allowed(node, cond)
            elif kind == "constant_input_max":
                verdict = _eval_constant_input_max(node, cond, constants)
            else:
                # Unknown condition kind: matrix is ahead of the code. Treat as
                # unresolvable rather than silently passing.
                verdict = "unresolved"
            if verdict == "pass":
                continue
            issues.append(self._condition_issue(node, entry, cond, verdict, graph_scope))
        return issues

    def _condition_issue(
        self,
        node: onnx.NodeProto,
        entry: dict[str, Any],
        cond: dict[str, Any],
        verdict: str,
        graph_scope: str,
    ) -> Issue:
        op = node.op_type
        evidence = cond.get("evidence", {})
        docs = evidence.get("source") or entry.get("github_issue")
        if verdict == "violated":
            severity = Severity(cond.get("severity", "warning"))
            message = (
                f"Operator '{op}' violates a TensorRT {self._target} support "
                f"condition: {cond.get('message', cond.get('id', 'condition'))}"
            )
            rule_id = RULE_CONDITION
            confidence = (
                Confidence.HIGH if evidence.get("status") == "official_docs" else Confidence.MEDIUM
            )
            verify = False
        else:  # unresolved
            severity = Severity.INFO
            message = (
                f"Operator '{op}' has a TensorRT {self._target} support condition "
                f"that static analysis cannot resolve: "
                f"{cond.get('message', cond.get('id', 'condition'))}"
            )
            rule_id = RULE_CONDITION_UNRESOLVED
            confidence = Confidence.LOW
            verify = True
        return Issue(
            severity=severity,
            category=CheckCategory.OPERATOR_SUPPORT,
            node_name=node.name or f"<unnamed {op}>",
            operator=op,
            message=message,
            remediation=cond.get("remediation")
            or "Verify with a real TensorRT parse (trtexec --onnx=model.onnx).",
            docs_link=docs,
            rule_id=rule_id,
            confidence=confidence,
            verify_required=verify,
            target_trt=self._target,
            graph_scope=graph_scope,
        )


def _collect_scalar_constants(model: onnx.ModelProto) -> dict[str, int | None]:
    """Map tensor name -> scalar int value for every single-element integer
    initializer or Constant node output in the model. Value is None when the
    tensor is constant but not a readable scalar int."""
    from onnx import numpy_helper

    out: dict[str, int | None] = {}
    for graph in iter_subgraphs(model.graph):
        for init in graph.initializer:
            out[init.name] = _scalar_int(numpy_helper, init)
        for node in graph.node:
            if node.op_type == "Constant" and node.output:
                tensor = next(
                    (a.t for a in node.attribute if a.name == "value" and a.HasField("t")),
                    None,
                )
                out[node.output[0]] = _scalar_int(numpy_helper, tensor) if tensor else None
    return out


def _scalar_int(numpy_helper: Any, tensor: onnx.TensorProto) -> int | None:
    try:
        arr = numpy_helper.to_array(tensor)
    except Exception:
        return None
    if arr.size != 1 or arr.dtype.kind not in ("i", "u"):
        return None
    return int(arr.reshape(()))


def _eval_attribute_allowed(node: onnx.NodeProto, cond: dict[str, Any]) -> str:
    """'pass' | 'violated'. Checks a node attribute against an allowed set."""
    attr_name = cond["attribute"]
    allowed = cond["allowed_values"]
    attr = next((a for a in node.attribute if a.name == attr_name), None)
    if attr is None:
        return "pass" if cond.get("default_ok", True) else "violated"
    if attr.type == onnx.AttributeProto.INT:
        return "pass" if attr.i in allowed else "violated"
    if attr.type == onnx.AttributeProto.STRING:
        return "pass" if attr.s.decode(errors="replace") in allowed else "violated"
    # Attribute exists but has a type this condition can't compare: be honest.
    return "unresolved"


def _eval_constant_input_max(
    node: onnx.NodeProto, cond: dict[str, Any], constants: dict[str, int | None]
) -> str:
    """'pass' | 'violated' | 'unresolved'.

    The input at ``input_index`` must, when statically constant, hold a scalar
    integer <= ``max_value``. A non-constant (runtime) input cannot be checked
    statically -> unresolved.
    """
    idx = cond["input_index"]
    if idx >= len(node.input) or not node.input[idx]:
        # Optional input absent: nothing to violate.
        return "pass"
    name = node.input[idx]
    if name not in constants:
        return "unresolved"
    value = constants[name]
    if value is None:
        return "unresolved"
    max_value = cond.get("max_value")
    if max_value is not None and value > max_value:
        return "violated"
    return "pass"
