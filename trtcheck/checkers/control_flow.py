"""Control-flow checks for Loop, If, and Scan nodes.

The TRT support story for these ops has a few well-known sharp edges:

  - Loop: trip count must be a graph-internal constant or shape-inferable.
    Runtime tensor inputs as trip count are silently rejected at engine
    build time.
  - Loop: nested loops are not supported at any TRT version.
  - If: both branches must produce identically-shaped, identically-typed
    outputs. We don't fully verify this here (shape inference would be
    required); we flag every If as a warning so the user knows to check.
  - Scan: sequence length must be statically known.
"""

from __future__ import annotations

import onnx

from trtcheck._graph import iter_initializers, iter_nodes
from trtcheck.types import CheckCategory, Issue, Severity


class ControlFlowChecker:
    name = "control_flow"

    def __init__(self, target_trt: str = "10.3") -> None:
        self._target = target_trt

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        # A control-flow op nested inside another subgraph still has to convert,
        # so walk every subgraph, not just the top level. Trip counts may be
        # defined in any enclosing scope, so treat the union of all initializer
        # names as "statically known" -- over-approximating here avoids false
        # "runtime trip count" warnings for outer-scope constants.
        initializer_names = {init.name for init, _ in iter_initializers(model.graph)}
        for node, _graph in iter_nodes(model.graph):
            if node.op_type == "Loop":
                issues.extend(self._check_loop(node, initializer_names))
            elif node.op_type == "If":
                issues.extend(self._check_if(node))
            elif node.op_type == "Scan":
                issues.append(self._scan_warning(node))
        return issues

    # -- Loop --------------------------------------------------------------

    def _check_loop(
        self,
        node: onnx.NodeProto,
        initializer_names: set[str],
    ) -> list[Issue]:
        issues: list[Issue] = []

        # Inputs to Loop: [M (trip count), cond, ...loop_state]
        # If the trip count input is not an initializer (i.e., it's a runtime
        # tensor or computed) we flag it.
        trip_input = node.input[0] if len(node.input) > 0 else ""
        if trip_input and trip_input not in initializer_names:
            issues.append(
                Issue(
                    severity=Severity.WARNING,
                    category=CheckCategory.CONTROL_FLOW,
                    node_name=node.name or "<Loop>",
                    operator="Loop",
                    message=(
                        f"Loop '{node.name}' uses input '{trip_input}' as its trip "
                        "count. TensorRT requires a statically-known or "
                        "shape-inferable trip count."
                    ),
                    remediation=(
                        "Refactor the loop to use a fixed iteration count, or "
                        "fully unroll the loop at export time."
                    ),
                    docs_link=(
                        "https://docs.nvidia.com/deeplearning/tensorrt/"
                        "developer-guide/index.html#loops"
                    ),
                )
            )

        # Nested loops: scan the body subgraph for another Loop.
        body_subgraph = _body_subgraph(node)
        if body_subgraph is not None and _contains_op(body_subgraph, "Loop"):
            issues.append(
                Issue(
                    severity=Severity.CRITICAL,
                    category=CheckCategory.CONTROL_FLOW,
                    node_name=node.name or "<Loop>",
                    operator="Loop",
                    message=(
                        f"Loop '{node.name}' contains a nested Loop in its body. "
                        "TensorRT does not support Loop-within-Loop."
                    ),
                    remediation=(
                        "Restructure the model to use a single Loop with combined "
                        "trip count, or fully unroll the inner iterations."
                    ),
                    docs_link=None,
                )
            )
        return issues

    # -- If ---------------------------------------------------------------

    def _check_if(self, node: onnx.NodeProto) -> list[Issue]:
        return [
            Issue(
                severity=Severity.WARNING,
                category=CheckCategory.CONTROL_FLOW,
                node_name=node.name or "<If>",
                operator="If",
                message=(
                    f"If '{node.name}' detected. TensorRT requires both branches "
                    "to produce identically-shaped, identically-typed tensors."
                ),
                remediation=(
                    "If branches diverge in output shape, pad or broadcast the "
                    "smaller branch to match before the If output."
                ),
                docs_link=None,
            )
        ]

    # -- Scan -------------------------------------------------------------

    def _scan_warning(self, node: onnx.NodeProto) -> Issue:
        return Issue(
            severity=Severity.WARNING,
            category=CheckCategory.CONTROL_FLOW,
            node_name=node.name or "<Scan>",
            operator="Scan",
            message=(
                f"Scan '{node.name}' detected. TensorRT requires the scan "
                "sequence length to be known at engine build time."
            ),
            remediation=(
                "Fix the scan length at export, or replace Scan with a Loop "
                "over a constant range."
            ),
            docs_link=None,
        )


# -- helpers --------------------------------------------------------------


def _body_subgraph(node: onnx.NodeProto) -> onnx.GraphProto | None:
    for attr in node.attribute:
        if attr.name == "body" and attr.type == onnx.AttributeProto.GRAPH:
            return attr.g  # type: ignore[no-any-return]
    return None


def _contains_op(graph: onnx.GraphProto, op_type: str) -> bool:
    # iter_nodes descends through every nested subgraph with a depth bound, so
    # this can't blow the stack on a pathologically nested model.
    return any(node.op_type == op_type for node, _ in iter_nodes(graph))
