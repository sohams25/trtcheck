"""Core dataclasses and enums shared by the analyzer, checkers, and reporters.

Kept deliberately free of `onnx` imports so that reporter and CLI code can pull
these types without dragging the protobuf stack along.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Issue severity, ordered worst-first when ranked."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    @staticmethod
    def rank(value: "Severity") -> int:
        # Lower rank = more severe. Used as a sort key.
        return {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}[value]


class CheckCategory(str, Enum):
    OPERATOR_SUPPORT = "operator_support"
    PRECISION = "precision"
    DYNAMIC_SHAPES = "dynamic_shapes"
    CONTROL_FLOW = "control_flow"
    GRAPH_STRUCTURE = "graph_structure"


class Confidence(str, Enum):
    """How much evidence backs a finding.

    HIGH   -- documented or empirically verified behavior; acting on the
              finding is safe.
    MEDIUM -- a static heuristic with known gaps (e.g. partial-support
              limitations that depend on exported attributes).
    LOW    -- an uncertainty marker: trtcheck cannot classify the construct
              statically and is saying so rather than guessing.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(str, Enum):
    """Four-state conversion verdict. Replaces the old boolean
    ``conversion_likely``.

    BLOCKED    -- at least one known-critical incompatibility.
    UNVERIFIED -- no known blocker, but unresolved conditions remain
                  (unclassified operators, custom domains, conditional
                  support that static analysis cannot settle).
    LIKELY     -- every static check passed with nothing unresolved. This
                  is still a static prediction, not a guarantee.
    VERIFIED   -- an optional real TensorRT (trtexec) parse/build succeeded
                  for the declared environment.
    """

    BLOCKED = "blocked"
    UNVERIFIED = "unverified"
    LIKELY = "likely"
    VERIFIED = "verified"


# JSON report schema version. Bump the major when a field is removed or
# changes meaning; bump the minor when fields are added. Consumers of the
# 1.x schema keep working: every 1.x key is still present in 2.0.
REPORT_SCHEMA_VERSION = "2.0"


@dataclass
class Issue:
    """A single finding from a checker."""

    severity: Severity
    category: CheckCategory
    node_name: str
    operator: str
    message: str
    remediation: str
    docs_link: str | None = None
    # Stable machine-readable identity/metadata (schema 2.0). Defaults keep
    # the constructor backward compatible for third-party checkers.
    rule_id: str = ""
    confidence: Confidence = Confidence.HIGH
    verify_required: bool = False
    target_trt: str | None = None
    graph_scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "node_name": self.node_name,
            "operator": self.operator,
            "message": self.message,
            "remediation": self.remediation,
            "docs_link": self.docs_link,
            "rule_id": self.rule_id,
            "confidence": self.confidence.value,
            "verify_required": self.verify_required,
            "target_trt": self.target_trt,
            "graph_scope": self.graph_scope,
        }

    def identity(self) -> tuple[str, str, str, str]:
        """Stable identity for diffing reports.

        Includes ``graph_scope`` so two same-named nodes in different
        subgraphs (legal ONNX: uniqueness is per-graph) never alias in a
        before/after comparison.
        """
        return (self.rule_id, self.node_name, self.operator, self.graph_scope)


@dataclass
class AnalysisReport:
    """Aggregated output of running every checker against an ONNX model.

    Counts and the verdict are derived from `issues` on access. Mutating
    `issues` is supported; downstream consumers see the recomputed values
    automatically.
    """

    filename: str
    onnx_ir_version: str
    opset_version: int
    producer: str

    total_nodes: int
    issues: list[Issue] = field(default_factory=list)
    # Which TensorRT version the operator-support checks targeted.
    target_trt: str | None = None
    # Set only by the optional runtime-verification path (trtexec parse/build
    # succeeded). Static analysis never sets this.
    runtime_verified: bool = False
    # Metadata from the runtime verification run (command, versions, status),
    # populated by the CLI when --verify-runtime is used.
    runtime_verification: dict[str, Any] | None = None

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity is Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity is Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity is Severity.INFO)

    @property
    def verdict(self) -> Verdict:
        """Four-state verdict derived from the findings (see :class:`Verdict`).

        Precedence: BLOCKED > VERIFIED > UNVERIFIED > LIKELY. A recorded
        runtime *failure* (parser or engine build) demotes an otherwise-LIKELY
        report to UNVERIFIED -- contradictory runtime evidence must never be
        hidden behind a clean static prediction. Verification that could not
        run (missing trtexec, timeout, spawn error) leaves the static verdict
        untouched; its metadata is still in ``runtime_verification``.
        """
        if self.critical_count > 0:
            return Verdict.BLOCKED
        if self.runtime_verified:
            return Verdict.VERIFIED
        if any(i.verify_required for i in self.issues):
            return Verdict.UNVERIFIED
        if self.runtime_verification is not None and self.runtime_verification.get("status") in (
            "parser_failure",
            "build_failure",
        ):
            return Verdict.UNVERIFIED
        return Verdict.LIKELY

    @property
    def conversion_likely(self) -> bool:
        """Deprecated boolean view of :attr:`verdict`.

        Kept for 1.x JSON consumers: True for every verdict except BLOCKED.
        Prefer ``verdict`` -- this property cannot express UNVERIFIED.
        """
        return self.verdict is not Verdict.BLOCKED

    @property
    def estimated_fix_time(self) -> str:
        """Rough human-readable fix-time estimate, scaled by critical count."""
        crit = self.critical_count
        if crit == 0:
            return "< 15 minutes" if self.warning_count else "no action needed"
        if crit == 1:
            return "15-30 minutes"
        if crit <= 3:
            return "1-2 hours"
        return "half a day or more"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "filename": self.filename,
            "onnx_ir_version": self.onnx_ir_version,
            "opset_version": self.opset_version,
            "producer": self.producer,
            "total_nodes": self.total_nodes,
            "target_trt": self.target_trt,
            "issues": [i.to_dict() for i in self.issues],
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "verdict": self.verdict.value,
            "runtime_verified": self.runtime_verified,
            "runtime_verification": self.runtime_verification,
            "conversion_likely": self.conversion_likely,
            "estimated_fix_time": self.estimated_fix_time,
        }
