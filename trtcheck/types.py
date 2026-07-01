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

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "node_name": self.node_name,
            "operator": self.operator,
            "message": self.message,
            "remediation": self.remediation,
            "docs_link": self.docs_link,
        }


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
    def conversion_likely(self) -> bool:
        return self.critical_count == 0

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
            "filename": self.filename,
            "onnx_ir_version": self.onnx_ir_version,
            "opset_version": self.opset_version,
            "producer": self.producer,
            "total_nodes": self.total_nodes,
            "issues": [i.to_dict() for i in self.issues],
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "conversion_likely": self.conversion_likely,
            "estimated_fix_time": self.estimated_fix_time,
        }
