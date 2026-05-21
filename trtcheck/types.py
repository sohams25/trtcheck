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
    """Aggregated output of running every checker against an ONNX model."""

    filename: str
    onnx_ir_version: str
    opset_version: int
    producer: str

    total_nodes: int
    issues: list[Issue] = field(default_factory=list)

    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    estimated_fusions: list[str] = field(default_factory=list)
    estimated_precision: dict[str, int] = field(default_factory=dict)

    conversion_likely: bool = False
    estimated_fix_time: str = ""

    def recompute_counts(self) -> None:
        self.critical_count = sum(1 for i in self.issues if i.severity is Severity.CRITICAL)
        self.warning_count = sum(1 for i in self.issues if i.severity is Severity.WARNING)
        self.info_count = sum(1 for i in self.issues if i.severity is Severity.INFO)

    def derive_verdict(self) -> None:
        """Populate `conversion_likely` and `estimated_fix_time` from current issues.

        Heuristic: any CRITICAL fails the verdict. Fix-time scales with the
        number of CRITICALs because each typically requires a targeted ONNX
        export change.
        """
        self.recompute_counts()
        self.conversion_likely = self.critical_count == 0

        if self.critical_count == 0:
            self.estimated_fix_time = "< 15 minutes" if self.warning_count else "no action needed"
        elif self.critical_count == 1:
            self.estimated_fix_time = "15-30 minutes"
        elif self.critical_count <= 3:
            self.estimated_fix_time = "1-2 hours"
        else:
            self.estimated_fix_time = "half a day or more"

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
            "estimated_fusions": list(self.estimated_fusions),
            "estimated_precision": dict(self.estimated_precision),
            "conversion_likely": self.conversion_likely,
            "estimated_fix_time": self.estimated_fix_time,
        }
