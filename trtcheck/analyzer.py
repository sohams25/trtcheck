"""Top-level analyzer that composes every checker into a single AnalysisReport."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

from trtcheck.checkers import Checker
from trtcheck.checkers.control_flow import ControlFlowChecker
from trtcheck.checkers.dynamic_shapes import DynamicShapeChecker
from trtcheck.checkers.graph_structure import GraphStructureChecker
from trtcheck.checkers.operator_support import OperatorSupportChecker
from trtcheck.checkers.precision import PrecisionChecker
from trtcheck.types import AnalysisReport, Severity


@dataclass
class AnalyzerConfig:
    target_trt: str = "10.3"
    matrix_path: Path | str | None = None  # for tests / custom matrices


class Analyzer:
    """Runs the full checker stack against a loaded ONNX model."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self.checkers: list[Checker] = self._build_checkers()

    def _build_checkers(self) -> list[Checker]:
        return [
            GraphStructureChecker(),
            PrecisionChecker(),
            OperatorSupportChecker(
                matrix_path=self.config.matrix_path,
                target_trt=self.config.target_trt,
            ),
            DynamicShapeChecker(),
            ControlFlowChecker(target_trt=self.config.target_trt),
        ]

    def analyze_path(self, path: Path | str) -> AnalysisReport:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX file not found: {path}")
        model = onnx.load(str(path))
        return self.analyze_model(model, filename=str(path))

    def analyze_model(
        self, model: onnx.ModelProto, *, filename: str = "<in-memory>"
    ) -> AnalysisReport:
        all_issues = []
        for checker in self.checkers:
            all_issues.extend(checker.check(model))

        # Sort: critical first, then warning, then info. Stable.
        all_issues.sort(key=lambda i: Severity.rank(i.severity))

        opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0
        )
        report = AnalysisReport(
            filename=filename,
            onnx_ir_version=str(model.ir_version),
            opset_version=opset,
            producer=model.producer_name or "unknown",
            total_nodes=len(model.graph.node),
            issues=all_issues,
        )
        return report


def analyze(path: Path | str, *, config: AnalyzerConfig | None = None) -> AnalysisReport:
    """Convenience entry point: build an Analyzer and run it against `path`."""
    return Analyzer(config).analyze_path(path)
