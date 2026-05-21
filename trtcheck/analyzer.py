"""Top-level analyzer that composes every checker into a single AnalysisReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import onnx

from trtcheck.checkers import Checker
from trtcheck.checkers.control_flow import ControlFlowChecker
from trtcheck.checkers.dynamic_shapes import DynamicShapeChecker
from trtcheck.checkers.graph_structure import GraphStructureChecker
from trtcheck.checkers.operator_support import OperatorSupportChecker
from trtcheck.checkers.precision import PrecisionChecker
from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity


@dataclass
class AnalyzerConfig:
    target_trt: str = "10.3"
    matrix_path: Path | str | None = None  # for tests / custom matrices
    max_model_size_mb: int = 500  # refuse to load files larger than this
    discover_entry_point_plugins: bool = True
    disable_plugins: list[str] = field(default_factory=list)


class Analyzer:
    """Runs the full checker stack against a loaded ONNX model."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self.checkers: list[Checker] = self._build_checkers()

    def _build_checkers(self) -> list[Checker]:
        built_in: list[Checker] = [
            GraphStructureChecker(),
            PrecisionChecker(),
            OperatorSupportChecker(
                matrix_path=self.config.matrix_path,
                target_trt=self.config.target_trt,
            ),
            DynamicShapeChecker(),
            ControlFlowChecker(target_trt=self.config.target_trt),
        ]
        if self.config.discover_entry_point_plugins:
            from trtcheck.plugins import load_plugins

            discovered, _, _ = load_plugins()
            built_in.extend(discovered)
        disabled = set(self.config.disable_plugins or [])
        return [c for c in built_in if getattr(c, "name", "") not in disabled]

    def analyze_path(self, path: Path | str) -> AnalysisReport:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX file not found: {path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_model_size_mb:
            raise ValueError(
                f"ONNX file is {size_mb:.1f} MB, above the "
                f"{self.config.max_model_size_mb} MB limit. "
                "Raise the limit via AnalyzerConfig(max_model_size_mb=...) "
                "or the --max-model-size CLI flag."
            )
        model = onnx.load(str(path))
        return self.analyze_model(model, filename=str(path))

    def analyze_model(
        self, model: onnx.ModelProto, *, filename: str = "<in-memory>"
    ) -> AnalysisReport:
        all_issues: list[Issue] = []
        for checker in self.checkers:
            try:
                all_issues.extend(checker.check(model))
            except Exception as exc:
                all_issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category=CheckCategory.OPERATOR_SUPPORT,
                        node_name=f"<plugin: {getattr(checker, 'name', checker.__class__.__name__)}>",
                        operator="Plugin",
                        message=(
                            f"plugin {getattr(checker, 'name', checker.__class__.__name__)!r} "
                            f"raised {exc.__class__.__name__}: {exc}"
                        ),
                        remediation=("Disable the plugin with --disable-plugin or uninstall it."),
                        docs_link=None,
                    )
                )

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
