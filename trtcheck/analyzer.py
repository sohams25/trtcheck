"""Top-level analyzer that composes every checker into a single AnalysisReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import onnx

from trtcheck._graph import count_nodes
from trtcheck.checkers import Checker
from trtcheck.checkers.control_flow import ControlFlowChecker
from trtcheck.checkers.dynamic_shapes import DynamicShapeChecker
from trtcheck.checkers.graph_structure import GraphStructureChecker
from trtcheck.checkers.operator_support import OperatorSupportChecker
from trtcheck.checkers.precision import PrecisionChecker
from trtcheck.types import AnalysisReport, CheckCategory, Confidence, Issue, Severity

# Names of the checkers trtcheck ships. A crash in one of these is a bug in
# trtcheck and must surface loudly; only *third-party plugin* checkers are
# isolated behind a try/except so one bad plugin can't kill the whole report.
_BUILTIN_CHECKER_NAMES = frozenset(
    {"graph_structure", "precision", "operator_support", "dynamic_shapes", "control_flow"}
)


def safe_load(path: Path | str) -> onnx.ModelProto:
    """``onnx.load`` with parse failures surfaced as a clean ``ValueError``.

    A corrupt, truncated, or non-ONNX file makes ``onnx.load`` raise a raw
    protobuf ``DecodeError`` (and other low-level errors). trtcheck is meant to
    run on untrusted models in CI, so every load goes through here and turns
    those into a single domain error the CLI can present without a traceback.
    """
    try:
        model: onnx.ModelProto = onnx.load(str(path))
    except Exception as exc:  # protobuf DecodeError, OSError, onnx errors, ...
        # repr() the path so a crafted filename can't inject terminal escapes
        # into the error message.
        raise ValueError(f"could not parse {str(path)!r} as an ONNX model: {exc}") from exc
    return model


@dataclass
class AnalyzerConfig:
    target_trt: str = "10.3"
    matrix_path: Path | str | None = None  # for tests / custom matrices
    max_model_size_mb: int = 500  # refuse to load files larger than this
    discover_entry_point_plugins: bool = True
    disable_plugins: list[str] = field(default_factory=list)
    # Custom ONNX domains the user declares as backed by an installed
    # TensorRT plugin. Ops in these domains stop producing
    # TRT-OP-CUSTOM-DOMAIN unverified findings.
    plugin_domains: list[str] = field(default_factory=list)


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
                plugin_domains=self.config.plugin_domains,
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
        model = safe_load(path)
        return self.analyze_model(model, filename=str(path))

    def analyze_model(
        self, model: onnx.ModelProto, *, filename: str = "<in-memory>"
    ) -> AnalysisReport:
        all_issues: list[Issue] = []
        for checker in self.checkers:
            name = getattr(checker, "name", checker.__class__.__name__)
            if name in _BUILTIN_CHECKER_NAMES:
                # Built-in: let exceptions propagate. Masking a crash here as a
                # fake WARNING would both mislabel a trtcheck bug as a plugin
                # issue and hide it from tests.
                all_issues.extend(checker.check(model))
                continue
            try:
                all_issues.extend(checker.check(model))
            except Exception as exc:
                all_issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        category=CheckCategory.OPERATOR_SUPPORT,
                        node_name=f"<plugin: {name}>",
                        operator="Plugin",
                        message=(f"plugin {name!r} raised {exc.__class__.__name__}: {exc}"),
                        remediation=("Disable the plugin with --disable-plugin or uninstall it."),
                        docs_link=None,
                        rule_id="TRT-PLUGIN-CHECKER-ERROR",
                        confidence=Confidence.LOW,
                        verify_required=True,
                    )
                )

        # Every finding in this report was made against the same TRT target;
        # stamp it on issues whose checker didn't (precision/graph checks are
        # target-independent but the report they land in is not).
        for issue in all_issues:
            if issue.target_trt is None:
                issue.target_trt = self.config.target_trt

        # Sort: critical first, then warning, then info. Stable.
        all_issues.sort(key=lambda i: Severity.rank(i.severity))

        opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0
        )
        report = AnalysisReport(
            target_trt=self.config.target_trt,
            filename=filename,
            onnx_ir_version=str(model.ir_version),
            opset_version=opset,
            producer=model.producer_name or "unknown",
            # Count nodes in subgraphs too -- otherwise the header can report
            # "5 nodes" while flagging issues found inside If/Loop/Scan bodies.
            total_nodes=count_nodes(model.graph),
            issues=all_issues,
        )
        return report


def analyze(path: Path | str, *, config: AnalyzerConfig | None = None) -> AnalysisReport:
    """Convenience entry point: build an Analyzer and run it against `path`."""
    return Analyzer(config).analyze_path(path)
