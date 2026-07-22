"""trtcheck -- static pre-flight checker for ONNX -> TensorRT conversion."""

from trtcheck.analyzer import Analyzer, AnalyzerConfig, analyze
from trtcheck.types import (
    REPORT_SCHEMA_VERSION,
    AnalysisReport,
    CheckCategory,
    Confidence,
    Issue,
    Severity,
    Verdict,
)

__version__ = "1.1.0"
__all__ = [
    "Analyzer",
    "AnalyzerConfig",
    "analyze",
    "AnalysisReport",
    "CheckCategory",
    "Confidence",
    "Issue",
    "Severity",
    "Verdict",
    "REPORT_SCHEMA_VERSION",
    "__version__",
]
