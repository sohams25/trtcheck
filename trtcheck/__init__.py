"""trtcheck -- static pre-flight checker for ONNX -> TensorRT conversion."""

from trtcheck.analyzer import Analyzer, AnalyzerConfig, analyze
from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity

__version__ = "1.0.0"
__all__ = [
    "Analyzer",
    "AnalyzerConfig",
    "analyze",
    "AnalysisReport",
    "CheckCategory",
    "Issue",
    "Severity",
    "__version__",
]
