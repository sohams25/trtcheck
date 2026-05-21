"""Tests for the high-level analyzer."""

from pathlib import Path

import onnx
import pytest

from trtcheck.analyzer import Analyzer, AnalyzerConfig, analyze
from trtcheck.types import AnalysisReport, Severity


class TestAnalyzer:
    def test_clean_model_yields_passing_report(self, fixture_dir: Path) -> None:
        report = analyze(fixture_dir / "clean_minimal.onnx")
        assert report.conversion_likely is True
        assert report.critical_count == 0

    def test_sequence_empty_yields_failing_report(self, fixture_dir: Path) -> None:
        report = analyze(fixture_dir / "failing" / "sequence_empty.onnx")
        assert report.conversion_likely is False
        assert report.critical_count >= 1

    def test_report_metadata_populated(self, fixture_dir: Path) -> None:
        report = analyze(fixture_dir / "clean_minimal.onnx")
        assert report.filename.endswith("clean_minimal.onnx")
        assert report.opset_version >= 17
        assert report.total_nodes >= 1
        assert isinstance(report.onnx_ir_version, str)
        assert report.producer  # at least non-empty

    def test_custom_target_trt_is_honored(self, fixture_dir: Path) -> None:
        # GroupNorm-like ops would fail on 8.0 but here we just check the
        # config plumbing reaches the operator checker.
        report = analyze(
            fixture_dir / "clean_minimal.onnx",
            config=AnalyzerConfig(target_trt="8.0"),
        )
        assert report.conversion_likely is True

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.onnx"
        with pytest.raises(FileNotFoundError):
            analyze(bogus)

    def test_combined_failure_modes_aggregate(self, fixture_dir: Path) -> None:
        # uint8_input model has UINT8 input (critical from precision).
        report = analyze(fixture_dir / "failing" / "uint8_input.onnx")
        assert report.critical_count >= 1
        # The report must be JSON-serializable
        as_dict = report.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["filename"].endswith("uint8_input.onnx")

    def test_analyzer_class_runs_all_checkers(self, fixture_dir: Path) -> None:
        analyzer = Analyzer(AnalyzerConfig())
        report = analyzer.analyze_path(fixture_dir / "clean_minimal.onnx")
        assert isinstance(report, AnalysisReport)
