"""Smoke tests for the trtcheck CLI."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from trtcheck.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCLI:
    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "trtcheck" in result.output.lower() or "0.1.0" in result.output

    def test_clean_model_exits_zero(self, runner: CliRunner, fixture_dir: Path) -> None:
        result = runner.invoke(main, [str(fixture_dir / "clean_minimal.onnx")])
        assert result.exit_code == 0, result.output

    def test_failing_model_exits_nonzero(self, runner: CliRunner, fixture_dir: Path) -> None:
        result = runner.invoke(main, [str(fixture_dir / "failing" / "sequence_empty.onnx")])
        assert result.exit_code != 0
        assert "SequenceEmpty" in result.output

    def test_json_format_outputs_parseable_json(
        self, runner: CliRunner, fixture_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        result = runner.invoke(
            main,
            [
                str(fixture_dir / "clean_minimal.onnx"),
                "--format",
                "json",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(out.read_text())
        assert payload["filename"].endswith("clean_minimal.onnx")

    def test_html_format_writes_html_file(
        self, runner: CliRunner, fixture_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        result = runner.invoke(
            main,
            [
                str(fixture_dir / "clean_minimal.onnx"),
                "--format",
                "html",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        body = out.read_text().lower()
        assert "<!doctype html" in body

    def test_severity_filter_hides_warnings(self, runner: CliRunner, fixture_dir: Path) -> None:
        # uint8_input has at least one critical, at least one warning maybe.
        result_all = runner.invoke(
            main,
            [str(fixture_dir / "failing" / "uint8_input.onnx"), "--format", "json"],
        )
        payload_all = json.loads(result_all.output)
        result_crit = runner.invoke(
            main,
            [
                str(fixture_dir / "failing" / "uint8_input.onnx"),
                "--format",
                "json",
                "--severity",
                "critical",
            ],
        )
        payload_crit = json.loads(result_crit.output)
        non_critical_in_all = [i for i in payload_all["issues"] if i["severity"] != "critical"]
        assert all(i["severity"] == "critical" for i in payload_crit["issues"])
        # Sanity: filter actually filters at least one issue (only meaningful
        # if there were any non-criticals in the unfiltered output).
        if non_critical_in_all:
            assert len(payload_crit["issues"]) < len(payload_all["issues"])

    def test_target_trt_flag_accepts_known_version(
        self, runner: CliRunner, fixture_dir: Path
    ) -> None:
        result = runner.invoke(
            main,
            [str(fixture_dir / "clean_minimal.onnx"), "--target-trt", "8.6"],
        )
        assert result.exit_code == 0

    def test_target_trt_unknown_version_errors(self, runner: CliRunner, fixture_dir: Path) -> None:
        result = runner.invoke(
            main,
            [str(fixture_dir / "clean_minimal.onnx"), "--target-trt", "99.9"],
        )
        assert result.exit_code != 0

    def test_diff_mode_runs_against_two_files(
        self, runner: CliRunner, fixture_dir: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            main,
            [
                str(fixture_dir / "clean_minimal.onnx"),
                str(fixture_dir / "failing" / "sequence_empty.onnx"),
                "--diff",
                "--format",
                "json",
            ],
        )
        # Exit code is non-zero because at least one of the two failed.
        # Output should be a JSON object with both reports.
        payload = json.loads(result.output)
        assert "before" in payload and "after" in payload

    def test_missing_file_errors_cleanly(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, [str(tmp_path / "nope.onnx")])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "no such" in result.output.lower()
