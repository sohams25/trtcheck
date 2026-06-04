"""CLI robustness regressions: corrupt input, --diff --force, console file output, severity precedence."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from trtcheck.analyzer import safe_load
from trtcheck.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_corrupt_onnx_errors_cleanly_not_traceback(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.onnx"
    bad.write_bytes(b"this is definitely not an onnx protobuf")
    result = runner.invoke(main, [str(bad)])
    assert result.exit_code != 0
    assert "could not parse" in result.output.lower() or "onnx" in result.output.lower()
    # A clean ClickException, not a leaked traceback.
    assert "Traceback (most recent call last)" not in result.output


def test_corrupt_onnx_in_fix_errors_cleanly(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.onnx"
    bad.write_bytes(b"garbage")
    out = tmp_path / "fixed.onnx"
    result = runner.invoke(main, [str(bad), "--fix", "--output", str(out)])
    assert result.exit_code != 0
    assert "Traceback (most recent call last)" not in result.output


def test_safe_load_raises_valueerror_on_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "x.onnx"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        safe_load(bad)


def test_console_output_to_file_has_no_ansi_codes(
    runner: CliRunner, fixture_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "report.txt"
    result = runner.invoke(
        main,
        [
            str(fixture_dir / "failing" / "uint8_input.onnx"),
            "--format",
            "console",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code != 0  # uint8 is a critical
    content = out.read_text()
    assert "\x1b[" not in content, "console file output must be plain text, not raw ANSI"
    assert "UINT8" in content


def test_diff_force_overwrites_console_output(
    runner: CliRunner, fixture_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "diff.txt"
    out.write_text("placeholder")
    result = runner.invoke(
        main,
        [
            str(fixture_dir / "clean_minimal.onnx"),
            str(fixture_dir / "clean_minimal.onnx"),
            "--diff",
            "--output",
            str(out),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text() != "placeholder"


def test_diff_console_refuses_overwrite_without_force(
    runner: CliRunner, fixture_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "diff.txt"
    out.write_text("keep me")
    result = runner.invoke(
        main,
        [
            str(fixture_dir / "clean_minimal.onnx"),
            str(fixture_dir / "clean_minimal.onnx"),
            "--diff",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output.lower()
    assert out.read_text() == "keep me"


def test_explicit_severity_wins_over_verbose(runner: CliRunner, fixture_dir: Path) -> None:
    """`--verbose --severity critical` must honour 'critical', not reset to info."""
    import json

    result = runner.invoke(
        main,
        [
            str(fixture_dir / "failing" / "uint8_input.onnx"),
            "--verbose",
            "--severity",
            "critical",
            "--format",
            "json",
        ],
    )
    payload = json.loads(result.output)
    assert all(i["severity"] == "critical" for i in payload["issues"])


def test_fix_without_output_errors_before_doing_work(runner: CliRunner, fixture_dir: Path) -> None:
    result = runner.invoke(main, [str(fixture_dir / "failing" / "int64_weights.onnx"), "--fix"])
    assert result.exit_code != 0
    assert "--output" in result.output
