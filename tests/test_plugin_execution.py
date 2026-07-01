"""Plugin fixers and reporters must be executed, not just listed.

QA found that ``load_plugins()`` output for fixers and reporters was only
consumed by ``--list-plugins``: ``--fix`` ran the hardcoded built-ins and
``--format`` was a closed three-way choice, so the documented extension
path (the shipped example ships a *fixer*) visibly did nothing. These
tests pin the wiring end-to-end through the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner

from trtcheck.cli import main
from trtcheck.fixers import FixApplied
from trtcheck.types import AnalysisReport


class _ProducerStampFixer:
    """Observable, always-valid rewrite: stamp the producer name."""

    name = "producer_stamp"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        if model.producer_name == "stamped":
            return []
        model.producer_name = "stamped"
        return [FixApplied(fixer=self.name, target="model", description="stamp producer name")]


class _CrashingFixer:
    name = "crashy"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        raise RuntimeError("intentional")


class _ShoutReporter:
    name = "shout"

    def render(self, report: AnalysisReport) -> str:
        return f"SHOUT {report.filename} crit={report.critical_count}"


@dataclass
class _FakeEntryPoint:
    name: str
    target: type

    def load(self) -> type:
        return self.target


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: dict[str, list]) -> None:
    from trtcheck import plugins

    monkeypatch.setattr(plugins, "_iter_entry_points", lambda group: eps.get(group, []))


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestPluginFixerExecution:
    def test_plugin_fixer_runs_under_fix(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(
            monkeypatch,
            {"trtcheck.fixers": [_FakeEntryPoint("producer_stamp", _ProducerStampFixer)]},
        )
        result = runner.invoke(
            main, [str(fixture_dir / "clean_minimal.onnx"), "--fix", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "producer_stamp" in result.output

    def test_disable_plugin_excludes_plugin_fixer(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(
            monkeypatch,
            {"trtcheck.fixers": [_FakeEntryPoint("producer_stamp", _ProducerStampFixer)]},
        )
        result = runner.invoke(
            main,
            [
                str(fixture_dir / "clean_minimal.onnx"),
                "--fix",
                "--dry-run",
                "--disable-plugin",
                "producer_stamp",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "producer_stamp" not in result.output
        assert "no fixes applied" in result.output

    def test_crashing_plugin_fixer_is_skipped_with_warning(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(
            monkeypatch, {"trtcheck.fixers": [_FakeEntryPoint("crashy", _CrashingFixer)]}
        )
        result = runner.invoke(
            main, [str(fixture_dir / "clean_minimal.onnx"), "--fix", "--dry-run"]
        )
        # The run survives; the failure is reported, not a traceback.
        assert result.exit_code == 0, result.output
        assert "crashy" in (result.output + result.stderr)


class TestPluginReporterExecution:
    def test_plugin_reporter_selectable_via_format(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(
            monkeypatch, {"trtcheck.reporters": [_FakeEntryPoint("shout", _ShoutReporter)]}
        )
        result = runner.invoke(main, [str(fixture_dir / "clean_minimal.onnx"), "--format", "shout"])
        assert result.exit_code == 0, result.output
        assert result.output.startswith("SHOUT ")

    def test_unknown_format_errors_and_lists_available(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(monkeypatch, {})
        result = runner.invoke(main, [str(fixture_dir / "clean_minimal.onnx"), "--format", "nope"])
        assert result.exit_code != 0
        assert "console" in result.output and "json" in result.output

    def test_disabled_plugin_reporter_is_not_selectable(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(
            monkeypatch, {"trtcheck.reporters": [_FakeEntryPoint("shout", _ShoutReporter)]}
        )
        result = runner.invoke(
            main,
            [
                str(fixture_dir / "clean_minimal.onnx"),
                "--format",
                "shout",
                "--disable-plugin",
                "shout",
            ],
        )
        assert result.exit_code != 0


class TestDisablePluginTypoGuard:
    def test_unknown_disable_plugin_name_warns(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(monkeypatch, {})
        result = runner.invoke(
            main,
            [str(fixture_dir / "clean_minimal.onnx"), "--disable-plugin", "totally_not_real"],
        )
        assert result.exit_code == 0, result.output
        assert "totally_not_real" in result.stderr
        assert "warning" in result.stderr.lower()

    def test_known_disable_plugin_name_does_not_warn(
        self, runner: CliRunner, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_entry_points(monkeypatch, {})
        result = runner.invoke(
            main, [str(fixture_dir / "clean_minimal.onnx"), "--disable-plugin", "precision"]
        )
        assert result.exit_code == 0, result.output
        assert "warning" not in result.stderr.lower()
