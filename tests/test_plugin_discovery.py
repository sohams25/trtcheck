"""Tests for entry-point plugin discovery on Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import onnx
import pytest

from trtcheck.analyzer import Analyzer, AnalyzerConfig
from trtcheck.plugins import Checker, Fixer, Reporter
from trtcheck.types import AnalysisReport, CheckCategory, Issue, Severity


# --- helper plugin classes ---------------------------------------------------


class _ChattyChecker:
    name = "chatty"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        return [
            Issue(
                severity=Severity.INFO,
                category=CheckCategory.GRAPH_STRUCTURE,
                node_name="<chatty>",
                operator="Plugin",
                message="hello from chatty",
                remediation="ignore me",
                docs_link=None,
            )
        ]


class _BrokenChecker:
    name = "broken"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        raise RuntimeError("intentional")


class _BadlyShapedChecker:
    """Lacks the `check` method, so isinstance(Checker) should reject it."""

    name = "wrong_shape"


# --- fake EntryPoint object that loads the helper classes -------------------


@dataclass
class _FakeEntryPoint:
    name: str
    target: type

    def load(self) -> type:
        if isinstance(self.target, type):
            return self.target
        raise self.target  # for tests that simulate an import error


class TestLoadPlugins:
    def test_discovers_protocol_compatible_class(self) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("chatty", _ChattyChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            checkers, fixers, reporters = plugins.load_plugins()
        assert any(c.name == "chatty" for c in checkers)
        assert fixers == []
        assert reporters == []

    def test_isinstance_mismatch_is_skipped(self) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("wrong", _BadlyShapedChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            checkers, _, _ = plugins.load_plugins()
        assert all(c.name != "wrong" for c in checkers)

    def test_import_failure_is_skipped(self) -> None:
        from trtcheck import plugins

        bad = _FakeEntryPoint("broken_import", ImportError("no module 'foo'"))
        eps = {
            "trtcheck.checkers": [bad],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            checkers, _, _ = plugins.load_plugins()
        assert all(c.name != "broken_import" for c in checkers)


class TestAnalyzerPluginIntegration:
    def test_analyzer_includes_discovered_checker(
        self, clean_model: onnx.ModelProto
    ) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("chatty", _ChattyChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            analyzer = Analyzer(AnalyzerConfig())
            report = analyzer.analyze_model(clean_model, filename="x.onnx")
        # The chatty checker emits an INFO issue on every model.
        assert any(i.message == "hello from chatty" for i in report.issues)

    def test_broken_plugin_emits_warning_and_does_not_kill_pipeline(
        self, clean_model: onnx.ModelProto
    ) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("broken", _BrokenChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            analyzer = Analyzer(AnalyzerConfig())
            report = analyzer.analyze_model(clean_model, filename="x.onnx")
        # Pipeline completed; analyzer emitted a warning row referencing the plugin.
        plugin_warnings = [
            i
            for i in report.issues
            if i.severity is Severity.WARNING and "broken" in i.node_name
        ]
        assert plugin_warnings, "broken plugin should produce a warning row"

    def test_disable_plugins_filters_by_name(self, clean_model: onnx.ModelProto) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("chatty", _ChattyChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            analyzer = Analyzer(AnalyzerConfig(disable_plugins=["chatty"]))
            report = analyzer.analyze_model(clean_model, filename="x.onnx")
        assert not any(i.message == "hello from chatty" for i in report.issues)

    def test_discover_entry_point_plugins_false_skips_discovery(
        self, clean_model: onnx.ModelProto
    ) -> None:
        from trtcheck import plugins

        eps = {
            "trtcheck.checkers": [_FakeEntryPoint("chatty", _ChattyChecker)],
            "trtcheck.fixers": [],
            "trtcheck.reporters": [],
        }
        with patch.object(plugins, "_iter_entry_points", side_effect=lambda g: eps[g]):
            analyzer = Analyzer(
                AnalyzerConfig(discover_entry_point_plugins=False)
            )
            report = analyzer.analyze_model(clean_model, filename="x.onnx")
        assert not any(i.message == "hello from chatty" for i in report.issues)

    def test_default_analyzer_has_no_third_party_plugins(
        self, clean_model: onnx.ModelProto
    ) -> None:
        """In a clean test env, no third-party plugins are installed, so the
        clean fixture should still produce zero issues."""
        analyzer = Analyzer(AnalyzerConfig())
        report = analyzer.analyze_model(clean_model, filename="x.onnx")
        assert report.issues == []
