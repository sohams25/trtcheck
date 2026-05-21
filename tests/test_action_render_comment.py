"""Tests for action/render_comment.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_RC_PATH = Path(__file__).parent.parent / "action" / "render_comment.py"


def _import_render_comment():
    spec = importlib.util.spec_from_file_location("render_comment", _RC_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _import_render_comment()


def _report(critical: int = 0, warning: int = 0, *, filename: str = "m.onnx") -> dict:
    issues = []
    for i in range(critical):
        issues.append(
            {
                "severity": "critical",
                "category": "operator_support",
                "node_name": f"n{i}",
                "operator": "SequenceEmpty",
                "message": "Not supported.",
                "remediation": "Replace List[Tensor] with torch.stack().",
                "docs_link": None,
            }
        )
    for i in range(warning):
        issues.append(
            {
                "severity": "warning",
                "category": "precision",
                "node_name": "w",
                "operator": "Initializer",
                "message": "INT64 detected.",
                "remediation": "Cast to int32.",
                "docs_link": None,
            }
        )
    return {
        "filename": filename,
        "issues": issues,
        "critical_count": critical,
        "warning_count": warning,
        "info_count": 0,
        "conversion_likely": critical == 0,
        "estimated_fix_time": "" if critical == 0 else "15-30 minutes",
        "estimated_fusions": [],
        "estimated_precision": {},
        "onnx_ir_version": "8",
        "opset_version": 17,
        "producer": "pytorch",
        "total_nodes": 5,
    }


class TestRender:
    def test_marker_is_first_line(self, rc) -> None:
        md = rc.render({"files": []})
        assert md.startswith(rc.MARKER)

    def test_no_files_message_when_aggregate_empty(self, rc) -> None:
        md = rc.render({"files": []})
        assert "No ONNX files" in md

    def test_summary_counts_match_files(self, rc) -> None:
        agg = {
            "files": [
                {"path": "a.onnx", "report": _report(critical=2)},
                {"path": "b.onnx", "report": _report(warning=3)},
            ]
        }
        md = rc.render(agg)
        assert "2 critical, 3 warning" in md

    def test_overall_verdict_failing_when_any_critical(self, rc) -> None:
        agg = {"files": [{"path": "a.onnx", "report": _report(critical=1)}]}
        md = rc.render(agg)
        assert "Conversion will fail" in md
        assert "❌ fail" in md

    def test_overall_verdict_passing_with_warnings_only(self, rc) -> None:
        agg = {"files": [{"path": "a.onnx", "report": _report(warning=2)}]}
        md = rc.render(agg)
        assert "Likely to convert" in md
        assert "✅ pass" in md

    def test_details_block_for_each_file_with_issues(self, rc) -> None:
        agg = {
            "files": [
                {"path": "a.onnx", "report": _report(critical=1)},
                {"path": "b.onnx", "report": _report()},
            ]
        }
        md = rc.render(agg)
        assert "<details><summary>Details for <code>a.onnx</code>" in md
        # No details for clean file
        assert "Details for <code>b.onnx</code>" not in md

    def test_table_pipe_in_message_is_escaped(self, rc) -> None:
        report = _report(critical=1)
        report["issues"][0]["message"] = "a | b | c"
        agg = {"files": [{"path": "x.onnx", "report": report}]}
        md = rc.render(agg)
        # The escaped pipe should appear inside the details table cell
        assert "a \\| b \\| c" in md

    def test_long_message_is_truncated(self, rc) -> None:
        report = _report(critical=1)
        report["issues"][0]["message"] = "x" * 500
        agg = {"files": [{"path": "x.onnx", "report": report}]}
        md = rc.render(agg)
        # 80-char truncation in the summary table
        assert "…" in md
