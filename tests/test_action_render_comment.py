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


class TestInjectionDefense:
    def _payload(
        self, rc, *, op: str = "Op", msg: str = "m", node: str = "n", path: str = "x.onnx"
    ) -> dict:
        return {
            "files": [
                {
                    "path": path,
                    "report": {
                        "filename": path,
                        "issues": [
                            {
                                "severity": "critical",
                                "category": "operator_support",
                                "node_name": node,
                                "operator": op,
                                "message": msg,
                                "remediation": "fix me",
                                "docs_link": None,
                            }
                        ],
                        "critical_count": 1,
                        "warning_count": 0,
                        "info_count": 0,
                        "conversion_likely": False,
                        "estimated_fix_time": "15-30 minutes",
                        "onnx_ir_version": "8",
                        "opset_version": 17,
                        "producer": "p",
                        "total_nodes": 1,
                    },
                }
            ]
        }

    def test_backticks_in_operator_are_neutralized(self, rc) -> None:
        md = rc.render(self._payload(rc, op="`evil`"))
        # The original backticks must not survive in a code span.
        assert "`evil`" not in md

    def test_pipe_in_operator_cannot_break_table(self, rc) -> None:
        # A raw '|' in a code-span table cell would add phantom columns.
        md = rc.render(self._payload(rc, op="A|B|C"))
        assert "A|B|C" not in md
        assert "A│B│C" in md

    def test_html_tags_in_node_name_are_entity_escaped(self, rc) -> None:
        md = rc.render(self._payload(rc, msg="<script>alert(1)</script>"))
        # Literal <script> must be escaped to &lt;script&gt;
        assert "<script>" not in md
        assert "&lt;script&gt;" in md

    def test_details_block_filename_cannot_break_out(self, rc) -> None:
        md = rc.render(self._payload(rc, path="</summary><b>x</b>"))
        # The literal closing tag must not appear inside the rendered comment.
        # </summary> ONLY shows up as the legitimate closer of the details block.
        # We assert the dangerous prefix did not become a raw </summary> inside
        # the code span.
        assert "<code></summary>" not in md

    def test_markdown_link_in_message_is_disarmed(self, rc) -> None:
        md = rc.render(self._payload(rc, msg="[click](https://evil.example)"))
        # We escape the brackets so the link syntax does not render.
        assert "[click]" not in md
        assert "\\[click\\]" in md
