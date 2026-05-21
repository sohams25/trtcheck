"""Tests for tools/build_operator_docs.py.

The generator reads operator_matrix.json and writes one markdown file per
operator. The tests use small synthetic matrices so they stay fast and
don't drift when the real matrix is updated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_operator_docs import build, render_index, render_operator


def _matrix(*ops: tuple[str, dict]) -> dict:
    return {
        "schema_version": "1.0",
        "last_updated": "2026-01-01",
        "target_trt_versions": ["8.0", "8.6", "10.0", "10.3"],
        "operators": dict(ops),
    }


class TestRenderOperator:
    def test_includes_op_name_as_h1(self) -> None:
        entry = {
            "support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}
        }
        md = render_operator("Conv", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        assert md.splitlines()[0] == "# Conv"

    def test_support_table_has_row_per_version(self) -> None:
        entry = {
            "support": {"8.0": "not_supported", "8.6": "partial", "10.0": "supported", "10.3": "supported"}
        }
        md = render_operator("Mish", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        for v in ["8.0", "8.6", "10.0", "10.3"]:
            assert v in md
        assert "not_supported" in md
        assert "partial" in md
        assert "supported" in md

    def test_notes_section_present_when_set(self) -> None:
        entry = {
            "support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"},
            "notes": "Full support. NCHW format.",
        }
        md = render_operator("Conv", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        assert "Full support. NCHW format." in md

    def test_limitations_render_as_bullets(self) -> None:
        entry = {
            "support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"},
            "limitations": ["3D Conv requires TRT 8.6+", "asymmetric padding handled separately"],
        }
        md = render_operator("Conv", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        assert "- 3D Conv requires TRT 8.6+" in md
        assert "- asymmetric padding handled separately" in md

    def test_remediation_renders_when_set(self) -> None:
        entry = {
            "support": {"8.0": "not_supported", "8.6": "not_supported", "10.0": "not_supported", "10.3": "not_supported"},
            "remediation": "Replace List[Tensor] with torch.stack().",
        }
        md = render_operator("SequenceEmpty", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        assert "Replace List[Tensor] with torch.stack()." in md

    def test_github_issue_link_renders(self) -> None:
        entry = {
            "support": {"8.0": "not_supported", "8.6": "not_supported", "10.0": "not_supported", "10.3": "not_supported"},
            "github_issue": "https://github.com/onnx/onnx-tensorrt/issues/1044",
        }
        md = render_operator("SequenceEmpty", entry, versions=["8.0", "8.6", "10.0", "10.3"])
        assert "https://github.com/onnx/onnx-tensorrt/issues/1044" in md


class TestRenderIndex:
    def test_index_lists_every_operator(self) -> None:
        matrix = _matrix(
            ("Conv", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
            ("Relu", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
        )
        md = render_index(matrix)
        assert "Conv" in md
        assert "Relu" in md
        # Each entry should be a link to the per-op page
        assert "Conv.md" in md
        assert "Relu.md" in md

    def test_index_is_alphabetically_sorted(self) -> None:
        matrix = _matrix(
            ("Relu", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
            ("Conv", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
            ("Add", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
        )
        md = render_index(matrix)
        # Order in the index should be sorted regardless of dict insertion order.
        pos_add = md.find("Add")
        pos_conv = md.find("Conv")
        pos_relu = md.find("Relu")
        assert 0 <= pos_add < pos_conv < pos_relu


class TestBuild:
    def test_build_writes_one_file_per_operator_plus_index(self, tmp_path: Path) -> None:
        matrix = _matrix(
            ("Conv", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}}),
            ("Mish", {"support": {"8.0": "not_supported", "8.6": "partial", "10.0": "supported", "10.3": "supported"}}),
        )
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        out_dir = tmp_path / "out"
        build(matrix_path, out_dir)
        assert (out_dir / "Conv.md").exists()
        assert (out_dir / "Mish.md").exists()
        assert (out_dir / "index.md").exists()

    def test_build_is_idempotent(self, tmp_path: Path) -> None:
        matrix = _matrix(
            ("Conv", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}})
        )
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        out_dir = tmp_path / "out"
        build(matrix_path, out_dir)
        first = (out_dir / "Conv.md").read_text()
        build(matrix_path, out_dir)
        second = (out_dir / "Conv.md").read_text()
        assert first == second

    def test_build_removes_stale_pages(self, tmp_path: Path) -> None:
        """Operator pages for ops that left the matrix must be cleaned up.

        Otherwise, dropping an operator from the matrix would leave a dangling
        page in the docs site forever.
        """
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "Stale.md").write_text("# leftover")

        matrix = _matrix(
            ("Conv", {"support": {"8.0": "supported", "8.6": "supported", "10.0": "supported", "10.3": "supported"}})
        )
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        build(matrix_path, out_dir)
        assert not (out_dir / "Stale.md").exists()
        assert (out_dir / "Conv.md").exists()

    def test_build_against_real_matrix(self, tmp_path: Path) -> None:
        """Smoke: the live operator_matrix.json must drive a successful build."""
        repo_root = Path(__file__).resolve().parent.parent
        matrix_path = repo_root / "trtcheck" / "data" / "operator_matrix.json"
        out_dir = tmp_path / "out"
        build(matrix_path, out_dir)
        # Sanity: at least the canonical ops should have pages.
        for op in ["Conv", "SequenceEmpty", "Loop", "Cast"]:
            assert (out_dir / f"{op}.md").exists(), op
