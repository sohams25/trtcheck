"""Tests for tools/validate_bench_manifest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.validate_bench_manifest import ManifestError, validate


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(body)
    return p


class TestValidate:
    def test_bundled_manifest_is_valid(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        entries = validate(repo_root / "bench" / "manifest.yaml")
        assert len(entries) >= 6
        # The bundled-fixture rows must all reference existing files in the tree.
        for e in entries:
            if not e["source"].startswith(("http://", "https://")):
                assert (repo_root / e["source"]).exists()

    def test_missing_root_models_key(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "other: thing\n")
        with pytest.raises(ManifestError, match="must be a non-empty list"):
            validate(p, repo_root=tmp_path)

    def test_empty_models_list(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "models: []\n")
        with pytest.raises(ManifestError, match="non-empty"):
            validate(p, repo_root=tmp_path)

    def test_missing_required_key(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n  - name: a\n    source: https://x/y.onnx\n    expected: fail\n",
        )
        with pytest.raises(ManifestError, match="missing keys"):
            validate(p, repo_root=tmp_path)

    def test_invalid_expected_value(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n"
            "  - name: a\n    source: https://x/y.onnx\n"
            "    expected: maybe\n    reason: none\n",
        )
        with pytest.raises(ManifestError, match="expected must be one of"):
            validate(p, repo_root=tmp_path)

    def test_invalid_reason_value(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n"
            "  - name: a\n    source: https://x/y.onnx\n"
            "    expected: fail\n    reason: vibes\n",
        )
        with pytest.raises(ManifestError, match="reason must be one of"):
            validate(p, repo_root=tmp_path)

    def test_duplicate_name(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n"
            "  - name: a\n    source: https://x/y.onnx\n    expected: fail\n    reason: none\n"
            "  - name: a\n    source: https://x/z.onnx\n    expected: convert\n    reason: none\n",
        )
        with pytest.raises(ManifestError, match="duplicate name"):
            validate(p, repo_root=tmp_path)

    def test_bundled_path_must_exist(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n"
            "  - name: ghost\n    source: not/a/real/file.onnx\n"
            "    expected: convert\n    reason: none\n",
        )
        with pytest.raises(ManifestError, match="does not exist"):
            validate(p, repo_root=tmp_path)

    def test_url_source_is_accepted_without_filesystem_check(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            "models:\n"
            "  - name: a\n    source: https://example.com/model.onnx\n"
            "    expected: convert\n    reason: none\n",
        )
        entries = validate(p, repo_root=tmp_path)
        assert len(entries) == 1
