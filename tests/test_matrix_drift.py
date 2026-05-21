"""Tests for tools/check_matrix_drift.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.check_matrix_drift import compare, parse_upstream_markdown

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "matrix_drift"


@pytest.fixture(scope="module")
def upstream_text() -> str:
    return (_FIXTURE_DIR / "operators_upstream.md").read_text()


@pytest.fixture(scope="module")
def matrix() -> dict:
    matrix_path = Path(__file__).parent.parent / "trtcheck" / "data" / "operator_matrix.json"
    return json.loads(matrix_path.read_text())


class TestParseUpstream:
    def test_extracts_table_rows(self, upstream_text: str) -> None:
        parsed = parse_upstream_markdown(upstream_text)
        assert "Abs" in parsed
        assert "Conv" in parsed
        # Y -> supported, N -> not_supported
        assert parsed["Abs"]["status"] == "supported"
        assert parsed["SequenceEmpty"]["status"] == "not_supported"

    def test_ignores_header_separator(self, upstream_text: str) -> None:
        parsed = parse_upstream_markdown(upstream_text)
        # Should never have an entry whose name is dashes/empty
        for op in parsed:
            assert op
            assert not op.startswith("-")
            assert op != "Operator"


class TestCompare:
    def test_flags_op_present_upstream_but_missing_in_matrix(
        self, upstream_text: str, matrix: dict
    ) -> None:
        parsed = parse_upstream_markdown(upstream_text)
        drift = compare(parsed, matrix, target_version="10.3")
        assert any("NewlyAddedOp" in line for line in drift)

    def test_flags_status_mismatch(self, upstream_text: str, matrix: dict) -> None:
        # The fixture says Mish="N" but our matrix says supported on 10.3.
        parsed = parse_upstream_markdown(upstream_text)
        drift = compare(parsed, matrix, target_version="10.3")
        assert any("Mish" in line for line in drift)

    def test_consistent_ops_are_not_flagged(self, upstream_text: str, matrix: dict) -> None:
        parsed = parse_upstream_markdown(upstream_text)
        drift = compare(parsed, matrix, target_version="10.3")
        # Conv is supported on 10.3 in both -> no drift line should mention it.
        assert not any(line.startswith("[mismatch] Conv") for line in drift)

    def test_empty_drift_when_inputs_align(self, matrix: dict) -> None:
        # Build a tiny upstream-equivalent from the matrix itself.
        synthetic_upstream = {
            op: {"status": entry["support"].get("10.3", "supported")}
            for op, entry in list(matrix["operators"].items())[:5]
        }
        drift = compare(synthetic_upstream, matrix, target_version="10.3")
        assert drift == []
