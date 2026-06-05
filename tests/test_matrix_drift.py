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


class TestVersionAwareness:
    def test_parser_tags_column_version(self, upstream_text: str) -> None:
        # The fixture header is "TensorRT 10.x" -> major "10".
        parsed = parse_upstream_markdown(upstream_text)
        assert parsed["Abs"]["version"] == "10"

    def test_non_matching_target_emits_no_spurious_mismatch(
        self, upstream_text: str, matrix: dict
    ) -> None:
        # The upstream table covers 10.x. GroupNormalization (8.0=not_supported)
        # and LayerNormalization (8.0=partial) would falsely "mismatch" the 10.x
        # 'supported' column if compared at 8.0. They must NOT be flagged.
        parsed = parse_upstream_markdown(upstream_text)
        drift = compare(parsed, matrix, target_version="8.0")
        assert not any("GroupNormalization" in line for line in drift)
        assert not any("LayerNormalization" in line for line in drift)
        # Nothing in a 10.x table is comparable to 8.0 -> no drift at all.
        assert drift == []

    def test_matching_target_still_flags_real_drift(self, upstream_text: str, matrix: dict) -> None:
        # 10.3 shares major 10 with the table, so the real Mish mismatch surfaces.
        parsed = parse_upstream_markdown(upstream_text)
        drift = compare(parsed, matrix, target_version="10.3")
        assert any("Mish" in line for line in drift)

    def test_versionless_upstream_is_compared_as_before(self, matrix: dict) -> None:
        # An upstream dict with no 'version' key (e.g. a synthetic one) must
        # still be compared, preserving back-compat.
        synthetic = {"Mish": {"status": "not_supported"}}
        drift = compare(synthetic, matrix, target_version="10.3")
        assert any("Mish" in line for line in drift)

    def test_trt_abbreviation_header_is_recognized(self) -> None:
        # Real upstream tables abbreviate "TensorRT" as "TRT".
        text = "| Operator | TRT 10.x | Restrictions |\n|---|---|---|\n| Conv | Y | |\n"
        parsed = parse_upstream_markdown(text)
        assert parsed["Conv"]["version"] == "10"
        assert parsed["Conv"]["status"] == "supported"

    def test_status_column_located_by_header_not_position(self) -> None:
        # The version column is not always the second cell; find it by header.
        text = (
            "| Operator | Domain | TensorRT 10.x | Restrictions |\n"
            "|---|---|---|---|\n"
            "| Conv | ai.onnx | Y | none |\n"
        )
        parsed = parse_upstream_markdown(text)
        assert parsed["Conv"]["status"] == "supported"
        assert parsed["Conv"]["version"] == "10"

    def test_multi_version_columns_warn_and_take_last(self, capsys: pytest.CaptureFixture) -> None:
        text = (
            "| Operator | TRT 8.x | TRT 10.x | Restrictions |\n"
            "|---|---|---|---|\n"
            "| Conv | N | Y | |\n"
        )
        parsed = parse_upstream_markdown(text)
        # Last version column wins (10.x -> supported), and a warning is emitted.
        assert parsed["Conv"]["version"] == "10"
        assert parsed["Conv"]["status"] == "supported"
        assert "multiple TRT version columns" in capsys.readouterr().err
