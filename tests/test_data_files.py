"""Schema validation for the bundled JSON data files.

Both files are hand-curated, so the test exists to catch typos and
structural drift during PR review rather than to validate any logic.
"""

import json
from importlib import resources
from pathlib import Path

import pytest

_DATA_ROOT = Path(__file__).parent.parent / "trtcheck" / "data"
_VALID_SUPPORT = {"supported", "partial", "not_supported", "unknown"}
_EXPECTED_VERSIONS = {"8.0", "8.6", "10.0", "10.3"}


@pytest.fixture(scope="module")
def matrix() -> dict:
    with open(_DATA_ROOT / "operator_matrix.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def remediation() -> dict:
    with open(_DATA_ROOT / "remediation_db.json") as f:
        return json.load(f)


class TestOperatorMatrix:
    def test_top_level_fields(self, matrix: dict) -> None:
        assert matrix["schema_version"] == "2.0"
        assert set(matrix["target_trt_versions"]) == _EXPECTED_VERSIONS
        assert isinstance(matrix["operators"], dict)

    def test_minimum_operator_count(self, matrix: dict) -> None:
        # The spec calls for 50+; we ship with 100.
        assert len(matrix["operators"]) >= 50

    def test_every_operator_covers_every_target_version(self, matrix: dict) -> None:
        for op, entry in matrix["operators"].items():
            assert "support" in entry, f"{op} missing 'support' key"
            assert (
                set(entry["support"].keys()) == _EXPECTED_VERSIONS
            ), f"{op} support keys are {set(entry['support'])}, expected {_EXPECTED_VERSIONS}"

    def test_support_values_are_valid(self, matrix: dict) -> None:
        for op, entry in matrix["operators"].items():
            for version, status in entry["support"].items():
                assert status in _VALID_SUPPORT, f"{op}[{version}]={status} not in {_VALID_SUPPORT}"

    def test_key_failure_modes_are_covered(self, matrix: dict) -> None:
        # If these ever fall out of the matrix the README examples lie.
        for required in ["SequenceEmpty", "Loop", "If", "Cast", "GroupNormalization"]:
            assert required in matrix["operators"], f"{required} missing from matrix"

    def test_sequence_ops_uniformly_unsupported(self, matrix: dict) -> None:
        # The whole sequence family is the canonical TRT failure mode.
        for op in matrix["operators"]:
            if op.startswith("Sequence"):
                for status in matrix["operators"][op]["support"].values():
                    assert status == "not_supported", f"{op} unexpectedly {status}"


class TestRemediationDb:
    def test_top_level_fields(self, remediation: dict) -> None:
        assert remediation["schema_version"] == "2.0"
        assert isinstance(remediation["issues"], dict)

    def test_minimum_entry_count(self, remediation: dict) -> None:
        assert len(remediation["issues"]) >= 20

    def test_every_entry_has_required_fields(self, remediation: dict) -> None:
        required = {"category", "severity", "summary", "explanation", "remediation"}
        for key, entry in remediation["issues"].items():
            missing = required - entry.keys()
            assert not missing, f"{key} missing fields: {missing}"

    def test_severities_are_valid(self, remediation: dict) -> None:
        for key, entry in remediation["issues"].items():
            assert entry["severity"] in {"critical", "warning", "info"}, key

    def test_canonical_issues_present(self, remediation: dict) -> None:
        for key in [
            "int64_weights",
            "uint8_input",
            "fully_dynamic_input_shape",
            "loop_dynamic_trip_count",
            "unsupported_operator",
        ]:
            assert key in remediation["issues"], f"{key} missing from remediation db"
