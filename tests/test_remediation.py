"""Tests for the trtcheck.remediation accessor.

remediation.py is the single source of truth for the explanation / remediation /
docs_link / severity of every non-operator finding: it loads remediation_db.json
once and hands back typed, frozen entries.
"""

from __future__ import annotations

import dataclasses

import pytest

from trtcheck import remediation
from trtcheck.types import CheckCategory, Severity


def test_get_returns_typed_entry() -> None:
    e = remediation.get("uint8_input")
    assert e.severity is Severity.CRITICAL
    assert e.category is CheckCategory.PRECISION
    assert "preprocessing" in e.remediation.lower()


def test_get_maps_severity_and_category_enums() -> None:
    e = remediation.get("int64_weights")
    assert e.severity is Severity.WARNING
    assert isinstance(e.severity, Severity)
    assert isinstance(e.category, CheckCategory)


def test_get_unknown_key_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        remediation.get("definitely_not_a_real_issue_id")


def test_entry_is_frozen() -> None:
    e = remediation.get("nested_loop")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.severity = Severity.WARNING  # type: ignore[misc]


def test_known_ids_contains_canonical_keys() -> None:
    ids = remediation.known_ids()
    assert isinstance(ids, frozenset)
    for key in ("int64_weights", "uint8_input", "nested_loop", "fully_dynamic_input_shape"):
        assert key in ids


def test_absent_docs_link_is_none() -> None:
    # bf16_unsupported has "docs_link": null in the DB.
    assert remediation.get("bf16_unsupported").docs_link is None


def test_to_entry_rejects_bad_severity() -> None:
    bad = {
        "category": "precision",
        "severity": "catastrophic",  # not a Severity member
        "summary": "x",
        "explanation": "y",
        "remediation": "z",
    }
    with pytest.raises(ValueError):
        remediation._to_entry("bad_key", bad)


def test_to_entry_rejects_bad_category() -> None:
    bad = {
        "category": "made_up_category",
        "severity": "warning",
        "summary": "x",
        "explanation": "y",
        "remediation": "z",
    }
    with pytest.raises(ValueError):
        remediation._to_entry("bad_key", bad)


def test_parse_rejects_malformed_json() -> None:
    # A truncated/invalid DB must fail with a clear ValueError, not a raw
    # JSONDecodeError that crashes every checker import opaquely.
    with pytest.raises(ValueError, match="malformed"):
        remediation._parse("{ this is not valid json")


def test_parse_rejects_missing_issues_key() -> None:
    with pytest.raises(ValueError, match="malformed"):
        remediation._parse('{"schema_version": "1.0"}')
