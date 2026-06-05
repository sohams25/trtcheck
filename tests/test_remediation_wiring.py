"""Guard the checker <-> remediation_db.json wiring.

Each checker declares an ``EMITS`` set of remediation-DB keys it can produce.
These tests make the DB the enforced single source of truth: every emitted key
must exist, its DB category must match the checker's domain, and the few keys
whose severity decides the PASS/FAIL verdict are pinned so a one-line JSON edit
can't silently flip a verdict.
"""

from __future__ import annotations

import pytest

from trtcheck import remediation
from trtcheck.checkers import control_flow, dynamic_shapes, graph_structure, precision
from trtcheck.types import CheckCategory, Severity

# (checker module, the CheckCategory every one of its findings should carry)
_CHECKERS = [
    (precision, CheckCategory.PRECISION),
    (control_flow, CheckCategory.CONTROL_FLOW),
    (dynamic_shapes, CheckCategory.DYNAMIC_SHAPES),
    (graph_structure, CheckCategory.GRAPH_STRUCTURE),
]

_ALL_EMITTED = sorted({k for mod, _cat in _CHECKERS for k in mod.EMITS})


@pytest.mark.parametrize("mod,category", _CHECKERS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_every_emitted_key_exists_in_db(mod, category) -> None:
    missing = mod.EMITS - remediation.known_ids()
    assert not missing, f"{mod.__name__} emits DB keys that don't exist: {missing}"


@pytest.mark.parametrize("mod,category", _CHECKERS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_emitted_keys_have_matching_category(mod, category) -> None:
    for key in mod.EMITS:
        assert remediation.get(key).category is category, (
            f"{mod.__name__} emits {key!r} but its DB category is "
            f"{remediation.get(key).category}, expected {category}"
        )


@pytest.mark.parametrize("issue_id", _ALL_EMITTED)
def test_emitted_entries_have_remediation_text(issue_id: str) -> None:
    entry = remediation.get(issue_id)
    assert entry.remediation.strip(), f"{issue_id} has empty remediation"
    assert entry.explanation.strip(), f"{issue_id} has empty explanation"


# Verdict-critical severities: a careless DB edit here would flip PASS/FAIL, so
# pin them. Re-stating these few in the test is deliberate insurance.
_SEVERITY_PINS = {
    "uint8_input": Severity.CRITICAL,
    "float64_tensors": Severity.CRITICAL,
    "string_tensors": Severity.CRITICAL,
    "missing_output": Severity.CRITICAL,
    "nested_loop": Severity.CRITICAL,
    "int64_weights": Severity.WARNING,
    "int64_input": Severity.WARNING,
    "bf16_unsupported": Severity.WARNING,
    "if_detected_unverified": Severity.WARNING,
    "loop_dynamic_trip_count": Severity.WARNING,
    "scan_dynamic_length": Severity.WARNING,
    "fully_dynamic_input_shape": Severity.WARNING,
    "duplicate_node_name": Severity.WARNING,
    "large_constant": Severity.INFO,
}


@pytest.mark.parametrize("issue_id,expected", sorted(_SEVERITY_PINS.items()))
def test_severity_pins(issue_id: str, expected: Severity) -> None:
    assert remediation.get(issue_id).severity is expected, (
        f"{issue_id} severity changed to {remediation.get(issue_id).severity}; "
        f"this flips the verdict -- update the checker/test deliberately if intended"
    )


def test_precision_dtype_maps_stay_in_sync_with_emits() -> None:
    # precision builds keys from private dtype->key maps (not literals), so pin
    # that their real key universe equals EMITS -- otherwise a new dtype mapping
    # could emit a key the EMITS<=known_ids guard never checks.
    real_keys = (
        {v[0] for v in precision._INPUT_DTYPES.values()}
        | {v[0] for v in precision._INIT_DTYPES.values()}
        | {"float64_tensors"}  # the internal Cast/Constant-to-DOUBLE literal
    )
    assert real_keys == precision.EMITS


def test_if_is_not_keyed_to_the_critical_mismatch_entry() -> None:
    # The If check is a heuristic warning; it must NOT use the CRITICAL
    # if_branch_shape_mismatch entry or every model with an If would fail.
    assert "if_branch_shape_mismatch" not in control_flow.EMITS
    assert "if_detected_unverified" in control_flow.EMITS
