"""Guard the checker <-> remediation_db.json wiring.

Each checker declares an ``EMITS`` set of remediation-DB keys it can produce.
These tests make the DB the enforced single source of truth: every emitted key
must exist, its DB category must match the checker's domain, and the few keys
whose severity decides the PASS/FAIL verdict are pinned so a one-line JSON edit
can't silently flip a verdict.
"""

from __future__ import annotations

import ast
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from trtcheck import remediation
from trtcheck.checkers import control_flow, dynamic_shapes, graph_structure, precision
from trtcheck.checkers.control_flow import ControlFlowChecker
from trtcheck.types import CheckCategory, Severity

_CHECKER_DIR = Path(__file__).resolve().parent.parent / "trtcheck" / "checkers"

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
    "loop_runtime_trip_count": Severity.CRITICAL,
    "loop_dynamic_trip_count": Severity.WARNING,
    "scan_dynamic_length": Severity.WARNING,
    "fully_dynamic_input_shape": Severity.WARNING,
    "duplicate_node_name": Severity.WARNING,
    "large_constant": Severity.INFO,
    "input_with_no_type": Severity.CRITICAL,
    "opset_too_old": Severity.WARNING,
    "isolated_node": Severity.WARNING,
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


def _make_issue_literal_keys(module_path: Path) -> set[str]:
    """AST-extract every string-literal first arg passed to remediation.make_issue."""
    tree = ast.parse(module_path.read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "make_issue"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


@pytest.mark.parametrize("mod,category", _CHECKERS, ids=lambda x: getattr(x, "__name__", str(x)))
def test_make_issue_literal_keys_are_declared_in_emits(mod, category) -> None:
    # Generalizes the precision-dict guard to the literal-based checkers: any key
    # passed as a literal to make_issue must be in EMITS, so the EMITS<=known_ids
    # + category + severity guards actually cover it (a typo'd literal can't slip
    # past). precision's variable-keyed calls are covered by the dtype-map test.
    path = _CHECKER_DIR / f"{mod.__name__.rsplit('.', 1)[-1]}.py"
    literals = _make_issue_literal_keys(path)
    assert (
        literals <= mod.EMITS
    ), f"{mod.__name__} emits literal keys outside EMITS: {literals - mod.EMITS}"


def test_every_db_key_is_emitted_operator_owned_or_reserved() -> None:
    """Pin the full DB key inventory so dead entries are a reviewed contract.

    Forces a conscious test update whenever a checker starts/stops emitting a key
    or a new DB entry is added -- silent drift can't accumulate.
    """
    emitted = {key for mod, _cat in _CHECKERS for key in mod.EMITS}
    # operator_support is matrix-driven (operator_matrix.json), not DB-driven, but
    # these generic entries describe its finding classes.
    operator_owned = {"unsupported_operator", "partial_operator"}
    # Defined but not yet emitted by any checker (reserved for future/alternative
    # findings, incl. the CRITICAL If-mismatch the heuristic If check avoids).
    reserved = {
        "in_place_aliasing",
        "external_data_missing",
        "if_branch_shape_mismatch",
    }
    assert remediation.known_ids() == emitted | operator_owned | reserved, (
        "remediation_db.json key inventory changed; update this contract: "
        f"unaccounted={remediation.known_ids() - (emitted | operator_owned | reserved)}"
    )


@pytest.mark.parametrize("issue_id", ["uint8_input", "int64_input", "int64_weights"])
def test_precision_dtype_keys_carry_docs_link(issue_id: str) -> None:
    # These findings gained a docs_link when wired to the DB (the old inline code
    # set None). Lock the intentional enrichment so a careless edit can't drop it.
    link = remediation.get(issue_id).docs_link
    assert link and link.startswith("https://"), f"{issue_id} should carry an https docs_link"


def _undef_out(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def test_if_check_emits_one_warning_end_to_end() -> None:
    # Behavioral counterpart to the metadata-level If guard: a real If model must
    # produce exactly one WARNING keyed through if_detected_unverified.
    then_g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["t"], name="t")], "then", [], [_undef_out("t")]
    )
    else_g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["e"], name="e")], "else", [], [_undef_out("e")]
    )
    cond = helper.make_node(
        "Constant", [], ["c"], name="c", value=helper.make_tensor("c", TensorProto.BOOL, [], [True])
    )
    ifn = helper.make_node("If", ["c"], ["o"], name="my_if", then_branch=then_g, else_branch=else_g)
    g = helper.make_graph(
        [cond, ifn],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [_undef_out("o")],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])

    issues = [i for i in ControlFlowChecker().check(model) if i.operator == "If"]
    assert len(issues) == 1
    assert issues[0].severity is Severity.WARNING
    assert "detected" in issues[0].message.lower()
