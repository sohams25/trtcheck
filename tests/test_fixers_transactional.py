"""Transactional guarantees of the fixer pipeline (run_fixers).

A fixer -- built-in or third-party -- must not be able to leave a partial
mutation in the output model, no matter how it fails: raising mid-mutation,
returning malformed records, mutating without declaring it, or producing an
ONNX-invalid model.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.fixers import FixApplied, run_fixers
from trtcheck.fixers.drop_dropout import DropDropoutFixer


def _valid_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
    relu = helper.make_node("Relu", ["x"], ["mid"], name="relu")
    drop = helper.make_node("Dropout", ["mid"], ["d"], name="drop")
    ident = helper.make_node("Identity", ["d"], ["y"], name="ident")
    graph = helper.make_graph([relu, drop, ident], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class _MutateThenCrash:
    """Adversarial fixer: renames every node, THEN raises."""

    name = "mutate_then_crash"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        for node in model.graph.node:
            node.name = "CORRUPTED_" + node.name
        model.graph.node[0].op_type = "TotallyBogusOp"
        raise RuntimeError("boom after mutation")


class _MalformedReturn:
    name = "malformed_return"

    def fix(self, model: onnx.ModelProto) -> object:
        model.graph.node[0].name = "MUTATED_ANYWAY"
        return {"not": "a list of FixApplied"}


class _UndeclaredMutation:
    """Claims it did nothing but actually mutated the candidate."""

    name = "undeclared_mutation"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        model.graph.node[0].name = "SNEAKY_EDIT"
        return []


class _EmitsInvalidModel:
    """Declares a fix but breaks the graph (dangling output reference)."""

    name = "emits_invalid"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        model.graph.node[-1].input[0] = "no_such_tensor"
        return [FixApplied(fixer=self.name, target="y", description="broke the graph")]


def _node_names(model: onnx.ModelProto) -> list[str]:
    return [n.name for n in model.graph.node]


class TestTransactionalPipeline:
    def test_mutate_then_crash_leaves_no_trace(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [_MutateThenCrash()])
        assert outcome.applied == []
        assert len(outcome.failures) == 1
        assert outcome.failures[0].fixer == "mutate_then_crash"
        assert "RuntimeError" in outcome.failures[0].reason
        assert not any(n.startswith("CORRUPTED_") for n in _node_names(outcome.model))
        onnx.checker.check_model(outcome.model, full_check=True)

    def test_failed_fixer_does_not_block_later_fixers(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [_MutateThenCrash(), DropDropoutFixer()])
        assert [f.fixer for f in outcome.failures] == ["mutate_then_crash"]
        assert [a.fixer for a in outcome.applied] == ["drop_dropout"]
        assert not any(n.op_type == "Dropout" for n in outcome.model.graph.node)
        assert not any(n.startswith("CORRUPTED_") for n in _node_names(outcome.model))
        onnx.checker.check_model(outcome.model, full_check=True)

    def test_malformed_return_is_rejected_and_mutation_discarded(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [_MalformedReturn()])  # type: ignore[list-item]
        assert outcome.applied == []
        assert [f.fixer for f in outcome.failures] == ["malformed_return"]
        assert "MUTATED_ANYWAY" not in _node_names(outcome.model)

    def test_undeclared_mutation_is_discarded(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [_UndeclaredMutation()])
        assert outcome.applied == []
        assert outcome.failures == []
        assert "SNEAKY_EDIT" not in _node_names(outcome.model)

    def test_invalid_candidate_is_discarded(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [_EmitsInvalidModel()])
        assert outcome.applied == []
        assert [f.fixer for f in outcome.failures] == ["emits_invalid"]
        assert "produced an invalid model" in outcome.failures[0].reason
        onnx.checker.check_model(outcome.model, full_check=True)

    def test_input_model_is_never_mutated(self) -> None:
        model = _valid_model()
        before = model.SerializeToString()
        run_fixers(model, [_MutateThenCrash(), DropDropoutFixer(), _EmitsInvalidModel()])
        assert model.SerializeToString() == before

    def test_validation_level_recorded(self) -> None:
        model = _valid_model()
        outcome = run_fixers(model, [DropDropoutFixer()])
        assert outcome.validation == "full"


def test_external_data_model_gets_basic_validation(tmp_path, monkeypatch) -> None:
    """External-data initializers cannot be read by full inference from an
    in-memory proto; the pipeline must degrade to the basic check, not crash."""
    # onnx.checker resolves external-data paths against the CWD for in-memory
    # protos; give it a real payload file so the basic check can pass.
    (tmp_path / "weights.bin").write_bytes(b"\x00" * 12)
    monkeypatch.chdir(tmp_path)
    model = _valid_model()
    ext = numpy_helper.from_array(np.zeros(3, dtype=np.float32), name="w_ext")
    ext.data_location = onnx.TensorProto.EXTERNAL
    ext.ClearField("raw_data")
    entry = ext.external_data.add()
    entry.key = "location"
    entry.value = "weights.bin"
    model.graph.initializer.append(ext)

    outcome = run_fixers(model, [DropDropoutFixer()])
    assert outcome.validation == "basic"
    assert [a.fixer for a in outcome.applied] == ["drop_dropout"]
