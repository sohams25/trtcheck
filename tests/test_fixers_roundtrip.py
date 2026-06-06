"""Round-trip contract for the built-in fixers.

Every fixer must satisfy three properties on a model it acts on:

  (a) **valid** -- the rewritten model passes ``onnx.checker.check_model`` with
      ``full_check=True`` (i.e. type/shape inference), not merely the shallow
      structural check the per-fixer suites use;
  (b) **resolves** -- the issue class the fixer targets is gone when the fixed
      model is re-analyzed; and
  (c) **no regression** -- the fix introduces no NEW critical issue.

These properties were previously unguarded. The per-fixer suites assert
validity only at the shallow ``check_model`` (``full_check=False``) level and
never re-analyze, so a fixer could silently emit a type-inference-invalid model
(it did -- see ``test_int64_fixer_updates_shadowed_input_dtype``) or quietly
become a no-op without any test failing.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.analyzer import Analyzer, AnalyzerConfig
from trtcheck.fixers import apply_all, default_fixers
from trtcheck.types import CheckCategory, Issue, Severity


def assert_valid_model(model: onnx.ModelProto) -> None:
    """Full validity: structural check + type/shape inference.

    ``full_check=True`` runs onnx's inference, which is what catches a fixer
    that rewrites an initializer's dtype but leaves a same-named graph input
    declaring the old dtype. The shallow default check_model misses that class
    of corruption entirely.
    """
    onnx.checker.check_model(model, full_check=True)


def _analyze(model: onnx.ModelProto) -> list[Issue]:
    # Disable entry-point plugin discovery so the local environment's installed
    # plugins can't perturb the issue set this test reasons about.
    analyzer = Analyzer(AnalyzerConfig(discover_entry_point_plugins=False))
    return analyzer.analyze_model(model).issues


def _critical_keys(issues: list[Issue]) -> set[tuple[str, str, str]]:
    return {(i.node_name, i.operator, i.message) for i in issues if i.severity is Severity.CRITICAL}


# --------------------------------------------------------------------------- #
# (a) validity -- the shadowed-input bugs the robustness sweep confirmed       #
# --------------------------------------------------------------------------- #


def _int64_initializer_shadows_input() -> onnx.ModelProto:
    """Legal ONNX: an INT64 initializer whose name also appears in graph.input.

    The initializer supplies the input's default value. ``Shape`` decouples the
    output type from the input dtype (Shape always yields INT64), so casting the
    initializer/input to INT32 leaves a model that is *fully* valid -- provided
    the fixer also retypes the shadowing input.
    """
    wi = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), name="wi")
    graph = helper.make_graph(
        [helper.make_node("Shape", ["wi"], ["shp"], name="shape0")],
        "g",
        [helper.make_tensor_value_info("wi", TensorProto.INT64, [3])],
        [helper.make_tensor_value_info("shp", TensorProto.INT64, [1])],
        initializer=[wi],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _float64_initializer_shadows_input() -> onnx.ModelProto:
    wf = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float64), name="wf")
    graph = helper.make_graph(
        [helper.make_node("Shape", ["wf"], ["shp"], name="shape0")],
        "g",
        [helper.make_tensor_value_info("wf", TensorProto.DOUBLE, [3])],
        [helper.make_tensor_value_info("shp", TensorProto.INT64, [1])],
        initializer=[wf],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_int64_fixer_updates_shadowed_input_dtype() -> None:
    model = _int64_initializer_shadows_input()
    # Premise: the input model is genuinely valid under full type inference.
    assert_valid_model(model)

    fixed, applied = apply_all(model, default_fixers())
    assert [a.fixer for a in applied] == ["int64_to_int32"]

    # The initializer was cast...
    new_init = next(i for i in fixed.graph.initializer if i.name == "wi")
    assert new_init.data_type == TensorProto.INT32
    # ...and the shadowing graph input must be retyped to match, or full type
    # inference rejects the model.
    new_input = next(i for i in fixed.graph.input if i.name == "wi")
    assert new_input.type.tensor_type.elem_type == TensorProto.INT32
    assert_valid_model(fixed)


def test_float64_fixer_updates_shadowed_input_dtype() -> None:
    model = _float64_initializer_shadows_input()
    assert_valid_model(model)

    fixed, applied = apply_all(model, default_fixers())
    assert [a.fixer for a in applied] == ["float64_to_float32"]

    new_init = next(i for i in fixed.graph.initializer if i.name == "wf")
    assert new_init.data_type == TensorProto.FLOAT
    new_input = next(i for i in fixed.graph.input if i.name == "wf")
    assert new_input.type.tensor_type.elem_type == TensorProto.FLOAT
    assert_valid_model(fixed)


# --------------------------------------------------------------------------- #
# (b) resolves -- re-analyzing a fixed model no longer reports the target      #
# --------------------------------------------------------------------------- #


def _int64_weights_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])
    idx = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="indices")
    gather = helper.make_node("Gather", ["input", "indices"], ["output"], name="g", axis=0)
    graph = helper.make_graph([gather], "m", [inp], [out], initializer=[idx])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _float64_initializer_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3])
    wf = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="bias")
    wd = numpy_helper.from_array(np.array([0.5, 0.5, 0.5], dtype=np.float64), name="dbl")
    # 'dbl' (DOUBLE) is the flagged initializer; route it through Shape so the
    # post-fix model stays valid regardless of its dtype.
    add = helper.make_node("Add", ["input", "bias"], ["output"], name="add0")
    shp = helper.make_node("Shape", ["dbl"], ["shp"], name="shape0")
    out2 = helper.make_tensor_value_info("shp", TensorProto.INT64, [1])
    graph = helper.make_graph([add, shp], "m", [inp], [out, out2], initializer=[wf, wd])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _uint8_cast_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    cast = helper.make_node("Cast", ["input"], ["casted"], name="cast_1", to=TensorProto.FLOAT)
    ident = helper.make_node("Identity", ["casted"], ["output"], name="ident")
    graph = helper.make_graph([cast, ident], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _upsample_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )
    up = helper.make_node("Upsample", ["input", "scales"], ["output"], name="up_1", mode="nearest")
    graph = helper.make_graph([up], "m", [inp], [out], initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def test_int64_fix_resolves_precision_issue() -> None:
    model = _int64_weights_model()
    before = [
        i for i in _analyze(model) if i.category is CheckCategory.PRECISION and "INT64" in i.message
    ]
    assert before, "fixture should be flagged for an INT64 initializer"
    fixed, _ = apply_all(model, default_fixers())
    assert_valid_model(fixed)
    after = [
        i for i in _analyze(fixed) if i.category is CheckCategory.PRECISION and "INT64" in i.message
    ]
    assert not after, "INT64 precision issue must be gone after the fix"


def test_float64_fix_resolves_precision_issue() -> None:
    model = _float64_initializer_model()
    before = [
        i
        for i in _analyze(model)
        if i.category is CheckCategory.PRECISION and "DOUBLE" in i.message
    ]
    assert before, "fixture should be flagged for a DOUBLE initializer"
    fixed, _ = apply_all(model, default_fixers())
    assert_valid_model(fixed)
    after = [
        i
        for i in _analyze(fixed)
        if i.category is CheckCategory.PRECISION and "DOUBLE" in i.message
    ]
    assert not after, "DOUBLE precision issue must be gone after the fix"


def test_uint8_fix_resolves_precision_issue() -> None:
    model = _uint8_cast_model()
    before = [
        i for i in _analyze(model) if i.category is CheckCategory.PRECISION and "UINT8" in i.message
    ]
    assert before, "fixture should be flagged for a UINT8 input"
    fixed, _ = apply_all(model, default_fixers())
    assert_valid_model(fixed)
    after = [
        i for i in _analyze(fixed) if i.category is CheckCategory.PRECISION and "UINT8" in i.message
    ]
    assert not after, "UINT8 precision issue must be gone after the fix"


def test_upsample_fix_emits_valid_resize() -> None:
    # trtcheck's matrix treats Upsample as convertible, so it is not flagged as
    # an issue -- the upsample fixer's contract is purely the transform: replace
    # the legacy Upsample with a Resize that passes full type/shape inference.
    model = _upsample_model()
    fixed, applied = apply_all(model, default_fixers())
    assert any(a.fixer == "upsample_to_resize" for a in applied)
    op_types = [n.op_type for n in fixed.graph.node]
    assert "Upsample" not in op_types and "Resize" in op_types
    assert_valid_model(fixed)


# --------------------------------------------------------------------------- #
# (c) no regression -- a composed multi-fixer model adds no NEW critical       #
# --------------------------------------------------------------------------- #


def _composed_model() -> onnx.ModelProto:
    """INT64 initializer + DOUBLE initializer + an interior Dropout.

    Dropout's input is an interior value (Relu output), not a graph input, so
    the dropout fixer actually removes it rather than refusing a pass-through.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3])
    bias = numpy_helper.from_array(
        np.array([1.0, 2.0, 3.0], dtype=np.float64), name="bias"
    )  # DOUBLE
    idx = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="idx")  # INT64

    relu = helper.make_node("Relu", ["input"], ["r"], name="relu0")
    drop = helper.make_node("Dropout", ["r"], ["d"], name="drop0")
    ident = helper.make_node("Identity", ["d"], ["output"], name="ident0")
    shp_b = helper.make_node("Shape", ["bias"], ["shp_b"], name="shape_b")
    shp_i = helper.make_node("Shape", ["idx"], ["shp_i"], name="shape_i")
    out_b = helper.make_tensor_value_info("shp_b", TensorProto.INT64, [1])
    out_i = helper.make_tensor_value_info("shp_i", TensorProto.INT64, [1])
    graph = helper.make_graph(
        [relu, drop, ident, shp_b, shp_i],
        "m",
        [inp],
        [out, out_b, out_i],
        initializer=[bias, idx],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_composed_fixers_introduce_no_new_critical() -> None:
    model = _composed_model()
    before = _critical_keys(_analyze(model))
    fixed, applied = apply_all(model, default_fixers())
    assert_valid_model(fixed)
    after = _critical_keys(_analyze(fixed))
    new_criticals = after - before
    assert not new_criticals, f"fixers introduced new critical issues: {new_criticals}"


def test_composed_fixers_emit_valid_model_and_apply_each() -> None:
    model = _composed_model()
    fixed, applied = apply_all(model, default_fixers())
    fixers = {a.fixer for a in applied}
    # All three relevant fixers ran on the composed model.
    assert {"int64_to_int32", "float64_to_float32", "drop_dropout"} <= fixers
    assert_valid_model(fixed)
    op_types = [n.op_type for n in fixed.graph.node]
    assert "Dropout" not in op_types
