"""Conditional-support rule evaluation (matrix schema 2.x `conditions`)."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.checkers.operator_support import (
    RULE_CONDITION,
    RULE_CONDITION_UNRESOLVED,
    OperatorSupportChecker,
)
from trtcheck.types import Confidence, Severity


def _topk_model(*, sorted_attr: int | None = None, k_value: int | None = 3, k_dynamic=False):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [100])]
    inits = []
    if k_dynamic:
        inputs.append(helper.make_tensor_value_info("k", TensorProto.INT64, [1]))
    else:
        inits.append(numpy_helper.from_array(np.array([k_value], dtype=np.int64), name="k"))
    kwargs = {} if sorted_attr is None else {"sorted": sorted_attr}
    topk = helper.make_node("TopK", ["x", "k"], ["vals", "idxs"], name="tk", axis=0, **kwargs)
    outs = [
        onnx.ValueInfoProto(name="vals"),
        onnx.ValueInfoProto(name="idxs"),
    ]
    graph = helper.make_graph([topk], "m", inputs, outs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _resize_model(**attrs):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], name="rz", **attrs)
    graph = helper.make_graph(
        [node], "m", [x], [onnx.ValueInfoProto(name="y")], initializer=[scales]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _issues(model, target="10.3"):
    return OperatorSupportChecker(target_trt=target).check(model)


class TestTopKConditions:
    def test_default_sorted_and_small_constant_k_pass(self) -> None:
        issues = _issues(_topk_model())
        assert all(i.rule_id not in (RULE_CONDITION,) for i in issues)

    def test_sorted_zero_is_a_violation(self) -> None:
        issues = _issues(_topk_model(sorted_attr=0))
        hit = next(i for i in issues if i.rule_id == RULE_CONDITION)
        assert hit.severity is Severity.CRITICAL
        assert "sorted" in hit.message
        assert hit.confidence is Confidence.HIGH  # official-docs evidence
        assert hit.docs_link and "onnx-tensorrt" in hit.docs_link

    def test_k_above_limit_is_a_violation(self) -> None:
        issues = _issues(_topk_model(k_value=5000))
        hit = next(i for i in issues if i.rule_id == RULE_CONDITION)
        assert "3840" in hit.message
        assert hit.severity is Severity.CRITICAL

    def test_dynamic_k_is_unresolved_not_violated(self) -> None:
        issues = _issues(_topk_model(k_dynamic=True))
        assert all(i.rule_id != RULE_CONDITION for i in issues)
        hit = next(i for i in issues if i.rule_id == RULE_CONDITION_UNRESOLVED)
        assert hit.severity is Severity.INFO
        assert hit.verify_required is True

    def test_conditions_do_not_apply_to_other_targets(self) -> None:
        # The evidence tracks TRT 10.x; an 8.6 target must not inherit it.
        issues = _issues(_topk_model(sorted_attr=0), target="8.6")
        assert all(i.rule_id not in (RULE_CONDITION, RULE_CONDITION_UNRESOLVED) for i in issues)


class TestResizeConditions:
    def test_cubic_mode_is_a_violation(self) -> None:
        issues = _issues(_resize_model(mode="cubic"))
        hit = next(i for i in issues if i.rule_id == RULE_CONDITION)
        assert hit.severity is Severity.CRITICAL
        assert "cubic" in hit.message or "mode" in hit.message

    def test_nearest_mode_passes(self) -> None:
        issues = _issues(_resize_model(mode="nearest"))
        assert all(i.rule_id != RULE_CONDITION for i in issues)

    def test_antialias_is_a_violation(self) -> None:
        issues = _issues(_resize_model(mode="linear", antialias=1))
        assert any(i.rule_id == RULE_CONDITION and "antialias" in i.message for i in issues)

    def test_unsupported_coord_transform_is_a_violation(self) -> None:
        issues = _issues(
            _resize_model(mode="linear", coordinate_transformation_mode="tf_crop_and_resize")
        )
        assert any(
            i.rule_id == RULE_CONDITION and "coordinate_transformation_mode" in i.message
            for i in issues
        )
