"""Tests for ControlFlowChecker."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.checkers.control_flow import ControlFlowChecker
from trtcheck.types import CheckCategory, Severity


def _scan_model() -> onnx.ModelProto:
    body_in = helper.make_tensor_value_info("body_in", TensorProto.FLOAT, [4])
    body_out = helper.make_tensor_value_info("body_out", TensorProto.FLOAT, [4])
    ident = helper.make_node("Identity", ["body_in"], ["body_out"], name="body_ident")
    body = helper.make_graph([ident], "body", [body_in], [body_out])

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])
    scan = helper.make_node(
        "Scan",
        ["input"],
        ["output"],
        name="scan_1",
        body=body,
        num_scan_inputs=1,
    )
    graph = helper.make_graph([scan], "scan_model", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _nested_loop_model() -> onnx.ModelProto:
    # Outer loop with an inner loop in the body. Both have static trip counts.
    # We just want the topology detected as nested.
    inner_iter = helper.make_tensor_value_info("inner_iter", TensorProto.INT64, [])
    inner_cond = helper.make_tensor_value_info("inner_cond", TensorProto.BOOL, [])
    inner_state = helper.make_tensor_value_info("inner_state", TensorProto.FLOAT, [1])
    inner_cond_out = helper.make_tensor_value_info("inner_cond_out", TensorProto.BOOL, [])
    inner_state_out = helper.make_tensor_value_info("inner_state_out", TensorProto.FLOAT, [1])
    inner_body = helper.make_graph(
        [
            helper.make_node("Identity", ["inner_state"], ["inner_state_out"], name="i_id"),
            helper.make_node("Identity", ["inner_cond"], ["inner_cond_out"], name="i_cid"),
        ],
        "inner_body",
        [inner_iter, inner_cond, inner_state],
        [inner_cond_out, inner_state_out],
    )

    outer_iter = helper.make_tensor_value_info("outer_iter", TensorProto.INT64, [])
    outer_cond = helper.make_tensor_value_info("outer_cond", TensorProto.BOOL, [])
    outer_state = helper.make_tensor_value_info("outer_state", TensorProto.FLOAT, [1])
    outer_cond_out = helper.make_tensor_value_info("outer_cond_out", TensorProto.BOOL, [])
    outer_state_out = helper.make_tensor_value_info("outer_state_out", TensorProto.FLOAT, [1])
    trip = numpy_helper.from_array(np.array(2, dtype=np.int64), name="inner_trip")
    cond_t = numpy_helper.from_array(np.array(True), name="inner_cond_init")
    inner_loop = helper.make_node(
        "Loop",
        ["inner_trip", "inner_cond_init", "outer_state"],
        ["outer_state_out"],
        name="inner_loop",
        body=inner_body,
    )
    outer_body = helper.make_graph(
        [
            inner_loop,
            helper.make_node("Identity", ["outer_cond"], ["outer_cond_out"], name="o_cid"),
        ],
        "outer_body",
        [outer_iter, outer_cond, outer_state],
        [outer_cond_out, outer_state_out],
        initializer=[trip, cond_t],
    )

    top_trip = helper.make_tensor_value_info("top_trip", TensorProto.INT64, [])
    top_cond = helper.make_tensor_value_info("top_cond", TensorProto.BOOL, [])
    top_state = helper.make_tensor_value_info("top_state", TensorProto.FLOAT, [1])
    top_state_out = helper.make_tensor_value_info("top_state_out", TensorProto.FLOAT, [1])
    outer_loop = helper.make_node(
        "Loop",
        ["top_trip", "top_cond", "top_state"],
        ["top_state_out"],
        name="outer_loop",
        body=outer_body,
    )
    graph = helper.make_graph(
        [outer_loop],
        "nested",
        [top_trip, top_cond, top_state],
        [top_state_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestControlFlowChecker:
    def test_clean_model_no_issues(self, clean_model: onnx.ModelProto) -> None:
        assert ControlFlowChecker().check(clean_model) == []

    def test_loop_with_dynamic_trip_emits_warning(
        self, control_flow_loop_model: onnx.ModelProto
    ) -> None:
        issues = ControlFlowChecker().check(control_flow_loop_model)
        assert any(
            i.severity is Severity.WARNING and i.operator == "Loop" for i in issues
        ), "Loop with runtime trip count should warn"
        for i in issues:
            assert i.category is CheckCategory.CONTROL_FLOW

    def test_nested_loop_is_critical(self) -> None:
        issues = ControlFlowChecker().check(_nested_loop_model())
        criticals = [i for i in issues if i.severity is Severity.CRITICAL]
        assert any("nested" in i.message.lower() for i in criticals)

    def test_scan_emits_warning(self) -> None:
        issues = ControlFlowChecker().check(_scan_model())
        assert any(i.severity is Severity.WARNING and i.operator == "Scan" for i in issues)

    def test_target_version_affects_loop_severity(
        self, control_flow_loop_model: onnx.ModelProto
    ) -> None:
        # Loop is partial on 8.0 too; checker should still emit (the dynamic
        # trip count is the actual concern, not the version)
        issues_old = ControlFlowChecker(target_trt="8.0").check(control_flow_loop_model)
        issues_new = ControlFlowChecker(target_trt="10.3").check(control_flow_loop_model)
        assert issues_old and issues_new
