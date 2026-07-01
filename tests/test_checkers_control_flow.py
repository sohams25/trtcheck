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


def _computed_trip_loop_model() -> onnx.ModelProto:
    # Loop whose trip count comes from a Constant node's output: not an
    # initializer, not a graph input -- an internally computed value.
    body_iter = helper.make_tensor_value_info("body_iter", TensorProto.INT64, [])
    body_cond_in = helper.make_tensor_value_info("body_cond", TensorProto.BOOL, [])
    body_state_in = helper.make_tensor_value_info("body_state_in", TensorProto.FLOAT, [1])
    body_cond_out = helper.make_tensor_value_info("body_cond_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["body_state_in"], ["body_state_out"], name="b_id"),
            helper.make_node("Identity", ["body_cond"], ["body_cond_out"], name="b_cid"),
        ],
        "computed_trip_body",
        [body_iter, body_cond_in, body_state_in],
        [body_cond_out, body_state_out],
    )

    trip_const = helper.make_node(
        "Constant",
        [],
        ["trip_from_const"],
        name="trip_const",
        value=numpy_helper.from_array(np.array(4, dtype=np.int64)),
    )
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    state_in = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1])
    state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1])
    loop = helper.make_node(
        "Loop",
        ["trip_from_const", "cond", "state_in"],
        ["state_out"],
        name="computed_trip_loop",
        body=body,
    )
    graph = helper.make_graph(
        [trip_const, loop],
        "computed_trip_model",
        [cond, state_in],
        [state_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestControlFlowChecker:
    def test_clean_model_no_issues(self, clean_model: onnx.ModelProto) -> None:
        assert ControlFlowChecker().check(clean_model) == []

    def test_loop_with_runtime_trip_count_is_critical(
        self, control_flow_loop_model: onnx.ModelProto
    ) -> None:
        # Trip count fed from a graph input is provably runtime-dynamic; TRT
        # rejects it at engine build, so this must fail the verdict.
        issues = ControlFlowChecker().check(control_flow_loop_model)
        assert any(
            i.severity is Severity.CRITICAL and i.operator == "Loop" for i in issues
        ), "Loop with a graph-input trip count should be critical"
        for i in issues:
            assert i.category is CheckCategory.CONTROL_FLOW

    def test_loop_with_computed_trip_count_warns(self) -> None:
        # Trip count produced by an internal node may still be shape-inferable
        # by TRT, so it stays a warning -- escalating it would cry wolf.
        issues = ControlFlowChecker().check(_computed_trip_loop_model())
        loop_issues = [i for i in issues if i.operator == "Loop"]
        assert loop_issues, "computed trip count should still be flagged"
        assert all(i.severity is Severity.WARNING for i in loop_issues)

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
