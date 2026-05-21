"""Build deterministic ONNX fixtures for the test suite.

Every model is constructed with onnx.helper -- no PyTorch dependency. Each
function returns a complete ModelProto with a one-line docstring describing
the failure mode it is designed to provoke.

Run from the repo root:

    python tests/fixtures/generate_broken.py

This regenerates every .onnx file under tests/fixtures/. The output is
committed to the repo so CI does not need to run this script.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

OUT_DIR = Path(__file__).parent
FAIL_DIR = OUT_DIR / "failing"
OPSET = 17
IR_VERSION = 8


def _save(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _make_model(graph: onnx.GraphProto, *, producer: str = "trtcheck-fixtures") -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        producer_name=producer,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    return model


# -- Clean baseline -----------------------------------------------------------


def create_clean_minimal() -> onnx.ModelProto:
    """Conv + Relu, fixed-shape FLOAT32 input. Should report zero issues."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 30, 30])

    weight = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32),
        name="conv_weight",
    )
    bias = numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="conv_bias")

    conv = helper.make_node(
        "Conv",
        ["input", "conv_weight", "conv_bias"],
        ["conv_out"],
        name="conv_1",
        kernel_shape=[3, 3],
    )
    relu = helper.make_node("Relu", ["conv_out"], ["output"], name="relu_1")

    graph = helper.make_graph(
        nodes=[conv, relu],
        name="clean_minimal",
        inputs=[inp],
        outputs=[out],
        initializer=[weight, bias],
    )
    return _make_model(graph)


# -- Failure: SequenceEmpty (PyTorch List[Tensor] pattern) --------------------


def create_sequence_empty() -> onnx.ModelProto:
    """Uses SequenceEmpty -> SequenceInsert -> SequenceAt. Critical for TRT."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    seq_empty = helper.make_node("SequenceEmpty", [], ["seq0"], name="seq_empty")
    seq_insert = helper.make_node(
        "SequenceInsert", ["seq0", "input"], ["seq1"], name="seq_insert"
    )
    # position is required as input for SequenceAt in opset 17
    pos_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="pos")
    seq_at = helper.make_node("SequenceAt", ["seq1", "pos"], ["output"], name="seq_at")

    graph = helper.make_graph(
        nodes=[seq_empty, seq_insert, seq_at],
        name="sequence_empty_failure",
        inputs=[inp],
        outputs=[out],
        initializer=[pos_init],
    )
    return _make_model(graph)


# -- Failure: INT64 weights ---------------------------------------------------


def create_int64_weights() -> onnx.ModelProto:
    """A Gather op whose indices are baked-in as INT64 constants."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])

    idx = numpy_helper.from_array(np.array([0, 2, 5], dtype=np.int64), name="indices")

    gather = helper.make_node(
        "Gather", ["input", "indices"], ["output"], name="gather_1", axis=0
    )

    graph = helper.make_graph(
        nodes=[gather],
        name="int64_weights_failure",
        inputs=[inp],
        outputs=[out],
        initializer=[idx],
    )
    return _make_model(graph)


# -- Failure: fully dynamic input shape --------------------------------------


def create_fully_dynamic() -> onnx.ModelProto:
    """Every input dim is a symbol (batch, channels, h, w). TRT can build
    but cannot estimate memory or fuse well."""
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", "channels", "h", "w"]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", "channels", "h", "w"]
    )

    identity = helper.make_node("Identity", ["input"], ["output"], name="identity_1")

    graph = helper.make_graph(
        nodes=[identity],
        name="fully_dynamic_failure",
        inputs=[inp],
        outputs=[out],
    )
    return _make_model(graph)


# -- Failure: UINT8 graph input ----------------------------------------------


def create_uint8_input() -> onnx.ModelProto:
    """UINT8 image input that gets Cast to FLOAT inside the model."""
    inp = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 3, 32, 32])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])

    cast = helper.make_node("Cast", ["input"], ["output"], name="cast_1", to=TensorProto.FLOAT)

    graph = helper.make_graph(
        nodes=[cast],
        name="uint8_input_failure",
        inputs=[inp],
        outputs=[out],
    )
    return _make_model(graph)


# -- Failure: Loop with dynamic trip count -----------------------------------


def create_control_flow_loop() -> onnx.ModelProto:
    """Loop whose trip count is read from a runtime input tensor."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    state_in = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1])
    state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1])

    # Body subgraph: identity-like loop body
    body_iter = helper.make_tensor_value_info("body_iter", TensorProto.INT64, [])
    body_cond_in = helper.make_tensor_value_info("body_cond", TensorProto.BOOL, [])
    body_state_in = helper.make_tensor_value_info("body_state_in", TensorProto.FLOAT, [1])
    body_cond_out = helper.make_tensor_value_info("body_cond_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])

    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    body_add = helper.make_node("Add", ["body_state_in", "one"], ["body_state_out"], name="body_add")
    body_id = helper.make_node("Identity", ["body_cond"], ["body_cond_out"], name="body_cond_id")

    body_graph = helper.make_graph(
        nodes=[body_add, body_id],
        name="loop_body",
        inputs=[body_iter, body_cond_in, body_state_in],
        outputs=[body_cond_out, body_state_out],
        initializer=[one],
    )

    loop = helper.make_node(
        "Loop",
        inputs=["trip_count", "cond", "state_in"],
        outputs=["state_out"],
        name="loop_1",
        body=body_graph,
    )

    graph = helper.make_graph(
        nodes=[loop],
        name="control_flow_loop_failure",
        inputs=[trip_count, cond, state_in],
        outputs=[state_out],
    )
    return _make_model(graph)


# -- Driver ------------------------------------------------------------------


_CLEAN: dict[str, callable] = {
    "clean_minimal.onnx": create_clean_minimal,
}

_FAILING: dict[str, callable] = {
    "sequence_empty.onnx": create_sequence_empty,
    "int64_weights.onnx": create_int64_weights,
    "fully_dynamic.onnx": create_fully_dynamic,
    "uint8_input.onnx": create_uint8_input,
    "control_flow_loop.onnx": create_control_flow_loop,
}


def main() -> None:
    np.random.seed(0)
    for name, factory in _CLEAN.items():
        _save(factory(), OUT_DIR / name)
        print(f"wrote {OUT_DIR / name}")
    for name, factory in _FAILING.items():
        _save(factory(), FAIL_DIR / name)
        print(f"wrote {FAIL_DIR / name}")


if __name__ == "__main__":
    main()
