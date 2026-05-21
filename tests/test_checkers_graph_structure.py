"""Tests for GraphStructureChecker."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtcheck.checkers.graph_structure import GraphStructureChecker
from trtcheck.types import CheckCategory, Severity


def _empty_outputs_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    ident = helper.make_node("Identity", ["input"], ["unused"], name="ident")
    graph = helper.make_graph(
        nodes=[ident],
        name="no_output",
        inputs=[inp],
        outputs=[],  # deliberately empty
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _duplicate_node_names_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    a = helper.make_node("Identity", ["input"], ["mid"], name="duplicate")
    b = helper.make_node("Identity", ["mid"], ["output"], name="duplicate")
    graph = helper.make_graph(nodes=[a, b], name="dup", inputs=[inp], outputs=[out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _large_constant_model() -> onnx.ModelProto:
    # ~12MB float32 constant (3M floats * 4 bytes)
    big = np.zeros((3_000_000,), dtype=np.float32)
    big_tensor = numpy_helper.from_array(big, name="big_constant")
    const = helper.make_node("Constant", [], ["big"], name="big_const", value=big_tensor)
    out = helper.make_tensor_value_info("big", TensorProto.FLOAT, list(big.shape))
    graph = helper.make_graph(nodes=[const], name="big", inputs=[], outputs=[out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestGraphStructureChecker:
    def test_clean_model_produces_no_issues(self, clean_model: onnx.ModelProto) -> None:
        issues = GraphStructureChecker().check(clean_model)
        assert issues == []

    def test_missing_output_is_critical(self) -> None:
        issues = GraphStructureChecker().check(_empty_outputs_model())
        critical = [i for i in issues if i.severity is Severity.CRITICAL]
        assert any("output" in i.message.lower() for i in critical)
        assert all(i.category is CheckCategory.GRAPH_STRUCTURE for i in issues)

    def test_duplicate_node_names_emit_warning(self) -> None:
        issues = GraphStructureChecker().check(_duplicate_node_names_model())
        warnings = [i for i in issues if i.severity is Severity.WARNING]
        assert any(
            "duplicate" in i.message.lower() or "duplicate" in i.operator.lower() for i in warnings
        )

    def test_large_constant_emits_info(self) -> None:
        issues = GraphStructureChecker().check(_large_constant_model())
        infos = [i for i in issues if i.severity is Severity.INFO]
        assert any("constant" in i.message.lower() for i in infos)

    def test_checker_has_name_attribute(self) -> None:
        assert isinstance(GraphStructureChecker.name, str)
        assert GraphStructureChecker.name
