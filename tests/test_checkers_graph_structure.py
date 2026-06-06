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


def _undef_out(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def _model(nodes: list, inputs: list, outputs: list, opset: int = 17) -> onnx.ModelProto:
    g = helper.make_graph(nodes, "g", inputs, outputs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    return m


class TestOpsetTooOld:
    def _model_with_opsets(self, opset_imports: list) -> onnx.ModelProto:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [inp], [out])
        m = helper.make_model(g, opset_imports=opset_imports)
        m.ir_version = 8
        return m

    def _opset_issues(self, model: onnx.ModelProto) -> list:
        return [i for i in GraphStructureChecker().check(model) if "floor" in i.message.lower()]

    def test_old_opset_warns(self) -> None:
        issues = self._opset_issues(self._model_with_opsets([helper.make_opsetid("", 11)]))
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert issues[0].operator == "Model"
        assert "11" in issues[0].message and "17" in issues[0].message

    def test_ai_onnx_domain_alias_counts(self) -> None:
        issues = self._opset_issues(self._model_with_opsets([helper.make_opsetid("ai.onnx", 13)]))
        assert len(issues) == 1

    def test_opset_at_floor_does_not_fire(self) -> None:
        assert self._opset_issues(self._model_with_opsets([helper.make_opsetid("", 17)])) == []

    def test_opset_above_floor_does_not_fire(self) -> None:
        assert self._opset_issues(self._model_with_opsets([helper.make_opsetid("", 18)])) == []
        assert self._opset_issues(self._model_with_opsets([helper.make_opsetid("", 26)])) == []

    def test_custom_domain_only_does_not_report_opset_zero(self) -> None:
        # No ai.onnx/'' opset -> resolves to 0; must NOT fire ("opset 0").
        issues = self._opset_issues(
            self._model_with_opsets([helper.make_opsetid("com.microsoft", 1)])
        )
        assert issues == []

    def test_multi_domain_default_at_floor_does_not_fire(self) -> None:
        issues = self._opset_issues(
            self._model_with_opsets(
                [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
            )
        )
        assert issues == []


class TestInputWithNoType:
    def _input_type_issues(self, model: onnx.ModelProto) -> list:
        return [i for i in GraphStructureChecker().check(model) if i.operator == "Input"]

    def test_input_with_no_typeproto_is_critical(self) -> None:
        typeless = _undef_out("x")  # ValueInfoProto with a name but no type set
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        model = _model([helper.make_node("Identity", ["x"], ["y"])], [typeless], [out])
        issues = self._input_type_issues(model)
        assert len(issues) == 1
        assert issues[0].severity is Severity.CRITICAL
        assert "element type" in issues[0].message.lower()

    def test_input_with_undefined_elem_type_is_critical(self) -> None:
        undef = helper.make_tensor_value_info("x", TensorProto.UNDEFINED, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        model = _model([helper.make_node("Identity", ["x"], ["y"])], [undef], [out])
        issues = self._input_type_issues(model)
        assert len(issues) == 1
        assert issues[0].severity is Severity.CRITICAL

    def test_normal_float_input_does_not_fire(self) -> None:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        model = _model([helper.make_node("Identity", ["x"], ["y"])], [inp], [out])
        assert self._input_type_issues(model) == []

    def test_sequence_input_is_not_flagged(self) -> None:
        # A sequence-typed input has a value oneof set (sequence_type) -> not a
        # "no element type" defect; must be skipped, not false-flagged.
        seq = onnx.ValueInfoProto()
        seq.name = "s"
        seq.type.sequence_type.elem_type.tensor_type.elem_type = TensorProto.FLOAT
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        model = _model([helper.make_node("Identity", ["x"], ["y"])], [seq], [out])
        assert self._input_type_issues(model) == []

    def test_typeless_initializer_dup_is_not_flagged(self) -> None:
        # opset<9 duplicates initializers into graph.input; those carry a real
        # dtype via the initializer, so a typeless input ValueInfo for an
        # initializer name must not fire.
        import numpy as np
        from onnx import numpy_helper

        w = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="w")
        typeless_dup = _undef_out("w")
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        g = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "g",
            [inp, typeless_dup],
            [out],
            initializer=[w],
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        assert self._input_type_issues(m) == []


class TestIsolatedNode:
    def _isolated(self, model: onnx.ModelProto) -> list:
        return [i for i in GraphStructureChecker().check(model) if "isolated" in i.message.lower()]

    def test_dead_node_warns(self) -> None:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        live = helper.make_node("Identity", ["x"], ["y"], name="live")
        dead = helper.make_node("Relu", ["x"], ["dead_out"], name="dead")
        model = _model([live, dead], [inp], [out])
        issues = self._isolated(model)
        assert len(issues) == 1
        assert issues[0].severity is Severity.WARNING
        assert issues[0].operator == "Relu"
        assert issues[0].node_name == "dead"

    def test_clean_model_has_no_isolated(self, clean_model: onnx.ModelProto) -> None:
        assert self._isolated(clean_model) == []

    def test_last_node_feeding_output_is_not_isolated(self) -> None:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        model = _model([helper.make_node("Identity", ["x"], ["y"], name="last")], [inp], [out])
        assert self._isolated(model) == []

    def test_outer_scope_capture_is_not_isolated(self) -> None:
        # 'cap' is produced top-level and consumed only inside the If then-branch.
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        prod = helper.make_node("Identity", ["x"], ["cap"], name="prod")
        then_g = helper.make_graph(
            [helper.make_node("Identity", ["cap"], ["t"], name="t")], "then", [], [_undef_out("t")]
        )
        else_g = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["e"], name="e")], "else", [], [_undef_out("e")]
        )
        cond = helper.make_node(
            "Constant",
            [],
            ["c"],
            name="c",
            value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
        )
        ifn = helper.make_node("If", ["c"], ["o"], name="i", then_branch=then_g, else_branch=else_g)
        model = _model([prod, cond, ifn], [inp], [_undef_out("o")])
        assert self._isolated(model) == []

    def test_multi_output_with_one_used_is_not_isolated(self) -> None:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        drop = helper.make_node("Dropout", ["x"], ["y", "mask"], name="drop")  # mask unused
        model = _model([drop], [inp], [out])
        assert self._isolated(model) == []

    def test_dead_node_inside_subgraph_warns(self) -> None:
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        dead_in_branch = helper.make_node("Relu", ["x"], ["branch_dead"], name="branch_dead")
        t_id = helper.make_node("Identity", ["x"], ["t"], name="t")
        then_g = helper.make_graph([dead_in_branch, t_id], "then", [], [_undef_out("t")])
        else_g = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["e"], name="e")], "else", [], [_undef_out("e")]
        )
        cond = helper.make_node(
            "Constant",
            [],
            ["c"],
            name="c",
            value=helper.make_tensor("c", TensorProto.BOOL, [], [True]),
        )
        ifn = helper.make_node("If", ["c"], ["o"], name="i", then_branch=then_g, else_branch=else_g)
        model = _model([cond, ifn], [inp], [_undef_out("o")])
        issues = self._isolated(model)
        assert any(i.node_name == "branch_dead" for i in issues)
