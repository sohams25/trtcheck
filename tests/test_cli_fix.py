"""CLI tests for the audited --fix pipeline and verdict-driven exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from trtcheck.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _save(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, str(path))
    return path


def _fixable_model() -> onnx.ModelProto:
    """Gather with INT64 indices (safe to convert) plus an inference Dropout."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])
    idx = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="indices")
    gather = helper.make_node("Gather", ["input", "indices"], ["g"], name="g0", axis=0)
    drop = helper.make_node("Dropout", ["g"], ["d"], name="drop")
    ident = helper.make_node("Identity", ["d"], ["output"], name="ident")
    graph = helper.make_graph([gather, drop, ident], "m", [inp], [out], initializer=[idx])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _clean_model() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    relu = helper.make_node("Relu", ["input"], ["output"], name="r")
    graph = helper.make_graph([relu], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestFixPipeline:
    def test_successful_fix_writes_and_reports_deltas(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        dst = tmp_path / "out.onnx"
        result = runner.invoke(main, [str(src), "--fix", "--output", str(dst)])
        assert result.exit_code == 0, result.output
        assert dst.exists()
        assert "int64_to_int32" in result.output
        assert "drop_dropout" in result.output
        assert "verdict:" in result.output and "resolved" in result.output
        fixed = onnx.load(str(dst))
        onnx.checker.check_model(fixed, full_check=True)
        assert not any(n.op_type == "Dropout" for n in fixed.graph.node)

    def test_fix_uses_selected_target(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        dst = tmp_path / "out.onnx"
        result = runner.invoke(
            main, [str(src), "--fix", "--target-trt", "8.6", "--output", str(dst)]
        )
        assert result.exit_code == 0, result.output
        assert "TensorRT 8.6" in result.output

    def test_fix_json_summary_is_machine_readable(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        dst = tmp_path / "out.onnx"
        result = runner.invoke(main, [str(src), "--fix", "--format", "json", "--output", str(dst)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["target_trt"] == "10.3"
        assert payload["schema_version"] == "2.0"
        assert {f["fixer"] for f in payload["fixes_applied"]} >= {"int64_to_int32"}
        assert isinstance(payload["resolved"], list)
        assert isinstance(payload["remaining"], list)
        assert isinstance(payload["introduced"], list)
        assert payload["verdict_before"] in ("blocked", "unverified", "likely")
        assert payload["verdict_after"] in ("blocked", "unverified", "likely")

    def test_dry_run_writes_nothing(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        result = runner.invoke(main, [str(src), "--fix", "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "dry run" in result.output
        assert list(tmp_path.glob("*.onnx")) == [src]

    def test_noop_fix_reports_and_writes_nothing(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _save(_clean_model(), tmp_path / "in.onnx")
        dst = tmp_path / "out.onnx"
        result = runner.invoke(main, [str(src), "--fix", "--output", str(dst)])
        assert result.exit_code == 0, result.output
        assert "no fixes applied" in result.output
        assert not dst.exists()

    def test_invalid_input_model_is_refused(self, runner: CliRunner, tmp_path: Path) -> None:
        model = _clean_model()
        model.graph.node[0].input[0] = "missing_tensor"
        src = _save(model, tmp_path / "bad.onnx")
        result = runner.invoke(main, [str(src), "--fix", "--dry-run"])
        assert result.exit_code != 0
        assert "failed ONNX validation" in result.output

    def test_crashing_plugin_fixer_cannot_corrupt_output(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trtcheck import cli as cli_mod
        from trtcheck.fixers import FixApplied

        class EvilFixer:
            name = "evil"

            def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
                for node in model.graph.node:
                    node.op_type = "Bogus"
                raise RuntimeError("kaboom")

        monkeypatch.setattr(cli_mod, "load_plugins", lambda: ([], [EvilFixer()], []))
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        dst = tmp_path / "out.onnx"
        result = runner.invoke(main, [str(src), "--fix", "--output", str(dst)])
        assert result.exit_code == 0, result.output
        assert "kaboom" in result.output or "evil" in result.output  # warned on stderr
        fixed = onnx.load(str(dst))
        assert not any(n.op_type == "Bogus" for n in fixed.graph.node)
        onnx.checker.check_model(fixed, full_check=True)

    def test_refuses_to_overwrite_input(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _save(_fixable_model(), tmp_path / "in.onnx")
        result = runner.invoke(main, [str(src), "--fix", "--output", str(src)])
        assert result.exit_code != 0
        assert "refusing to overwrite the input file" in result.output


class TestExitCodes:
    def test_unverified_model_exits_zero_by_default(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        model = _clean_model()
        model.graph.node[0].op_type = "TotallyNovelOp"
        src = _save(model, tmp_path / "m.onnx")
        result = runner.invoke(main, [str(src)])
        assert result.exit_code == 0, result.output
        assert "UNVERIFIED" in result.output

    def test_fail_on_unverified_exits_one(self, runner: CliRunner, tmp_path: Path) -> None:
        model = _clean_model()
        model.graph.node[0].op_type = "TotallyNovelOp"
        src = _save(model, tmp_path / "m.onnx")
        result = runner.invoke(main, [str(src), "--fail-on", "unverified"])
        assert result.exit_code == 1

    def test_severity_filter_does_not_change_exit_code(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--severity trims the display only; hiding the unverified INFO
        finding must not flip a --fail-on unverified failure into a pass."""
        model = _clean_model()
        model.graph.node[0].op_type = "TotallyNovelOp"
        src = _save(model, tmp_path / "m.onnx")
        result = runner.invoke(
            main, [str(src), "--severity", "critical", "--fail-on", "unverified"]
        )
        assert result.exit_code == 1
