"""Runtime verification module -- all subprocess behavior mocked.

No test here requires TensorRT, a GPU, or network access.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from trtcheck import runtime_verify
from trtcheck.cli import main
from trtcheck.runtime_verify import RuntimeStatus, verify_model


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> dict:
    seen: dict = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(runtime_verify.subprocess, "run", fake_run)
    return seen


class TestVerifyModel:
    def test_missing_trtexec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: None)
        result = verify_model("model.onnx")
        assert result.status is RuntimeStatus.MISSING_TRTEXEC
        assert not result.verified

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")
        seen = _patch_run(
            monkeypatch,
            _FakeProc(0, stdout="[I] TensorRT version: 10.3.0\n[I] Engine built"),
        )
        result = verify_model("model.onnx", timeout_s=42)
        assert result.status is RuntimeStatus.SUCCESS
        assert result.verified
        assert result.trtexec_version and "TensorRT" in result.trtexec_version
        # No shell, list-args invocation, timeout honored.
        assert seen["cmd"][0] == "/usr/bin/trtexec"
        assert seen["cmd"][1] == "--onnx=model.onnx"
        assert seen["kwargs"]["timeout"] == 42
        assert "shell" not in seen["kwargs"] or seen["kwargs"]["shell"] is False

    def test_parser_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")
        _patch_run(
            monkeypatch,
            _FakeProc(1, stderr="[E] ModelImporter.cpp: Failed to parse ONNX model"),
        )
        result = verify_model("model.onnx")
        assert result.status is RuntimeStatus.PARSER_FAILURE

    def test_build_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")
        _patch_run(monkeypatch, _FakeProc(1, stderr="[E] Error: out of workspace memory"))
        result = verify_model("model.onnx")
        assert result.status is RuntimeStatus.BUILD_FAILURE

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")

        def raise_timeout(cmd, **kwargs):  # noqa: ANN001
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

        monkeypatch.setattr(runtime_verify.subprocess, "run", raise_timeout)
        result = verify_model("model.onnx", timeout_s=1)
        assert result.status is RuntimeStatus.TIMEOUT

    def test_output_tails_are_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")
        _patch_run(monkeypatch, _FakeProc(0, stdout="x" * 100_000))
        result = verify_model("model.onnx")
        assert len(result.stdout_tail) <= 2000


class TestCliIntegration:
    def _clean_path(self, tmp_path: Path) -> Path:
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        relu = helper.make_node("Relu", ["input"], ["output"], name="r")
        graph = helper.make_graph([relu], "m", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = tmp_path / "m.onnx"
        onnx.save(model, str(path))
        return path

    def test_successful_runtime_verification_yields_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: "/usr/bin/trtexec")
        _patch_run(monkeypatch, _FakeProc(0, stdout="TensorRT version 10.3"))
        result = CliRunner().invoke(
            main, [str(self._clean_path(tmp_path)), "--verify-runtime", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        assert '"verdict": "verified"' in result.stdout
        assert '"runtime_verified": true' in result.stdout

    def test_missing_trtexec_leaves_static_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runtime_verify.shutil, "which", lambda _: None)
        result = CliRunner().invoke(
            main, [str(self._clean_path(tmp_path)), "--verify-runtime", "--format", "json"]
        )
        assert result.exit_code == 0, result.output
        assert '"verdict": "likely"' in result.stdout
        assert '"status": "missing_trtexec"' in result.stdout
