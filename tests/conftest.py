"""Shared pytest fixtures for the trtcheck suite."""

from pathlib import Path

import onnx
import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return _FIXTURE_DIR


@pytest.fixture
def clean_model() -> onnx.ModelProto:
    return onnx.load(str(_FIXTURE_DIR / "clean_minimal.onnx"))


def _load_failing(name: str) -> onnx.ModelProto:
    return onnx.load(str(_FIXTURE_DIR / "failing" / name))


@pytest.fixture
def sequence_empty_model() -> onnx.ModelProto:
    return _load_failing("sequence_empty.onnx")


@pytest.fixture
def int64_weights_model() -> onnx.ModelProto:
    return _load_failing("int64_weights.onnx")


@pytest.fixture
def fully_dynamic_model() -> onnx.ModelProto:
    return _load_failing("fully_dynamic.onnx")


@pytest.fixture
def uint8_input_model() -> onnx.ModelProto:
    return _load_failing("uint8_input.onnx")


@pytest.fixture
def control_flow_loop_model() -> onnx.ModelProto:
    return _load_failing("control_flow_loop.onnx")
