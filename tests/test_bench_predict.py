"""Tests for bench/predict.py -- the trtcheck leg of the validation harness."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.predict import predict, resolve_model_path, verdict_from_report  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent


def test_resolve_url_entry_points_into_cache() -> None:
    entry = {"name": "resnet50_v2", "source": "https://example.com/m.onnx"}
    assert resolve_model_path(entry, _ROOT) == _ROOT / "bench" / "cache" / "resnet50_v2.onnx"


def test_resolve_bundled_entry_is_repo_relative() -> None:
    entry = {"name": "x", "source": "tests/fixtures/clean_minimal.onnx"}
    assert resolve_model_path(entry, _ROOT) == _ROOT / "tests" / "fixtures" / "clean_minimal.onnx"


def test_verdict_mapping() -> None:
    assert verdict_from_report({"conversion_likely": True}) == "convert"
    assert verdict_from_report({"conversion_likely": False}) == "fail"


def test_verdict_requires_the_key() -> None:
    with pytest.raises(KeyError):
        verdict_from_report({})


def test_predict_end_to_end_on_bundled_fixtures(tmp_path: Path) -> None:
    # Real CLI, real fixtures: one clean model, one guaranteed-critical model.
    entries = [
        {"name": "clean", "source": "tests/fixtures/clean_minimal.onnx"},
        {"name": "uint8", "source": "tests/fixtures/failing/uint8_input.onnx"},
    ]
    out_path = tmp_path / "outcomes.json"
    predictions = predict(
        entries, _ROOT, trtcheck_cmd=[sys.executable, "-m", "trtcheck"], out_path=out_path
    )
    assert predictions == {
        "clean": {"trtcheck": "convert"},
        "uint8": {"trtcheck": "fail"},
    }
    assert json.loads(out_path.read_text()) == {"predictions": predictions}
