"""Analyzer edge-path coverage: external-data load failures and the size guard.

These pin behaviour that already works but was unguarded -- exactly the
"run on every model in CI" paths (a model exported with external weights whose
.bin is absent, and the max-model-size rejection) where a regression would let a
raw exception escape or flip the size boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from trtcheck.analyzer import Analyzer, AnalyzerConfig, safe_load


def _model_with_initializer() -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])
    w = numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="w")
    add = helper.make_node("Add", ["input", "w"], ["output"], name="add0")
    graph = helper.make_graph([add], "m", [inp], [out], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_external_data_missing_bin_is_clean_valueerror(tmp_path: Path) -> None:
    """A model whose weights live in an external .bin that is absent must surface
    as the same clean ValueError safe_load uses for any unloadable file, not a
    raw onnx/IO exception."""
    model = _model_with_initializer()
    path = tmp_path / "ext.onnx"
    # Force every initializer out to a sidecar file, then delete it.
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    sidecar = tmp_path / "weights.bin"
    assert sidecar.exists()
    sidecar.unlink()

    with pytest.raises(ValueError, match="could not parse"):
        safe_load(path)


def test_in_memory_external_model_analyzes_without_crashing() -> None:
    """An already-in-memory model carrying an EXTERNAL data_location (no file to
    read) must analyze without raising -- the checkers read metadata, not bytes."""
    model = _model_with_initializer()
    for init in model.graph.initializer:
        init.data_location = onnx.TensorProto.EXTERNAL
    analyzer = Analyzer(AnalyzerConfig(discover_entry_point_plugins=False))
    report = analyzer.analyze_model(model)  # must not raise
    assert report.total_nodes == 1


def test_size_limit_zero_rejects_nonempty_file(tmp_path: Path, fixture_dir: Path) -> None:
    """The guard is a strict ``>``: with a 0 MB limit any nonempty file (>0 MB)
    is rejected, and the message names the limit."""
    analyzer = Analyzer(AnalyzerConfig(max_model_size_mb=0, discover_entry_point_plugins=False))
    with pytest.raises(ValueError, match=r"above the .* MB limit"):
        analyzer.analyze_path(fixture_dir / "clean_minimal.onnx")


def test_size_limit_admits_file_under_limit(fixture_dir: Path) -> None:
    """A tiny fixture (well under 1 MB) loads when the limit is 1 MB -- the
    strict ``>`` admits anything at or below the limit."""
    analyzer = Analyzer(AnalyzerConfig(max_model_size_mb=1, discover_entry_point_plugins=False))
    report = analyzer.analyze_path(fixture_dir / "clean_minimal.onnx")
    assert report.total_nodes >= 0
