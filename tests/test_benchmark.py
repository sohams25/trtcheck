"""Optional performance benchmark.

Skipped by default. Run with TRTCHECK_BENCH=1 set in the environment, or
via pytest -m benchmark, after dropping a real ONNX model at
tests/fixtures/benchmark/model.onnx (gitignored).
"""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from trtcheck import analyze
from trtcheck.analyzer import Analyzer, AnalyzerConfig

_BENCH_DIR = Path(__file__).parent / "fixtures" / "benchmark"
_BENCH_MODEL = _BENCH_DIR / "model.onnx"
_TIME_BUDGET_S = 10.0
_MEM_BUDGET_MB = 200.0

# Synthetic regression tripwire (runs in CI, no external assets). Generous vs the
# real work so it doesn't flake on slow runners, but tight enough to catch an
# accidental O(n^2) blow-up in a checker on a few-thousand-node graph.
_SYNTH_NODES = 4000
_SYNTH_TIME_BUDGET_S = 5.0


def _chain_model(n_nodes: int) -> onnx.ModelProto:
    """A deterministic flat graph: Input -> Relu -> Relu -> ... -> Output."""
    nodes = [
        helper.make_node("Relu", [f"x{i}"], [f"x{i + 1}"], name=f"relu{i}") for i in range(n_nodes)
    ]
    inp = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, 16])
    out = helper.make_tensor_value_info(f"x{n_nodes}", TensorProto.FLOAT, [1, 16])
    graph = helper.make_graph(nodes, "synthetic", [inp], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_analysis_speed_and_memory_on_medium_graph() -> None:
    """Analyzing a few-thousand-node graph must stay fast and bounded in memory.

    This is the only always-on performance guard; the real-model benchmark below
    is opt-in. It validates the product's speed/footprint thesis against a
    deterministic, GPU-free graph and trips if a checker regresses to O(n^2) or
    starts allocating per-node.
    """
    model = _chain_model(_SYNTH_NODES)  # built outside the timed/traced region
    analyzer = Analyzer(AnalyzerConfig())

    tracemalloc.start()
    start = time.perf_counter()
    report = analyzer.analyze_model(model)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    assert report.total_nodes == _SYNTH_NODES
    assert elapsed < _SYNTH_TIME_BUDGET_S, (
        f"analysis of {_SYNTH_NODES} nodes took {elapsed:.2f}s "
        f"(budget {_SYNTH_TIME_BUDGET_S}s) -- possible algorithmic regression"
    )
    assert (
        peak_mb < _MEM_BUDGET_MB
    ), f"analysis allocated {peak_mb:.1f} MB (budget {_MEM_BUDGET_MB} MB)"


@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("TRTCHECK_BENCH") != "1",
    reason="Set TRTCHECK_BENCH=1 and provide tests/fixtures/benchmark/model.onnx to run.",
)
def test_analysis_speed_within_budget() -> None:
    if not _BENCH_MODEL.exists():
        pytest.skip(
            f"No benchmark model at {_BENCH_MODEL}. "
            "Drop a real ONNX file there (e.g. resnet50) to enable this test."
        )
    start = time.perf_counter()
    report = analyze(_BENCH_MODEL)
    elapsed = time.perf_counter() - start
    print(f"\nanalyzed {report.total_nodes} nodes in {elapsed:.2f}s")
    assert elapsed < _TIME_BUDGET_S, f"analysis took {elapsed:.2f}s, budget is {_TIME_BUDGET_S}s"
