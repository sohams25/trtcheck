"""Optional performance benchmark.

Skipped by default. Run with TRTCHECK_BENCH=1 set in the environment, or
via pytest -m benchmark, after dropping a real ONNX model at
tests/fixtures/benchmark/model.onnx (gitignored).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from trtcheck import analyze

_BENCH_DIR = Path(__file__).parent / "fixtures" / "benchmark"
_BENCH_MODEL = _BENCH_DIR / "model.onnx"
_TIME_BUDGET_S = 10.0
_MEM_BUDGET_MB = 200.0


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
