"""`python -m trtcheck` must work -- it is the documented local-run form.

Uses a subprocess because the `-m` module-execution path (trtcheck/__main__.py)
cannot be exercised in-process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _clean_env() -> dict[str, str]:
    # Strip a leaked ROS PYTHONPATH so the subprocess imports our package, not
    # a stray system one (see scripts/run-tests.sh).
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("AMENT_PREFIX_PATH", None)
    return env


def test_python_m_trtcheck_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "trtcheck", "--version"],
        capture_output=True,
        text=True,
        env=_clean_env(),
        cwd=str(_REPO),
    )
    assert result.returncode == 0, result.stderr
    assert "trtcheck" in result.stdout.lower()


def test_python_m_trtcheck_runs_against_clean_fixture() -> None:
    model = _REPO / "tests" / "fixtures" / "clean_minimal.onnx"
    result = subprocess.run(
        [sys.executable, "-m", "trtcheck", str(model)],
        capture_output=True,
        text=True,
        env=_clean_env(),
        cwd=str(_REPO),
    )
    assert result.returncode == 0, result.stderr
    assert "convert" in result.stdout.lower()
