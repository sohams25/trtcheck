"""Optional runtime verification via NVIDIA's ``trtexec``.

Static analysis predicts; only a real TensorRT parse/build verifies. This
module shells out to ``trtexec --onnx=<model>`` when the user asks for it
(``trtcheck --verify-runtime``). It is deliberately isolated: nothing else
in trtcheck imports TensorRT, and every result state is explicit --
verification that could not run is never conflated with verification that
passed.

Security/robustness notes:
  - the subprocess is invoked as an argument list (no shell), so a crafted
    model filename cannot inject commands;
  - a timeout bounds runaway engine builds;
  - stdout/stderr are captured and truncated to tails, never echoed raw.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 600
_TAIL_CHARS = 2000

# Markers that indicate the ONNX parser (not the engine builder) rejected the
# model. Heuristic: trtexec does not exit with distinct codes for the two
# phases, but parser failures consistently mention the ONNX importer.
_PARSER_MARKERS = (
    "failed to parse onnx",
    "modelimporter",
    "onnx2trt",
    "could not parse the model",
    "parsing model failed",
    "assertion failed",
    "in function importmodel",
)


class RuntimeStatus(str, Enum):
    SUCCESS = "success"
    PARSER_FAILURE = "parser_failure"
    BUILD_FAILURE = "build_failure"
    MISSING_TRTEXEC = "missing_trtexec"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class RuntimeVerification:
    """Outcome of one trtexec run, with enough metadata to reproduce it."""

    status: RuntimeStatus
    trtexec_path: str | None = None
    trtexec_version: str | None = None
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    duration_s: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.status is RuntimeStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "trtexec_path": self.trtexec_path,
            "trtexec_version": self.trtexec_version,
            "command": self.command,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "detail": self.detail,
        }


def find_trtexec(explicit_path: str | None = None) -> str | None:
    """Resolve the trtexec executable, or None when unavailable."""
    if explicit_path:
        p = Path(explicit_path)
        return str(p) if p.is_file() else None
    return shutil.which("trtexec")


def _tail(text: str) -> str:
    return text[-_TAIL_CHARS:]


def _extract_version(output: str) -> str | None:
    for line in output.splitlines():
        lowered = line.lower()
        if "tensorrt" in lowered and ("version" in lowered or " v" in lowered):
            return line.strip()[:200]
    return None


def _looks_like_parser_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _PARSER_MARKERS)


def verify_model(
    model_path: Path | str,
    *,
    trtexec_path: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> RuntimeVerification:
    """Run ``trtexec --onnx=<model>`` and classify the outcome.

    Never raises for expected failure modes; every outcome is a
    :class:`RuntimeVerification` with an explicit status.
    """
    exe = find_trtexec(trtexec_path)
    if exe is None:
        return RuntimeVerification(
            status=RuntimeStatus.MISSING_TRTEXEC,
            detail=(
                "trtexec not found on PATH (or at the given --trtexec path). "
                "Install TensorRT or point --trtexec at the executable."
            ),
        )

    command = [exe, f"--onnx={Path(model_path)}"]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RuntimeVerification(
            status=RuntimeStatus.TIMEOUT,
            trtexec_path=exe,
            command=command,
            duration_s=time.monotonic() - start,
            detail=f"trtexec did not finish within {timeout_s}s",
        )
    except OSError as exc:
        return RuntimeVerification(
            status=RuntimeStatus.ERROR,
            trtexec_path=exe,
            command=command,
            detail=f"could not execute trtexec: {exc}",
        )

    duration = time.monotonic() - start
    combined = proc.stdout + "\n" + proc.stderr
    version = _extract_version(combined)
    if proc.returncode == 0:
        status = RuntimeStatus.SUCCESS
        detail = "trtexec parsed the model and built an engine"
    elif _looks_like_parser_failure(combined):
        status = RuntimeStatus.PARSER_FAILURE
        detail = "the TensorRT ONNX parser rejected the model"
    else:
        status = RuntimeStatus.BUILD_FAILURE
        detail = "the model parsed but the engine build failed"
    return RuntimeVerification(
        status=status,
        trtexec_path=exe,
        trtexec_version=version,
        command=command,
        returncode=proc.returncode,
        duration_s=duration,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
        detail=detail,
    )
