"""Public extension surface for trtcheck.

Third-party packages implement these Protocols to ship checkers, fixers,
and reporters that load through entry-points. The Protocols live here
(rather than next to each plugin family) so the import path is stable for
v1.0 and beyond.

Existing import paths (`trtcheck.checkers.Checker`,
`trtcheck.fixers.Fixer`, `trtcheck.reporters.Reporter`) re-export from
this module unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import onnx

from trtcheck.types import AnalysisReport, Issue

if TYPE_CHECKING:
    # Imported only for the Fixer.fix() return type. Avoids a runtime
    # circular import (trtcheck.fixers re-imports from this module).
    from trtcheck.fixers import FixApplied


@runtime_checkable
class Checker(Protocol):
    """Read-only ONNX analysis. Returns findings; never mutates the model."""

    name: str

    def check(self, model: onnx.ModelProto) -> list[Issue]: ...


@runtime_checkable
class Fixer(Protocol):
    """Conservative ONNX rewrite. Mutates the model in place when the
    rewrite is unambiguously safe; refuses otherwise."""

    name: str

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]: ...


@runtime_checkable
class Reporter(Protocol):
    """Pure formatter: AnalysisReport in, string out."""

    name: str

    def render(self, report: AnalysisReport) -> str: ...


__all__ = ["Checker", "Fixer", "Reporter"]
