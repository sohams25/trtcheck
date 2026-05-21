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


__all__ = ["Checker", "Fixer", "Reporter", "load_plugins"]


import importlib.metadata
import logging
from typing import Iterable

_logger = logging.getLogger("trtcheck.plugins")

_CHECKER_GROUP = "trtcheck.checkers"
_FIXER_GROUP = "trtcheck.fixers"
_REPORTER_GROUP = "trtcheck.reporters"


def _iter_entry_points(group: str) -> Iterable:
    """Yield entry-points for `group`. Wrapped in a helper so tests can
    monkey-patch it without touching importlib.metadata internals."""
    return importlib.metadata.entry_points(group=group)


def _load_one(ep, expected_proto: type) -> object | None:
    """Load and instantiate a single entry-point. Returns the instance, or
    None if anything fails. Failures are logged at WARNING level and never
    propagate."""
    try:
        cls = ep.load()
    except Exception as exc:
        _logger.warning("plugin %s: load failed (%s)", ep.name, exc.__class__.__name__)
        return None
    try:
        instance = cls()
    except Exception as exc:
        _logger.warning("plugin %s: construction failed (%s)", ep.name, exc)
        return None
    if not isinstance(instance, expected_proto):
        _logger.warning(
            "plugin %s: does not satisfy %s; skipping", ep.name, expected_proto.__name__
        )
        return None
    return instance


def load_plugins() -> tuple[list[Checker], list[Fixer], list[Reporter]]:
    """Discover and instantiate every entry-point plugin.

    Returns three lists (checkers, fixers, reporters) in entry-point
    iteration order. Plugins that fail to load, construct, or satisfy
    their Protocol are logged and skipped.
    """
    checkers: list[Checker] = []
    for ep in _iter_entry_points(_CHECKER_GROUP):
        inst = _load_one(ep, Checker)
        if inst is not None:
            checkers.append(inst)  # type: ignore[arg-type]

    fixers: list[Fixer] = []
    for ep in _iter_entry_points(_FIXER_GROUP):
        inst = _load_one(ep, Fixer)
        if inst is not None:
            fixers.append(inst)  # type: ignore[arg-type]

    reporters: list[Reporter] = []
    for ep in _iter_entry_points(_REPORTER_GROUP):
        inst = _load_one(ep, Reporter)
        if inst is not None:
            reporters.append(inst)  # type: ignore[arg-type]

    return checkers, fixers, reporters
