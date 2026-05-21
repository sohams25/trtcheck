"""Reporter plugins. Each takes an AnalysisReport and returns a string."""

from __future__ import annotations

from typing import Protocol

from trtcheck.types import AnalysisReport


class Reporter(Protocol):
    name: str

    def render(self, report: AnalysisReport) -> str:
        ...


__all__ = ["Reporter"]
