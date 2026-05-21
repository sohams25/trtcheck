"""JSON reporter -- machine-readable output for CI pipelines."""

from __future__ import annotations

import json

from trtcheck.types import AnalysisReport


class JSONReporter:
    name = "json"

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def render(self, report: AnalysisReport) -> str:
        return json.dumps(report.to_dict(), indent=self._indent)
