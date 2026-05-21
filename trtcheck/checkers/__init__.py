"""Checker plugins.

Every checker implements the `Checker` protocol: a single `check(model)`
method that returns a list of `Issue` instances. Checkers are pure -- they
read the ONNX model, return findings, and never print, format, or mutate.
"""

from __future__ import annotations

from typing import Protocol

import onnx

from trtcheck.types import Issue


class Checker(Protocol):
    name: str

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        ...


__all__ = ["Checker"]
