"""ONNX graph fixers.

Where checkers diagnose, fixers rewrite. Each Fixer mutates the model in
place and returns a list of `FixApplied` records describing what changed.

Fixers are conservative: if a rewrite is not unambiguously safe (e.g. an
INT64 weight whose values exceed INT32 range) the fixer must skip it and
emit nothing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

import onnx


@dataclass
class FixApplied:
    """A single transformation a fixer made to the model."""

    fixer: str
    target: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixer": self.fixer,
            "target": self.target,
            "description": self.description,
        }


class Fixer(Protocol):
    name: str

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]: ...


def apply_all(
    model: onnx.ModelProto,
    fixers: list[Fixer],
) -> tuple[onnx.ModelProto, list[FixApplied]]:
    """Deep-copy `model`, apply every fixer in order, return the new model.

    The original `model` is never mutated. Useful when the caller wants to
    keep both the before and after for diffing.
    """
    new_model = copy.deepcopy(model)
    applied: list[FixApplied] = []
    for fixer in fixers:
        applied.extend(fixer.fix(new_model))
    return new_model, applied


def default_fixers() -> list[Fixer]:
    """The built-in fixer pipeline applied by `trtcheck --fix`."""
    from trtcheck.fixers.drop_dropout import DropDropoutFixer
    from trtcheck.fixers.float64_to_float32 import Float64ToFloat32Fixer
    from trtcheck.fixers.int64_to_int32 import Int64ToInt32Fixer
    from trtcheck.fixers.uint8_input import Uint8InputFixer
    from trtcheck.fixers.upsample_to_resize import UpsampleToResizeFixer

    return [
        Int64ToInt32Fixer(),
        Float64ToFloat32Fixer(),
        DropDropoutFixer(),
        UpsampleToResizeFixer(),
        Uint8InputFixer(),
    ]


__all__ = ["Fixer", "FixApplied", "apply_all", "default_fixers"]
