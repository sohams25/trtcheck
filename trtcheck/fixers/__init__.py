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
from typing import Any

import onnx

from trtcheck.plugins import Fixer


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


def sync_value_info_dtype(graph: onnx.GraphProto, name: str, elem_type: int) -> None:
    """Retype any graph input/output ``ValueInfo`` named ``name`` to ``elem_type``.

    ONNX legally permits an initializer name to also appear in ``graph.input``
    (the initializer supplies the input's default) or ``graph.output``. When a
    fixer rewrites that initializer's dtype it MUST update the matching
    ``ValueInfoProto`` too -- otherwise the input/output still declares the old
    element type and the model fails full type inference
    (``onnx.checker.check_model(..., full_check=True)``), shipping a corrupt
    ``--fix`` artifact that the shallow default checker never catches.

    Only existing ``tensor_type`` value-infos are touched: a sequence/optional
    input (or one with no declared type) is left alone rather than having a
    tensor type fabricated for it.
    """
    for value_info in list(graph.input) + list(graph.output):
        if value_info.name != name:
            continue
        if value_info.type.WhichOneof("value") == "tensor_type":
            value_info.type.tensor_type.elem_type = elem_type


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


__all__ = [
    "Fixer",
    "FixApplied",
    "apply_all",
    "default_fixers",
    "sync_value_info_dtype",
]
