"""ONNX graph fixers.

Where checkers diagnose, fixers rewrite. Each Fixer mutates the model in
place and returns a list of `FixApplied` records describing what changed.

Fixers are conservative: if a rewrite is not unambiguously safe (e.g. an
INT64 weight whose values exceed INT32 range) the fixer must skip it and
emit nothing.
"""

from __future__ import annotations

import copy
import os
import traceback
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


@dataclass
class FixFailure:
    """A fixer that was rejected by the transactional pipeline.

    ``reason`` is a short human-readable explanation (no traceback -- set
    ``TRTCHECK_DEBUG=1`` for those). The failing fixer's changes were
    discarded; the model the pipeline returns never contains them.
    """

    fixer: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"fixer": self.fixer, "reason": self.reason}


@dataclass
class FixOutcome:
    """Result of a transactional :func:`run_fixers` pass."""

    model: onnx.ModelProto
    applied: list[FixApplied]
    failures: list[FixFailure]
    validation: str = "full"  # "full" | "basic" (external-data models)


def uses_external_data(model: onnx.ModelProto) -> bool:
    """True when any initializer stores its payload outside the protobuf."""
    from trtcheck._graph import iter_initializers

    return any(
        init.data_location == onnx.TensorProto.EXTERNAL
        for init, _ in iter_initializers(model.graph)
    )


def validate_model(model: onnx.ModelProto, *, level: str = "full") -> None:
    """Validate a (candidate) model, raising on any problem.

    ``level="full"`` runs ``onnx.checker.check_model(..., full_check=True)``,
    which includes strict shape/type inference -- this is what catches a
    dtype rewrite that basic checking waves through (e.g. an INT32 tensor
    feeding an input whose schema demands INT64). ``level="basic"`` is the
    fallback for external-data models, where full inference cannot read the
    tensor payloads safely from an in-memory proto. ``level="none"`` skips
    validation entirely (used only when the input model itself cannot pass
    the basic check -- a candidate is never held to a bar the input missed).
    """
    if level == "none":
        return
    onnx.checker.check_model(model, full_check=(level == "full"))


def validation_level_for(model: onnx.ModelProto) -> str:
    """Choose the strongest validation level the *input* model already passes.

    A candidate is only ever held to a bar the pre-fix model could meet --
    otherwise a pre-existing quirk (custom-domain ops that defeat strict
    shape inference, external-data initializers) would make every fix look
    like the fixer's failure. Returns ``"full"``, ``"basic"``, or ``"none"``
    (input fails even the basic check; the CLI refuses to --fix such models,
    but library callers may still run fixers on them at their own risk).
    """
    if uses_external_data(model):
        level_ceiling = "basic"
    else:
        try:
            validate_model(model, level="full")
            return "full"
        except Exception:
            level_ceiling = "basic"
    try:
        validate_model(model, level="basic")
    except Exception:
        return "none"
    return level_ceiling


def run_fixers(
    model: onnx.ModelProto,
    fixers: list[Fixer],
    *,
    validate: bool = True,
) -> FixOutcome:
    """Run every fixer transactionally against a deep-copied candidate.

    Invariants:
      - the input ``model`` is never mutated;
      - each fixer runs against a fresh deep copy of the last *valid* model,
        so a fixer that mutates and then crashes (or produces an invalid
        model, or returns malformed records) cannot leak partial edits;
      - one failed fixer does not stop later fixers -- they run against the
        last valid state.
    """
    current = copy.deepcopy(model)
    level = validation_level_for(model)
    applied: list[FixApplied] = []
    failures: list[FixFailure] = []
    for fixer in fixers:
        name = getattr(fixer, "name", fixer.__class__.__name__)
        candidate = copy.deepcopy(current)
        try:
            fixes = fixer.fix(candidate)
        except Exception as exc:
            reason = f"raised {exc.__class__.__name__}: {exc}; changes discarded"
            # Tracebacks (possibly from third-party plugins) are opt-in only.
            if os.environ.get("TRTCHECK_DEBUG", "") not in ("", "0"):
                reason += "\n" + traceback.format_exc()
            failures.append(FixFailure(name, reason))
            continue
        if not isinstance(fixes, list) or not all(isinstance(f, FixApplied) for f in fixes):
            failures.append(FixFailure(name, "returned malformed fix records; changes discarded"))
            continue
        if not fixes:
            # Fixer claims it changed nothing: discard the candidate anyway
            # (an undeclared mutation must not survive).
            continue
        if candidate.SerializeToString() == current.SerializeToString():
            # Fixer claims fixes but changed nothing: refusing keeps the
            # applied-fixes list truthful (a --fix report must never list a
            # change that is not in the output model).
            failures.append(
                FixFailure(
                    name,
                    f"reported {len(fixes)} fix(es) but did not modify the model; "
                    "records discarded",
                )
            )
            continue
        if validate:
            try:
                validate_model(candidate, level=level)
            except Exception as exc:
                failures.append(
                    FixFailure(
                        name,
                        f"produced an invalid model ({exc.__class__.__name__}); "
                        "changes discarded",
                    )
                )
                continue
        current = candidate
        applied.extend(fixes)
    return FixOutcome(model=current, applied=applied, failures=failures, validation=level)


def apply_all(
    model: onnx.ModelProto,
    fixers: list[Fixer],
) -> tuple[onnx.ModelProto, list[FixApplied]]:
    """Deep-copy `model`, apply every fixer transactionally, return the new model.

    The original `model` is never mutated. Kept for API compatibility;
    :func:`run_fixers` additionally reports per-fixer failures.
    """
    outcome = run_fixers(model, fixers)
    return outcome.model, outcome.applied


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
    "FixFailure",
    "FixOutcome",
    "apply_all",
    "run_fixers",
    "validate_model",
    "validation_level_for",
    "uses_external_data",
    "default_fixers",
    "sync_value_info_dtype",
]
