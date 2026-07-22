"""Cast INT64 initializers to INT32 -- only where every use provably allows it.

TensorRT casts INT64 to INT32 at engine build time. Doing it at ONNX-rewrite
time surfaces overflow early -- but it is only legal where the consuming
operator's schema accepts INT32 at that input. Many ONNX inputs *require*
INT64: ``Reshape``'s ``shape``, ``Slice``'s ``starts``/``ends``, ``Squeeze``'s
``axes``, ``Pad``'s ``pads``, ``Tile``'s ``repeats``, ... Blindly converting
those produces a model that passes the shallow checker but fails full type
inference (and would fail in any conforming runtime).

This fixer is therefore use-aware and conservative:

  - Every use of the initializer, across the whole model including nested
    If/Loop/Scan bodies (subgraphs may capture outer-scope names), must be at
    a consumer input position on the explicit allowlist below -- positions
    whose ONNX type constraint admits ``tensor(int32)`` *independently* of
    the operator's other inputs and outputs.
  - Positions like elementwise ``Add``/``Mul`` operands are deliberately NOT
    allowlisted even though int32 is an allowed dtype there: their type
    variable binds several inputs/outputs at once, so retyping one operand
    breaks the binding.
  - A name defined in more than one scope (shadowing), consumed by a
    custom-domain or unknown node, exposed as a graph input/output, or not
    consumed at all is skipped.
  - Values outside INT32 range and empty tensors are skipped.

No speculative Cast nodes are ever inserted to force a conversion through.
Skipped conversions are logged at INFO level on the ``trtcheck.fixers``
logger.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from trtcheck._graph import iter_nodes, iter_subgraphs
from trtcheck.fixers import FixApplied

_logger = logging.getLogger("trtcheck.fixers")

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# (op_type, input_index) positions whose ONNX schema accepts tensor(int32)
# through a type variable that binds ONLY that input (so retyping it cannot
# break a binding with another input or output):
#   - Gather / GatherElements / ScatterElements ``indices`` use the dedicated
#     Tind constraint = {tensor(int32), tensor(int64)}.
#   - Cast input 0 accepts any tensor type; its output type comes from the
#     ``to`` attribute, unchanged by this rewrite.
#   - Shape / Size input 0 accept any tensor type and always emit INT64;
#     the input dtype cannot leak anywhere.
_INT32_SAFE_POSITIONS: frozenset[tuple[str, int]] = frozenset(
    {
        ("Gather", 1),
        ("GatherElements", 1),
        ("ScatterElements", 1),
        ("Cast", 0),
        ("Shape", 0),
        ("Size", 0),
    }
)

_DEFAULT_DOMAINS = ("", "ai.onnx")


class Int64ToInt32Fixer:
    name = "int64_to_int32"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        uses = _collect_uses(model)
        definition_counts = _definition_counts(model)
        for graph in iter_subgraphs(model.graph):
            boundary_names = {vi.name for vi in graph.input} | {vi.name for vi in graph.output}
            for init in graph.initializer:
                if init.data_type != TensorProto.INT64:
                    continue
                reason = self._skip_reason(init, uses, definition_counts, boundary_names)
                if reason is not None:
                    _logger.info("int64_to_int32: skipping '%s': %s", init.name, reason)
                    continue
                arr = numpy_helper.to_array(init)
                new_init = numpy_helper.from_array(arr.astype(np.int32), name=init.name)
                init.CopyFrom(new_init)
                applied.append(
                    FixApplied(
                        fixer=self.name,
                        target=init.name,
                        description=(
                            f"cast initializer '{init.name}' from INT64 to INT32 "
                            f"({arr.size} elements, range "
                            f"[{int(arr.min())}, {int(arr.max())}]); all uses are "
                            "at INT32-compatible input positions"
                        ),
                    )
                )
        return applied

    def _skip_reason(
        self,
        init: onnx.TensorProto,
        uses: dict[str, list[tuple[str, str, int]]],
        definition_counts: dict[str, int],
        boundary_names: set[str],
    ) -> str | None:
        """Return why this initializer must not be converted, or None if safe."""
        name = init.name
        if definition_counts.get(name, 0) > 1:
            return "name is defined in more than one scope (shadowing is ambiguous)"
        if name in boundary_names:
            return "initializer is also a graph input/output; converting changes the signature"
        consumer_positions = uses.get(name, [])
        if not consumer_positions:
            return "initializer has no consumers; nothing to gain from converting"
        for domain, op_type, idx in consumer_positions:
            if domain not in _DEFAULT_DOMAINS:
                return f"consumed by custom-domain op '{domain}::{op_type}'"
            if (op_type, idx) not in _INT32_SAFE_POSITIONS:
                return (
                    f"consumed by '{op_type}' input {idx}, which is not a "
                    "known INT32-compatible position (e.g. Reshape's shape "
                    "input requires INT64)"
                )
        arr = numpy_helper.to_array(init)
        if arr.size == 0:
            return "empty tensor; casting is a no-op"
        if arr.min() < _INT32_MIN or arr.max() > _INT32_MAX:
            return "values exceed INT32 range"
        return None


def _collect_uses(model: onnx.ModelProto) -> dict[str, list[tuple[str, str, int]]]:
    """name -> [(domain, op_type, input_index)] for every node input in the
    model, including nodes inside nested subgraphs (which may capture
    outer-scope initializers by name)."""
    uses: dict[str, list[tuple[str, str, int]]] = {}
    for node, _owner in iter_nodes(model.graph):
        for idx, inp in enumerate(node.input):
            if inp:
                uses.setdefault(inp, []).append((node.domain, node.op_type, idx))
    return uses


def _definition_counts(model: onnx.ModelProto) -> dict[str, int]:
    """How many scopes define each name (initializers, graph inputs, node
    outputs, subgraph inputs). >1 means uses of the name are scope-dependent
    and a rename-free rewrite cannot be proven safe."""
    counts: dict[str, int] = {}

    def bump(name: str) -> None:
        if name:
            counts[name] = counts.get(name, 0) + 1

    for graph in iter_subgraphs(model.graph):
        for init in graph.initializer:
            bump(init.name)
        for vi in graph.input:
            # opset<9 models list initializers in graph.input too; that pair is
            # one definition, not two.
            if all(init.name != vi.name for init in graph.initializer):
                bump(vi.name)
        for node in graph.node:
            for out in node.output:
                bump(out)
    return counts
