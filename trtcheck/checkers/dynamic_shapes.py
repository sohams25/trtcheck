"""Dynamic-shape checks.

A healthy export has at most one symbolic dim (typically batch). When
multiple dims are symbolic, TRT can still build an engine but loses much
of its ability to optimize and pre-allocate buffers, and the resulting
engine often performs worse than a fixed-shape one.

Remediation/explanation/severity live in remediation_db.json (via
:mod:`trtcheck.remediation`); this checker supplies the per-node prefix.
"""

from __future__ import annotations

import onnx

from trtcheck import remediation
from trtcheck.types import Issue

# Remediation-DB keys this checker can emit (guarded by tests/test_data_files.py).
EMITS = frozenset({"fully_dynamic_input_shape"})


class DynamicShapeChecker:
    name = "dynamic_shapes"

    def check(self, model: onnx.ModelProto) -> list[Issue]:
        issues: list[Issue] = []
        for inp in model.graph.input:
            shape = _shape_of(inp)
            if shape is None:
                continue  # scalar or no shape info
            symbolic_count = sum(1 for dim in shape if isinstance(dim, str))
            total = len(shape)
            if total > 0 and symbolic_count >= 2:
                rendered = [d if isinstance(d, str) else str(d) for d in shape]
                issues.append(
                    remediation.make_issue(
                        "fully_dynamic_input_shape",
                        node_name=inp.name,
                        operator="Input",
                        prefix=(
                            f"Input '{inp.name}' has {symbolic_count} of {total} "
                            f"dimensions dynamic: [{', '.join(rendered)}]"
                        ),
                    )
                )
        return issues


def _shape_of(value_info: onnx.ValueInfoProto) -> list[int | str] | None:
    if not value_info.type.tensor_type.shape.dim:
        return None
    out: list[int | str] = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_param:
            out.append(d.dim_param)
        elif not d.HasField("dim_value"):
            # Unnamed dynamic dim -- exporter left it symbolic but did not
            # give it a name. Treat as symbolic so the rest of the checker
            # counts it as dynamic.
            out.append("?")
        elif d.dim_value < 0:
            # Several exporters (and TensorRT itself) encode an unknown/dynamic
            # dimension as a concrete -1 rather than a dim_param. A negative
            # extent is never a real static size, so treat it as symbolic --
            # otherwise a [-1, -1, -1, -1] fully dynamic input reads as static.
            out.append("?")
        else:
            out.append(d.dim_value)
    return out
