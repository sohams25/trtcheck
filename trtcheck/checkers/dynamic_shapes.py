"""Dynamic-shape checks.

A healthy export has at most one symbolic dim (typically batch). When
multiple dims are symbolic, TRT can still build an engine but loses much
of its ability to optimize and pre-allocate buffers, and the resulting
engine often performs worse than a fixed-shape one.
"""

from __future__ import annotations

import onnx

from trtcheck.types import CheckCategory, Issue, Severity

_REMEDIATION = (
    "When exporting from PyTorch, pass dynamic_axes only for the dimensions "
    "that truly vary at runtime (typically batch). Leave height/width "
    "concrete: dynamic_axes={'input': {0: 'batch'}}."
)
_DOCS = (
    "https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html"
    "#work_dynamic_shapes"
)


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
                    Issue(
                        severity=Severity.WARNING,
                        category=CheckCategory.DYNAMIC_SHAPES,
                        node_name=inp.name,
                        operator="Input",
                        message=(
                            f"Input '{inp.name}' has {symbolic_count} of {total} "
                            f"dimensions dynamic: [{', '.join(rendered)}]. "
                            "TensorRT can still build but loses significant "
                            "optimization opportunities."
                        ),
                        remediation=_REMEDIATION,
                        docs_link=_DOCS,
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
