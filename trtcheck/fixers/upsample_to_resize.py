"""Rewrite deprecated Upsample nodes to Resize.

Upsample was deprecated after ONNX opset 9 in favour of Resize. TRT prefers
Resize and the two ops are equivalent for the simple cases this fixer
handles. The fixer ONLY rewrites:

  - mode = nearest or linear
  - Upsample opset 9 form: inputs are (X, scales), mode is an attribute

Anything else (mode=cubic, scales as attribute, etc.) is left alone --
those edge cases need human review.

The rewrite preserves the original output name so downstream consumers
need no edits.
"""

from __future__ import annotations

import onnx
from onnx import AttributeProto, helper

from trtcheck._graph import iter_subgraphs
from trtcheck.fixers import FixApplied

_SAFE_MODES = {"nearest", "linear"}


class UpsampleToResizeFixer:
    name = "upsample_to_resize"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        # Resize is opset 10+, but `check_model` only accepts empty-string
        # placeholders for the optional roi/sizes inputs starting at opset 13.
        # Refuse below that -- the resulting graph would not validate. The opset
        # is a model-level property, so decide once before walking subgraphs.
        default_opset = next(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            0,
        )
        if default_opset < 13:
            return []

        applied: list[FixApplied] = []
        for graph in iter_subgraphs(model.graph):
            applied.extend(self._fix_graph(graph))
        return applied

    def _fix_graph(self, graph: onnx.GraphProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        for node in list(graph.node):
            if node.op_type != "Upsample":
                continue
            if len(node.input) != 2:
                # opset 7 form has scales as an attribute -- skip.
                continue

            mode_attr = next((a for a in node.attribute if a.name == "mode"), None)
            if mode_attr is None:
                continue
            mode = mode_attr.s.decode() if mode_attr.type == AttributeProto.STRING else ""
            if mode not in _SAFE_MODES:
                continue

            x_in, scales_in = node.input[0], node.input[1]
            new_resize = helper.make_node(
                "Resize",
                inputs=[x_in, "", scales_in, ""],  # X, roi, scales, sizes
                outputs=list(node.output),
                name=node.name or "",
                mode=mode,
            )
            # Replace in-place to preserve node order.
            real_idx = list(graph.node).index(node)
            graph.node.remove(node)
            graph.node.insert(real_idx, new_resize)

            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=node.name or "<Upsample>",
                    description=(
                        f"rewrote Upsample '{node.name}' (mode={mode}) as a "
                        "Resize node with empty roi/sizes placeholders"
                    ),
                )
            )
        return applied
