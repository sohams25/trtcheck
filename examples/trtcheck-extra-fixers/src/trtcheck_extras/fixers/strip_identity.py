"""Remove redundant Identity nodes.

ONNX Identity nodes survive some export pipelines as leftover scaffolding.
TensorRT folds them out anyway, but removing them up front keeps the
diagnostic report cleaner.

This is intentionally trivial. It exists to demonstrate the plugin API,
not to be a state-of-the-art simplifier.
"""

from __future__ import annotations

import onnx

from trtcheck.fixers import FixApplied


class StripIdentityFixer:
    name = "strip_identity"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        graph = model.graph
        applied: list[FixApplied] = []
        # Use a set lookup so we don't rescan the node list per Identity.
        for node in list(graph.node):
            if node.op_type != "Identity":
                continue
            inp_name = node.input[0]
            out_name = node.output[0]

            # If the Identity output is a graph output, refuse -- renaming it
            # would change the model's public interface.
            if any(o.name == out_name for o in graph.output):
                continue
            # If the input is a graph input AND has multiple consumers, the
            # safe rewrite gets messy. Keep this example narrow.
            consumers = [
                other for other in graph.node if other is not node and out_name in other.input
            ]
            if not consumers:
                continue
            for other in consumers:
                for i, name in enumerate(other.input):
                    if name == out_name:
                        other.input[i] = inp_name
            graph.node.remove(node)
            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=node.name or "<Identity>",
                    description=(
                        f"removed redundant Identity '{node.name}'; "
                        f"downstream consumers now read '{inp_name}' directly"
                    ),
                )
            )
        return applied
