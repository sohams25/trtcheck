"""Cast INT64 initializers to INT32 where values fit.

TensorRT does not natively support INT64 -- it casts to INT32 at engine
build time. Doing the cast at ONNX-rewrite time has two advantages:

  1. It surfaces overflow at fix time rather than during engine build.
  2. It shaves a few bytes per element off the engine binary.

The fixer refuses to act if any value falls outside INT32 range.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from trtcheck._graph import iter_subgraphs
from trtcheck.fixers import FixApplied, sync_value_info_dtype

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


class Int64ToInt32Fixer:
    name = "int64_to_int32"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        # Descend into If/Loop/Scan subgraphs: an INT64 weight buried in a
        # branch body blocks conversion just the same.
        for graph in iter_subgraphs(model.graph):
            applied.extend(self._fix_graph(graph))
        return applied

    def _fix_graph(self, graph: onnx.GraphProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        for init in graph.initializer:
            if init.data_type != TensorProto.INT64:
                continue
            arr = numpy_helper.to_array(init)
            if arr.size == 0:
                # Empty initializer is trivially in range; casting is a no-op.
                # Skip it -- arr.min()/arr.max() on a zero-size array raises.
                continue
            if arr.min() < _INT32_MIN or arr.max() > _INT32_MAX:
                # Out-of-range; the user must handle this manually.
                continue
            new_arr = arr.astype(np.int32)
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_init)
            # If this initializer also shadows a graph input/output, retype that
            # ValueInfo to INT32 too -- otherwise full type inference rejects the
            # fixed model (legal ONNX: an initializer may also be a graph input).
            sync_value_info_dtype(graph, init.name, TensorProto.INT32)
            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=init.name,
                    description=(
                        f"cast initializer '{init.name}' from INT64 to INT32 "
                        f"({arr.size} elements, range [{int(arr.min())}, {int(arr.max())}])"
                    ),
                )
            )
        return applied
