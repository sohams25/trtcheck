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

from trtcheck.fixers import FixApplied

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


class Int64ToInt32Fixer:
    name = "int64_to_int32"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        for init in model.graph.initializer:
            if init.data_type != TensorProto.INT64:
                continue
            arr = numpy_helper.to_array(init)
            if arr.min() < _INT32_MIN or arr.max() > _INT32_MAX:
                # Out-of-range; the user must handle this manually.
                continue
            new_arr = arr.astype(np.int32)
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_init)
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
