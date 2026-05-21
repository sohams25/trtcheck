"""Cast FLOAT64 initializers to FLOAT32 when values fit and are finite.

TensorRT has no double-precision support. The cast happens at engine build
time anyway, but doing it at fix time surfaces overflow and NaN/Inf
contamination as a clear refusal rather than an opaque engine-build
failure.

The fixer refuses when:
  - any value is NaN or +/-inf (TRT may or may not preserve these; do not
    introduce silent semantic changes)
  - any finite value exceeds the FLOAT32 range (would round to +/-inf)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from trtcheck.fixers import FixApplied

_FP32_MAX = np.finfo(np.float32).max


class Float64ToFloat32Fixer:
    name = "float64_to_float32"

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        applied: list[FixApplied] = []
        for init in model.graph.initializer:
            if init.data_type != TensorProto.DOUBLE:
                continue
            arr = numpy_helper.to_array(init)
            if arr.size == 0:
                continue
            if not np.all(np.isfinite(arr)):
                continue
            if np.max(np.abs(arr)) > _FP32_MAX:
                continue
            new_arr = arr.astype(np.float32)
            new_init = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_init)
            applied.append(
                FixApplied(
                    fixer=self.name,
                    target=init.name,
                    description=(
                        f"cast initializer '{init.name}' from FLOAT64 to "
                        f"FLOAT32 ({arr.size} elements)"
                    ),
                )
            )
        return applied
