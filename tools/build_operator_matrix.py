"""Generate operator_matrix.json from a curated Python source-of-truth.

This script is intentionally not part of the package -- it produces a data
file that ships with the wheel. Rerunning it should be deterministic.
"""

import copy
import json
from pathlib import Path
from typing import Any

VERSIONS = ["8.0", "8.6", "10.0", "10.3"]

# Status shorthand to keep the source readable:
#   S = supported, P = partial, N = not_supported, U = unknown
_STATUS = {"S": "supported", "P": "partial", "N": "not_supported", "U": "unknown"}


def expand(*codes: str) -> dict[str, str]:
    if len(codes) != len(VERSIONS):
        raise ValueError(f"need {len(VERSIONS)} status codes, got {len(codes)}")
    return {v: _STATUS[c] for v, c in zip(VERSIONS, codes)}


OPERATORS: dict[str, dict[str, Any]] = {
    # Convolutions
    "Conv": {
        "support": expand("S", "S", "S", "S"),
        "notes": "Full support across all TRT versions. NCHW format.",
        "limitations": ["3D convolution requires TRT 8.6+ for best performance."],
    },
    "ConvTranspose": {
        "support": expand("S", "S", "S", "S"),
        "notes": "Supported. Asymmetric padding handled via explicit Pad fusion in TRT 10.",
    },
    # GEMM / MatMul
    "MatMul": {"support": expand("S", "S", "S", "S"), "notes": "Supported with broadcasting."},
    "Gemm": {
        "support": expand("S", "S", "S", "S"),
        "notes": "Supported. Fused into Conv when possible.",
    },
    # Activations
    "Relu": {"support": expand("S", "S", "S", "S")},
    "LeakyRelu": {"support": expand("S", "S", "S", "S")},
    "PRelu": {"support": expand("S", "S", "S", "S")},
    "Sigmoid": {"support": expand("S", "S", "S", "S")},
    "Tanh": {"support": expand("S", "S", "S", "S")},
    "Elu": {"support": expand("S", "S", "S", "S")},
    "Selu": {"support": expand("S", "S", "S", "S")},
    "HardSigmoid": {"support": expand("S", "S", "S", "S")},
    "HardSwish": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Native fusion in TRT 8.6+. Decomposed in 8.0.",
    },
    "Gelu": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Native Gelu op added in TRT 8.6. Earlier versions emulate via Erf.",
    },
    "Mish": {
        "support": expand("N", "P", "S", "S"),
        "notes": "Added in TRT 8.6 as a plugin, native in 10.0+.",
    },
    "Softmax": {"support": expand("S", "S", "S", "S")},
    "LogSoftmax": {"support": expand("S", "S", "S", "S")},
    # Normalization
    "BatchNormalization": {"support": expand("S", "S", "S", "S")},
    "InstanceNormalization": {"support": expand("S", "S", "S", "S")},
    "LayerNormalization": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Native LayerNormalization op added in TRT 8.6. Pre-8.6 must decompose.",
        "remediation": "Use ONNX opset 17+ with native LayerNormalization, or upgrade TRT to 8.6+.",
    },
    "GroupNormalization": {
        "support": expand("N", "N", "S", "S"),
        "notes": "Added in TRT 10.0. Replace with BatchNormalization for older TRT.",
        "remediation": "Replace nn.GroupNorm with nn.BatchNorm2d, or upgrade TRT to 10.0+.",
    },
    # Element-wise arithmetic
    "Add": {"support": expand("S", "S", "S", "S")},
    "Sub": {"support": expand("S", "S", "S", "S")},
    "Mul": {"support": expand("S", "S", "S", "S")},
    "Div": {"support": expand("S", "S", "S", "S")},
    "Pow": {"support": expand("S", "S", "S", "S")},
    "Sqrt": {"support": expand("S", "S", "S", "S")},
    "Exp": {"support": expand("S", "S", "S", "S")},
    "Log": {"support": expand("S", "S", "S", "S")},
    "Abs": {"support": expand("S", "S", "S", "S")},
    "Neg": {"support": expand("S", "S", "S", "S")},
    "Reciprocal": {"support": expand("S", "S", "S", "S")},
    # Reductions
    "ReduceMean": {"support": expand("S", "S", "S", "S")},
    "ReduceSum": {"support": expand("S", "S", "S", "S")},
    "ReduceMax": {"support": expand("S", "S", "S", "S")},
    "ReduceMin": {"support": expand("S", "S", "S", "S")},
    "ReduceProd": {"support": expand("P", "S", "S", "S"), "notes": "Pre-8.6 requires opset >= 13."},
    # Comparison and logical
    "Equal": {"support": expand("S", "S", "S", "S")},
    "Greater": {"support": expand("S", "S", "S", "S")},
    "GreaterOrEqual": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Decomposed before TRT 8.6.",
    },
    "Less": {"support": expand("S", "S", "S", "S")},
    "LessOrEqual": {"support": expand("P", "S", "S", "S")},
    "And": {"support": expand("S", "S", "S", "S")},
    "Or": {"support": expand("S", "S", "S", "S")},
    "Not": {"support": expand("S", "S", "S", "S")},
    "Where": {"support": expand("S", "S", "S", "S")},
    # Shape ops
    "Reshape": {"support": expand("S", "S", "S", "S")},
    "Transpose": {"support": expand("S", "S", "S", "S")},
    "Slice": {"support": expand("S", "S", "S", "S")},
    "Concat": {"support": expand("S", "S", "S", "S")},
    "Split": {"support": expand("S", "S", "S", "S")},
    "Squeeze": {"support": expand("S", "S", "S", "S")},
    "Unsqueeze": {"support": expand("S", "S", "S", "S")},
    "Tile": {"support": expand("S", "S", "S", "S")},
    "Expand": {"support": expand("S", "S", "S", "S")},
    "Pad": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Only constant mode in 8.0. Reflect/edge added in 8.6.",
    },
    "Shape": {"support": expand("S", "S", "S", "S")},
    "Size": {"support": expand("S", "S", "S", "S")},
    "Gather": {"support": expand("S", "S", "S", "S")},
    "GatherElements": {"support": expand("P", "S", "S", "S")},
    "GatherND": {"support": expand("P", "S", "S", "S")},
    "Scatter": {"support": expand("P", "S", "S", "S")},
    "ScatterElements": {"support": expand("P", "S", "S", "S")},
    "ScatterND": {"support": expand("P", "S", "S", "S")},
    # Pooling
    "MaxPool": {"support": expand("S", "S", "S", "S")},
    "AveragePool": {"support": expand("S", "S", "S", "S")},
    "GlobalMaxPool": {"support": expand("S", "S", "S", "S")},
    "GlobalAveragePool": {"support": expand("S", "S", "S", "S")},
    # Resize / Upsample
    "Resize": {
        "support": expand("P", "P", "S", "S"),
        "notes": "Only nearest and linear modes pre-10.0. Cubic added in 10.0.",
        "limitations": ["antialias attribute not supported before TRT 10.0."],
    },
    "Upsample": {
        "support": expand("S", "S", "S", "S"),
        "notes": "Deprecated in newer ONNX opsets, use Resize.",
    },
    # Quantization
    "QuantizeLinear": {"support": expand("S", "S", "S", "S")},
    "DequantizeLinear": {"support": expand("S", "S", "S", "S")},
    "Cast": {
        "support": expand("P", "P", "S", "S"),
        "notes": "UINT8 -> INT8 cast unreliable pre-10.0. FLOAT64 source always rejected.",
        "remediation": "Cast inputs to FLOAT32/INT32 before exporting from PyTorch.",
    },
    # Sequence ops (PyTorch list[]) -- the headliner failure mode
    "SequenceEmpty": {
        "support": expand("N", "N", "N", "N"),
        "notes": "Sequence ops are not supported by TensorRT. Emitted when PyTorch code uses List[Tensor].",
        "remediation": "Replace List[Tensor] with torch.stack() or pre-allocate a tensor.",
        "github_issue": "https://github.com/onnx/onnx-tensorrt/issues/1044",
    },
    "SequenceInsert": {"support": expand("N", "N", "N", "N"), "notes": "See SequenceEmpty."},
    "SequenceAt": {"support": expand("N", "N", "N", "N"), "notes": "See SequenceEmpty."},
    "SequenceConstruct": {"support": expand("N", "N", "N", "N"), "notes": "See SequenceEmpty."},
    "SequenceLength": {"support": expand("N", "N", "N", "N"), "notes": "See SequenceEmpty."},
    "SequenceErase": {"support": expand("N", "N", "N", "N"), "notes": "See SequenceEmpty."},
    "SplitToSequence": {"support": expand("N", "N", "N", "N"), "notes": "Use Split instead."},
    # Control flow
    "If": {
        "support": expand("P", "P", "S", "S"),
        "notes": "Both branches must produce identical output shapes and dtypes.",
        "limitations": ["Branch outputs with differing shapes fail at engine build."],
    },
    "Loop": {
        "support": expand("P", "P", "P", "P"),
        "notes": "Requires a static or shape-inferable trip count. Dynamic trip counts fail.",
        "remediation": "Replace dynamic-length loops with fixed iteration counts at export time.",
        "limitations": [
            "Nested loops are not supported.",
            "Carried tensors must keep stable shapes.",
        ],
    },
    "Scan": {
        "support": expand("P", "P", "P", "P"),
        "notes": "Sequence length must be known at build time.",
    },
    # Misc
    "Constant": {"support": expand("S", "S", "S", "S")},
    "ConstantOfShape": {"support": expand("S", "S", "S", "S")},
    "ArgMax": {"support": expand("S", "S", "S", "S")},
    "ArgMin": {"support": expand("S", "S", "S", "S")},
    "TopK": {"support": expand("S", "S", "S", "S")},
    "NonZero": {"support": expand("P", "S", "S", "S")},
    "NonMaxSuppression": {
        "support": expand("P", "S", "S", "S"),
        "notes": "Single-class NMS only pre-8.6. Use EfficientNMS plugin for batched NMS.",
    },
    "Range": {"support": expand("S", "S", "S", "S")},
    "OneHot": {"support": expand("S", "S", "S", "S")},
    "Identity": {"support": expand("S", "S", "S", "S")},
    "Dropout": {
        "support": expand("S", "S", "S", "S"),
        "notes": "Folded out during TRT engine build.",
    },
    "Trilu": {
        "support": expand("N", "P", "S", "S"),
        "notes": "Added in TRT 8.6 plugin set, native 10.0+.",
    },
    "EyeLike": {"support": expand("N", "P", "S", "S")},
    "Det": {"support": expand("N", "N", "N", "N"), "notes": "No TRT support."},
    "RoiAlign": {"support": expand("P", "S", "S", "S"), "notes": "Required for detection models."},
    "DepthToSpace": {"support": expand("S", "S", "S", "S")},
    "SpaceToDepth": {"support": expand("S", "S", "S", "S")},
}


def build_matrix() -> dict[str, Any]:
    """The full matrix document, as a dict. Pure; safe to call from tests.

    Returns a deep copy so callers can mutate the result without corrupting the
    module-level ``OPERATORS`` source of truth.
    """
    return {
        "schema_version": "1.0",
        "last_updated": "2026-05-21",
        "target_trt_versions": list(VERSIONS),
        "operators": copy.deepcopy(OPERATORS),
    }


def render_matrix_json(matrix: dict[str, Any]) -> str:
    """Serialize the matrix exactly as it is written to disk."""
    return json.dumps(matrix, indent=2, sort_keys=False) + "\n"


def main() -> None:
    matrix = build_matrix()
    out = Path(__file__).resolve().parent.parent / "trtcheck" / "data" / "operator_matrix.json"
    out.write_text(render_matrix_json(matrix))
    print(f"Wrote {out} with {len(OPERATORS)} operators.")


if __name__ == "__main__":
    main()
