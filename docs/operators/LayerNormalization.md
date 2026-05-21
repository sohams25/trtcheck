# LayerNormalization

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | partial |
| 8.6 | supported |
| 10.0 | supported |
| 10.3 | supported |

## Notes

Native LayerNormalization op added in TRT 8.6. Pre-8.6 must decompose.

## Remediation

Use ONNX opset 17+ with native LayerNormalization, or upgrade TRT to 8.6+.
