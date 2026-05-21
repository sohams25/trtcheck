# SequenceEmpty

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | not_supported |
| 8.6 | not_supported |
| 10.0 | not_supported |
| 10.3 | not_supported |

## Notes

Sequence ops are not supported by TensorRT. Emitted when PyTorch code uses List[Tensor].

## Remediation

Replace List[Tensor] with torch.stack() or pre-allocate a tensor.

## See also

- https://github.com/onnx/onnx-tensorrt/issues/1044
