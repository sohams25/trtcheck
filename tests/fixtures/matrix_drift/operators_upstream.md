# Supported ONNX Operators

This document lists the ONNX operators currently supported by the TensorRT
ONNX parser.

| Operator | TensorRT 10.x | Restrictions |
|----------|---------------|--------------|
| Abs | Y | |
| Add | Y | |
| ArgMax | Y | |
| Conv | Y | |
| GroupNormalization | Y | Added in TRT 10.0 |
| LayerNormalization | Y | |
| MatMul | Y | |
| NewlyAddedOp | Y | Hypothetical op upstream knows but our matrix does not |
| Relu | Y | |
| SequenceEmpty | N | Sequence ops not supported |
| Mish | N | This row contradicts our matrix on purpose |
