# Resize

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | partial |
| 8.6 | partial |
| 10.0 | supported |
| 10.3 | supported |

## Notes

Only nearest and linear modes; cubic is not supported (onnx-tensorrt docs, retrieved 2026-07-22).

## Limitations

- Antialiasing (antialias=1) is not supported.
- coordinate_transformation_mode limited to half_pixel, pytorch_half_pixel, tf_half_pixel_for_nn, asymmetric, align_corners.
