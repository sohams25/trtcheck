# GroupNormalization

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | not_supported |
| 8.6 | not_supported |
| 10.0 | supported |
| 10.3 | supported |

## Notes

Added in TRT 10.0. Replace with BatchNormalization for older TRT.

## Remediation

Replace nn.GroupNorm with nn.BatchNorm2d, or upgrade TRT to 10.0+.
