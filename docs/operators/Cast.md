# Cast

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | partial |
| 8.6 | partial |
| 10.0 | supported |
| 10.3 | supported |

## Notes

UINT8 -> INT8 cast unreliable pre-10.0. FLOAT64 source always rejected.

## Remediation

Cast inputs to FLOAT32/INT32 before exporting from PyTorch.
