# Loop

## TensorRT support

| Version | Status |
| --- | --- |
| 8.0 | partial |
| 8.6 | partial |
| 10.0 | partial |
| 10.3 | partial |

## Notes

Requires a static or shape-inferable trip count. Dynamic trip counts fail.

## Limitations

- Nested loops are not supported.
- Carried tensors must keep stable shapes.

## Remediation

Replace dynamic-length loops with fixed iteration counts at export time.
