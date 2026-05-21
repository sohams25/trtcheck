# Usage

## Basic check

```bash
trtcheck model.onnx
```

Exit code is `1` if conversion is unlikely, `0` otherwise. Wire that into
CI to catch regressions at PR time.

## Common flags

```bash
# target a specific TensorRT version
trtcheck model.onnx --target-trt 8.6

# machine-readable output for CI
trtcheck model.onnx --format json --output report.json

# self-contained HTML report
trtcheck model.onnx --format html --output report.html

# filter to blockers only
trtcheck model.onnx --severity critical

# compare two ONNX files
trtcheck before.onnx after.onnx --diff --format html --output diff.html

# auto-fix simple cases (INT64 indices, UINT8 inputs after Cast, etc.)
trtcheck model.onnx --fix --dry-run --output fixed.onnx
trtcheck model.onnx --fix --output fixed.onnx
```

## All flags

| Flag | Default | Meaning |
|---|---|---|
| `--target-trt` | `10.3` | TensorRT version to check against |
| `--format` | `console` | `console`, `json`, or `html` |
| `--output` | (stdout) | Write the report to this path |
| `--severity` | `info` | Minimum severity to include |
| `--verbose` | off | Lowers severity threshold to `info` |
| `--diff` | off | Compare two ONNX files |
| `--force` | off | Allow `--output` to overwrite existing files |
| `--max-model-size` | `500` | Refuse to load ONNX files larger than this (MB) |
| `--fix` | off | Apply built-in fixers; writes to `--output` |
| `--dry-run` | off | With `--fix`, print changes without writing |

## CI integration

A composite GitHub Action ships in this repo. See
[the README](https://github.com/sohams25/trtcheck#using-trtcheck-as-a-github-action)
for the dual-workflow setup that posts a sticky PR comment when ONNX
files change.
