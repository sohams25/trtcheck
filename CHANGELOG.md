# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [0.2.1] - 2026-05-21

### Changed
- `--diff --format html` now produces a true side-by-side layout via
  `HTMLReporter.render_diff()`. Each side wraps a `render_fragment()`
  output with its filename above; the grid collapses to one column on
  narrow viewports.

## [0.2.0] - 2026-05-21

### Added
- `--fix` and `--dry-run` flags to automatically rewrite known-safe
  failure patterns into a new ONNX file.
- Two built-in fixers in `trtcheck/fixers/`:
  - `int64_to_int32` casts INT64 initializers to INT32 when values fit.
  - `uint8_input` promotes a UINT8 graph input to FLOAT when the only
    consumer is a Cast to FLOAT.
- `--fix` refuses to overwrite the input file and (without `--force`)
  refuses to overwrite any pre-existing `--output` file.

## [0.1.1] - 2026-05-21

### Added
- `--force` flag to overwrite an existing `--output` file.
- `--max-model-size` flag (default 500 MB) to refuse oversized ONNX files
  before loading.
- `HTMLReporter.render_fragment()` for safe composition of multiple
  reports into one document.

### Changed
- `AnalysisReport` counts and verdict are now derived properties.
  Callers no longer need to call `recompute_counts()` or
  `derive_verdict()` explicitly.

## [0.1.0] - 2026-05-21

First public release.

### Added
- Five static checkers for ONNX -> TensorRT compatibility:
  - operator support against a curated TRT 8.0 to 10.3 matrix
  - precision (UINT8, INT64, FLOAT64, BFLOAT16, string tensors)
  - dynamic shape detection
  - control flow (Loop trip counts, nested Loop, If, Scan)
  - graph structure (missing outputs, duplicate names, large constants)
- `trtcheck` CLI with `--target-trt`, `--format` (console/json/html),
  `--output`, `--severity`, `--verbose`, `--diff`, `--version`.
- Self-contained HTML reporter, rich-based console reporter, JSON reporter.
- Bundled operator matrix (100 ops) and remediation database (20 entries).
- Six deterministic ONNX fixtures for the test suite.
- GitHub Actions CI on Python 3.10, 3.11, 3.12.

### Known limitations
- No auto-fix mode (`--fix` is on the v0.2 roadmap).
- `--diff` shows two reports back to back, not a true side-by-side diff.
- Operator matrix is hand maintained; quarterly refresh tooling is on the
  v0.2 roadmap.
