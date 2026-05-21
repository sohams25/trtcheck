# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com).

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
