# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [0.6.0] - 2026-05-21

### Added
- Documentation site at <https://sohams25.github.io/trtcheck/> built with
  mkdocs-material, deployed by `.github/workflows/docs.yml` on pushes that
  touch `docs/`, `mkdocs.yml`, the operator matrix, or the generator.
- `tools/build_operator_docs.py` renders one markdown page per operator
  from `operator_matrix.json`. Idempotent; the workflow's drift check
  refuses pages that disagree with the matrix.
- Validation harness scaffolding under `bench/`:
  - `manifest.yaml` (3 public ONNX urls + 6 bundled fixtures)
  - `fetch.py` for url-source caching with SHA-256 verification
  - `score.py` for confusion-matrix scoring against an outcomes file
  - `tools/validate_bench_manifest.py` schema check
  - `bench/README.md` for the GPU-required end-to-end flow
- All five top-level docs pages: index, install, usage, fixers, and the
  generated operators index.

### Note

No 0.5.0 release was published; the validation harness is tooling-only
and shipped under this version.

## [0.4.0] - 2026-05-21

### Added
- Composite GitHub Action at the repo root (`action.yml`). Runs trtcheck
  against PR-changed `*.onnx` files and emits a sticky-comment markdown
  body as an artifact. Inputs: `version`, `target-trt`, `severity`,
  `fail-on`, `paths`, `changed-only`. Outputs: `report-json`, `comment-md`,
  `critical-count`, `warning-count`, `status`.
- Helper scripts under `action/`:
  - `run.sh` discovers files and aggregates per-file JSON reports.
  - `render_comment.py` formats the aggregate as markdown.
  - `post_comment.py` upserts the sticky PR comment via the GitHub REST.
- Example consumer workflows under `.github/workflows/example-consumer/`
  showing the safe dual-workflow pattern for fork PRs.
- Dogfood `selftest.yml` that runs the action against the bundled
  fixtures on every push.

### Security
- Strict allowlists on `inputs.paths` and `inputs.version`.
- All ONNX-derived strings are HTML/markdown-escaped before being
  rendered into the sticky comment.

## [0.3.0] - 2026-05-21

### Added
- Three new built-in fixers in `trtcheck/fixers/`:
  - `float64_to_float32` casts FLOAT64 initializers to FLOAT32 when no
    value is NaN, infinite, or exceeds FP32 range.
  - `drop_dropout` removes Dropout nodes and rewires consumers. Skips
    nodes whose mask output is referenced.
  - `upsample_to_resize` rewrites deprecated Upsample ops as Resize
    when the mode is nearest or linear and the graph opset is >= 13.
- `tools/check_matrix_drift.py` and an offline fixture for it. Compares
  the bundled operator matrix against upstream onnx-tensorrt docs.

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
