# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Added
- `SCORECARD.md`: first published run of the `bench/` validation harness.
  9-model corpus (3 ONNX Model Zoo models + 6 bundled fixtures), scored at
  the CI gate configuration (`--severity critical`, TRT 10.3): precision
  1.000, recall 1.000. Raw predictions committed as `bench/outcomes.json`.
- `bench/predict.py`: the trtcheck leg of the harness as a real script
  (replaces the inline wrapper in `bench/README.md`); defaults to the
  invoking interpreter's trtcheck, writes `bench/outcomes.json`.
- New critical check `loop_runtime_trip_count`: a Loop trip count fed
  straight from a graph input is runtime-dynamic by construction and always
  fails the TRT engine build, so it now fails the verdict. Found by the
  first scorecard run — the fixture scored as a false negative at
  `--severity critical`. A trip count computed by an internal node (possibly
  shape-inferable) keeps the `loop_dynamic_trip_count` warning.
- Scheduled `matrix-drift` GitHub Action (`.github/workflows/matrix-drift.yml`).
  Runs `tools/check_matrix_drift.py` weekly (Mondays at 04:17 UTC) plus
  on `workflow_dispatch`. On drift, opens or refreshes a single rolling
  `[matrix-drift] YYYY-MM-DD` issue with the per-operator diff and a
  triage recipe. Auto-closes the issue when the matrix is back in sync.
- Always-on performance + memory gate: a synthetic few-thousand-node graph is
  analyzed in CI with wall-clock and `tracemalloc` budgets, tripping on an
  accidental O(n^2)/per-node-allocation regression (no GPU/assets needed).
- Generator drift guards: tests assert `operator_matrix.json` and every
  `docs/operators/*.md` exactly match `tools/build_operator_matrix.py` /
  `build_operator_docs.py` output, and that no operator page is stale/missing.
- Version-sync test: `pyproject` version, `trtcheck.__version__`, and the
  Action's default `version` input must agree.

### Changed
- `bench/manifest.yaml`: replaced the dead `yolov8n_static` URL (asset no
  longer published) with `squeezenet1_1` from the ONNX Model Zoo, and pinned
  SHA-256 hashes for all three URL-sourced models.
- `tools/check_matrix_drift.py` is version-aware: it locates the upstream status
  column by header (supporting "TensorRT"/"TRT" and non-second-column layouts),
  tags it with the TRT major, and refuses to diff a `--target` the table does
  not cover instead of emitting spurious mismatches. Multiple version columns
  now warn loudly.
- `tools/build_operator_matrix.py` exposes a pure `build_matrix()` (deep-copied)
  and writes via an absolute path, so regeneration works from any directory.
- `remediation_db.json` is now the single source of truth for the
  explanation / remediation / docs_link / severity of the precision,
  control-flow, dynamic-shape, and graph-structure findings (loaded via the new
  `trtcheck.remediation` module); the checkers no longer hard-code that text, so
  it can't drift. Finding messages now end with the DB explanation, and the
  UINT8/INT64 precision findings now carry a docs link. (operator support keeps
  using `operator_matrix.json`.)

### Removed
- `raw_trtcheck.md` (internal spec brief) and `docs/screenshot.svg` (orphaned,
  unreferenced) no longer ship in the repo.

### Fixed
- **Subgraph blind spot (correctness).** Every checker except control-flow
  scanned only the top-level graph, and control-flow recursed only via the
  `body` attribute. An unsupported op hidden in an `If` / `Loop` / `Scan`
  branch was reported as "likely to convert". Checkers and fixers now descend
  into all subgraph bodies through a shared, depth-bounded walker
  (`trtcheck._graph`).
- **INT64 graph inputs** are now flagged (the most common real TRT input
  problem; previously only INT64 *initializers* were caught).
- **FLOAT64 introduced inside the graph** (via `Cast(to=DOUBLE)` or a `DOUBLE`
  `Constant`) is now detected, not just FLOAT64 at the boundary.
- **Dynamic dims encoded as `-1`** are now treated as symbolic, so a
  fully-dynamic `[-1,-1,-1,-1]` input is no longer read as static.
- `int64_to_int32` no longer crashes on an empty INT64 initializer.
- Corrupt / truncated / non-ONNX files now produce a clean error message
  instead of a raw protobuf traceback (all CLI paths).
- `--diff` to an existing file now honours `--force` for console/text output.
- `--format console --output FILE` writes plain text instead of raw ANSI codes.
- Long, unbroken remediation text in the console table is folded instead of
  ellipsis-truncated, so the exact fix is never dropped.
- `python -m trtcheck` now works (added `trtcheck/__main__.py`).
- `pyproject.toml` project URLs pointed at a non-existent GitHub repo.
- `pyyaml` added to `[project.optional-dependencies].dev`; the bench test
  modules import it transitively, so a fresh `pip install -e ".[dev]"` (what
  CI runs) failed pytest collection without it.

### Security
- GitHub Action: the `base-ref` input is validated before reaching `git`,
  closing an option-injection / arbitrary-write vector; `pip install` uses
  `--` to stop option parsing of the version/source-path inputs.
- Sticky-comment renderer neutralizes `|` (and `&`) in model-derived strings
  so a hostile operator/node name cannot break out of the markdown table.
- `release.yml` now declares least-privilege `permissions`.
- Console/JSON reporters strip control characters and neutralize Rich markup
  from model-derived strings (the HTML reporter was already escaped).
- `bench/fetch.py` validates a manifest entry's `name` (rejecting separators and
  dot-only/traversal forms) and confirms the resolved path stays under the cache
  directory before writing, closing a path-traversal write.

### Changed
- Only third-party **plugin** checkers are isolated behind failure handling;
  exceptions from built-in checkers now propagate (a crash there is a trtcheck
  bug and should be loud, not silently downgraded to a warning).
- `total_nodes` in the report now counts nodes inside subgraphs.
- CI tests Python 3.13 and enforces a 90% coverage floor; the smoke test
  exercises both the console script and `python -m trtcheck`.

## [1.0.0] - 2026-05-21

First stable release. The public extension API is now frozen; semver
applies from here onward.

### Added
- New `trtcheck.plugins` module holding the public `Checker`, `Fixer`,
  and `Reporter` Protocols. The Protocols are `@runtime_checkable` so the
  plugin loader can `isinstance`-validate discovered classes.
- Entry-point discovery: third-party packages can declare plugins under
  the `trtcheck.checkers`, `trtcheck.fixers`, and `trtcheck.reporters`
  groups. `trtcheck.plugins.load_plugins()` walks them on Analyzer
  construction.
- Per-plugin failure isolation: a checker that raises during analysis is
  caught and surfaced as a `WARNING` issue instead of killing the run.
- New CLI flags `--list-plugins` and `--disable-plugin NAME` (repeatable).
- New `AnalyzerConfig` fields: `discover_entry_point_plugins: bool = True`
  and `disable_plugins: list[str]`.
- `examples/trtcheck-extra-fixers/` ships a worked out-of-tree plugin
  example with project layout, `pyproject.toml`, a trivial fixer, and a
  walk-through README.
- `docs/design/plugin-sdk.md` documents the v1.0 surface, discovery
  semantics, error model, and back-compat policy.

### Changed
- The existing import paths (`trtcheck.checkers.Checker`,
  `trtcheck.fixers.Fixer`, `trtcheck.reporters.Reporter`) now re-export
  from `trtcheck.plugins`. Object identity is preserved; v0.x callers
  continue to work unchanged. New code should import directly from
  `trtcheck.plugins`.

### Compatibility

No breaking changes. v0.x callers that import the Protocols, use
`Analyzer` / `analyze()`, or run the CLI keep working. The CLI surface
gains flags but does not remove any.

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
