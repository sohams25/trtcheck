# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [1.1.0] - 2026-07-22

### Added
- Real-TensorRT smoke validation (2026-07-22): the 7-model corpus ran
  against genuine TensorRT 10.3.0 (official NGC `tensorrt:24.08-py3`
  container, RTX 4050 Laptop GPU) using the installed wheel — 5 genuine
  engine builds (including the `--fix` output and the Reshape-INT64
  regression model), 2 genuine parser failures (SequenceEmpty, custom
  domain without plugin), and zero disagreements between
  `--verify-runtime` and independent direct `trtexec` runs. New
  repo-owned runner: `scripts/real-smoke-container.sh` +
  `scripts/real_tensorrt_smoke.py`; evidence in
  `REAL_TENSORRT_VALIDATION_REPORT.md` and
  `bench/real_tensorrt_smoke_results.json`. Recorded TensorRT-10.3
  trtexec behavior: dynamic models without shape flags are auto-overridden
  to 1x1x1x1 (warning), not rejected.
- Release-readiness pass (2026-07-22): a recorded trtexec parser/build
  *failure* now demotes an otherwise-`likely` verdict to `unverified`
  (runtime evidence against the model is never hidden behind a clean
  static prediction); fixers that report changes without modifying the
  model are rejected; plugin checker findings without a `rule_id` get a
  namespaced `PLUGIN-<name>` fallback; `--fix` before/after identity now
  includes the owning graph scope; `bench/score.py --json` emits a
  machine-readable summary; `scripts/package-smoke.sh` installs the built
  wheel into a fresh venv and exercises the CLI from outside the repo;
  `SECURITY_REVIEW.md` documents the reviewed surfaces and trust model.
- **Four-state verdict model.** `AnalysisReport.verdict` is now one of
  `blocked` / `unverified` / `likely` / `verified` (`trtcheck.Verdict`).
  `unverified` is new: no known blocker, but unresolved conditions remain.
  The boolean `conversion_likely` survives as a deprecated compatibility
  property (`verdict != blocked`). Exit codes unchanged by default
  (`1` on blocked); new `--fail-on unverified` tightens the gate.
- **Stable rule ids + report schema 2.0.** Every finding carries
  `rule_id` (e.g. `TRT-OP-UNSUPPORTED`, `TRT-DTYPE-UINT8-INPUT`),
  `confidence` (high/medium/low), `verify_required`, `target_trt`, and
  `graph_scope`. The registry is documented in `docs/rules.md` and pinned
  by a stability test. All 1.x JSON keys are preserved.
- **Honest uncertainty for unknown operators.** Default-domain ops absent
  from the support matrix now emit `TRT-OP-UNCLASSIFIED` (info,
  aggregated per op type) instead of silently passing; custom-domain ops
  emit `TRT-OP-CUSTOM-DOMAIN` unless the domain is declared plugin-backed
  via `--plugin-domain` / `AnalyzerConfig.plugin_domains`.
- **Conditional operator support (matrix schema 2.0).** Matrix entries may
  carry evidence-backed `conditions` (`attribute_allowed`,
  `constant_input_max`, scoped by `applies_to` TRT versions). Violations
  emit `TRT-OP-CONDITION`; statically unresolvable conditions emit
  `TRT-OP-CONDITION-UNRESOLVED` (unverified). First converted operators,
  sourced from the upstream onnx-tensorrt table (retrieved 2026-07-22):
  TopK (`sorted=1` required, `K < 3840`) and Resize (nearest/linear only,
  restricted `coordinate_transformation_mode`, no antialias). `Clip`
  added to the matrix (TRT 10.x supported per the same source; 8.x left
  unknown).
- **Optional runtime verification.** `--verify-runtime` runs
  `trtexec --onnx=MODEL` (list-args, timeout, captured output; no shell)
  and records status/version/command metadata in the report. Only a
  successful build upgrades the verdict to `verified`; a missing trtexec,
  timeout, or failure leaves the static verdict untouched. Statuses:
  success / parser_failure / build_failure / missing_trtexec / timeout.
- **Audited `--fix` pipeline.** `--fix` now analyzes with the selected
  `--target-trt`, applies fixers, re-analyzes with the same target, and
  reports findings resolved / remaining / introduced (keyed by rule id +
  node identity). `--format json` emits a machine-readable fix summary.
  Structurally invalid input models are refused.
- Bench harness three-way vocabulary: predictions may be `unverified`,
  which is excluded from the blocker confusion matrix and reported as
  coverage (never counted as success). New corpus fixtures:
  `topk_unsorted`, `custom_domain`, `reshape_int64_shape`.
- `docs/rules.md` (rule registry) and
  `docs/design/analysis-verdicts-and-fix-safety.md` (invariants, trust
  model, transactional fixing).

### Fixed
- **`Int64ToInt32Fixer` no longer corrupts models whose INT64 initializers
  feed schema-required-INT64 inputs** (`Reshape` shape, `Slice`
  starts/ends, ...). The old "fits in INT32" rule produced models that
  passed the shallow checker but failed strict type inference. The fixer
  is now use-aware: it converts only when every use (including nested
  subgraph captures) is at an allowlisted INT32-compatible position, and
  refuses shadowed names, signature tensors, custom-domain consumers,
  overflow, and dead initializers. It also no longer retypes an
  initializer that doubles as a graph input (that silently changed the
  model's public signature).
- **Fixers are transactional.** Every fixer (built-in or plugin) runs
  against an isolated deep-copy candidate that is committed only after it
  passes `onnx.checker.check_model(full_check=True)` (basic check for
  external-data models). A fixer that crashes mid-mutation, returns
  malformed records, mutates without declaring it, or emits an invalid
  model cannot affect the output; one failed fixer no longer prevents
  later fixers from running. Plugin tracebacks are hidden unless
  `TRTCHECK_DEBUG=1`.
- **`DropDropoutFixer` respects training mode.** Opset >= 12 Dropouts are
  removed only when `training_mode` is absent or statically false
  (initializer or Constant, unambiguous across scopes); true, dynamic,
  computed, or ambiguous training modes are left alone. Opset <= 6
  requires `is_test=1`.
- Resize matrix prose corrected against current upstream onnx-tensorrt
  docs: cubic mode and antialiasing are *not* supported in TRT 10.x (the
  old notes claimed both were added in 10.0).

### Changed
- Console/HTML headlines use conservative verdict wording ("LIKELY --
  static analysis found no known blocker" instead of "LIKELY TO
  CONVERT"; "CONVERSION BLOCKED" instead of "CONVERSION WILL FAIL"), and
  both reporters gained a Rule column plus the TRT target in the header.
- `bench/predict.py` runs the full report (dropping `--severity
  critical`) because verdicts require the INFO-level uncertainty
  findings; it maps schema-2.0 verdicts with a 1.x fallback.
- The exit code is computed from the unfiltered report: `--severity`
  affects display only.

_Also in this release (earlier unreleased entries):_

### Fixed
- The README / case-study `--fix` walkthrough is reproducible again. The
  bundled `uint8_input.onnx` fixture was the degenerate
  input→Cast→output shape the fixer deliberately refuses (rewriting it
  would leave a node-less graph), so the documented demo silently
  no-opped. The fixture now carries a downstream Relu — the realistic
  preprocessing-cast pattern — and a regression test pins the full
  fixture→fix→re-check contract. Case study and demo SVG updated to the
  real CLI output.
- Plugin fixers and reporters are now actually executed. Discovery
  worked, but `--fix` only ran the five built-ins and `--format` was a
  closed choice, so a third-party fixer (like the shipped
  `strip_identity` example) was listed by `--list-plugins` and then
  ignored. `--fix` now appends discovered fixers (isolated: a crashing
  plugin is skipped with a stderr warning), and `--format` accepts a
  plugin reporter name alongside `console`/`json`/`html`.
- `--disable-plugin` with a name that matches nothing now warns on
  stderr instead of silently no-opping, so a typo can't leave a checker
  enabled unnoticed.
- Demo SVG: the issue-table header used run-length spaces for fake
  column alignment, which SVG collapses — it rendered as one cramped
  line. Header cells now share the data rows' column positions, and the
  depicted output matches the CLI verbatim.

### Removed
- `--verbose/--quiet`: with `--severity` defaulting to `info`, the flag
  was a documented no-op. Use `--severity` directly.
- `estimated_fusions` / `estimated_precision` from `AnalysisReport` and
  the JSON report: spec-era scaffolding that nothing ever populated,
  removed before anyone can depend on two always-empty keys. (If you
  parsed them, they were always `[]` / `{}`.)
- The GitHub Pages site, its deploy workflow (`docs.yml`), and
  `mkdocs.yml`. No project website; the docs stay in the repo as plain
  markdown under `docs/`, which GitHub renders directly.

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
- Operator matrix: `Scan` on TRT 10.0/10.3 corrected from `partial` to
  `supported`, matching the upstream onnx-tensorrt table (the one real
  mismatch `tools/check_matrix_drift.py` reported). The
  static-sequence-length caveat stays as the operator note and the
  control-flow warning.
- README Action-inputs table now lists `base-ref` and `source-path`;
  assorted doc drift fixed (CI matrix in CONTRIBUTING, Python badge,
  flag tables, stale fixture paths).
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
