# Overnight hardening report — trtcheck

Branch: `claude/trtcheck-hardening` (off `main` @ 374fe48). Nothing pushed,
published, tagged, or released. Date: 2026-07-22.

## 1. Executive summary

trtcheck is now a release-candidate static pre-flight checker **and safe
repair tool** for ONNX → TensorRT deployment. The three transformation
soundness bugs (INT64 schema corruption, non-transactional fixers, Dropout
training-mode) are fixed with regression proofs; analysis is honest about
what it cannot know (four-state verdict, per-finding confidence, no
silently-ignored operators); diagnostics are CI-stable (rule-id registry,
JSON schema 2.0 that is a strict superset of 1.x); the compatibility matrix
can express evidence-backed conditional support; and an optional, fully
isolated `trtexec` runtime-verification path exists. All 445 tests, strict
mypy, black, and isort pass.

## 2. Baseline

Environment: Linux 6.8, repo venv (`.venv`, Python 3.10), onnx 1.21.0,
click 8.4.1, numpy 2.4.6. No TensorRT, no GPU, no Docker. Network access
was available (used only to retrieve the upstream onnx-tensorrt operator
table for evidence).

Baseline commands and results (clean tree at 374fe48):

- `./scripts/run-tests.sh -q` → **380 passed, 1 skipped** (benchmark
  opt-in), 1.64 s
- `.venv/bin/mypy trtcheck/ --strict` → **Success: no issues in 25 files**
- `.venv/bin/black --check .` → clean; `.venv/bin/isort --check-only .` → clean

## 3. Major architectural changes

1. **Verdict lattice** (`trtcheck/types.py`): `Verdict.{BLOCKED,
   UNVERIFIED, LIKELY, VERIFIED}` derived from findings;
   `conversion_likely` kept as a deprecated property. Invariant: no
   operator disappears silently — everything is classified, flagged as
   uncertain, or explicitly declared plugin-backed.
2. **Stable diagnostics** (schema 2.0): every `Issue` carries `rule_id`,
   `confidence`, `verify_required`, `target_trt`, `graph_scope`.
   Rule ids live in `remediation_db.json` + checker constants; the
   registry (`docs/rules.md`) is pinned by
   `tests/test_verdicts.py::test_rule_id_registry_is_stable`.
3. **Transactional fixer pipeline** (`trtcheck/fixers/run_fixers`):
   per-fixer deep-copy candidates, committed only after ONNX validation at
   the strongest level the input model itself passes (full → basic → none;
   the CLI refuses to fix models that fail even basic).
4. **Conditional capability data** (matrix schema 2.0): per-operator
   `conditions` with `applies_to` version scoping and `evidence`
   metadata; evaluators for `attribute_allowed` and `constant_input_max`;
   unknown kinds and unresolvable conditions fail toward *unverified*,
   never toward *pass*.
5. **Isolated runtime verification** (`trtcheck/runtime_verify.py`):
   list-args subprocess, timeout, truncated output capture, five distinct
   failure states; only a successful build sets `VERIFIED`, and never over
   a static `BLOCKED`.
6. **Audited `--fix`**: analyze(target) → transactional fix → validate →
   re-analyze(same target) → resolved/remaining/introduced diff by
   `(rule_id, node, operator)`; JSON summary via `--format json`.

## 4. Bugs reproduced and fixed

| Bug | Reproduction | Fix |
|---|---|---|
| `Int64ToInt32Fixer` corrupted models whose INT64 initializer feeds a schema-required-INT64 input (e.g. `Reshape` shape) | `tests/test_fixers_int64_schema.py::test_blind_conversion_breaks_full_validation` proves the old rewrite passes the shallow checker and fails `full_check=True` | Use-aware conversion: every use (incl. nested subgraph captures) must be at an allowlisted INT32-independent position (Gather/GatherElements/ScatterElements indices, Cast/Shape/Size input); refuses shadowing, signature tensors, custom domains, overflow, dead/empty initializers. No speculative Casts. |
| Fixers could leave partial mutations after a crash; a crashed plugin fixer's edits leaked into `--fix` output | `tests/test_fixers_transactional.py` (`_MutateThenCrash`, `_EmitsInvalidModel`, `_UndeclaredMutation`, malformed returns) | `run_fixers()` transaction; failures recorded as `FixFailure`; later fixers still run; tracebacks only with `TRTCHECK_DEBUG=1` |
| `DropDropoutFixer` removed Dropouts regardless of `training_mode` | `tests/test_fixers_dropout.py::TestDropoutTrainingMode` (true/dynamic/computed/ambiguous kept; absent/static-false removed; opset-6 `is_test`) | Static resolution of opset semantics + `training_mode` through initializers/Constant nodes with scope-ambiguity refusal |
| Unknown default-domain and custom-domain operators silently passed | `tests/test_verdicts.py` | `TRT-OP-UNCLASSIFIED` / `TRT-OP-CUSTOM-DOMAIN` info findings (aggregated per op type), `verify_required=True` → verdict `unverified`; `--plugin-domain` opt-out |
| `--fix` ignored `--target-trt` and never compared before/after | `tests/test_cli_fix.py` | Target-aware audited pipeline (above) |
| Matrix prose for Resize contradicted current upstream docs (claimed cubic + antialias supported in 10.0) | upstream onnx-tensorrt `operators.md`, retrieved 2026-07-22 | Notes corrected; conditions added; drift-checked |
| `mobilenetv2` scored `unverified` — `Clip` missing from the matrix | bench run in this session | `Clip` added: TRT 10.x supported (official docs), 8.x left `unknown` (no evidence claimed) |

## 5. Files changed (56 files, +3547/−351)

- **Core types/analysis**: `trtcheck/types.py`, `analyzer.py`,
  `remediation.py`, `checkers/operator_support.py` (rewritten),
  `checkers/{precision,control_flow}.py` (scope threading)
- **Fixers**: `fixers/__init__.py` (transactional pipeline),
  `fixers/int64_to_int32.py` (rewritten), `fixers/drop_dropout.py`
- **CLI/runtime**: `cli.py` (verdicts, `--fail-on`, `--plugin-domain`,
  `--verify-runtime`, audited `--fix`), new `runtime_verify.py`
- **Data**: `data/operator_matrix.json` + `data/remediation_db.json`
  (schema 2.0), `tools/build_operator_matrix.py` (source of truth updated),
  regenerated `docs/operators/*`
- **Bench**: `bench/predict.py`, `bench/score.py`, `bench/manifest.yaml`,
  `bench/outcomes.json`, new fixtures via `tests/fixtures/generate_broken.py`
- **Tests**: 6 new suites (`test_fixers_int64_schema`,
  `test_fixers_transactional`, `test_verdicts`, `test_conditions`,
  `test_cli_fix`, `test_runtime_verify`) + targeted updates
- **Docs**: `usage.md`, `fixers.md`, new `rules.md`, new
  `design/analysis-verdicts-and-fix-safety.md`, `design/plugin-sdk.md`,
  `README.md`, `SCORECARD.md`, `CHANGELOG.md`, case study, SVG wording,
  `CLAUDE.md`, `RELEASE_NOTES_DRAFT.md`

## 6. Public API / schema changes and migration

- JSON report: `schema_version: "2.0"`. **Superset of 1.x** — every old
  key including `conversion_likely` is still emitted. Migrate consumers
  from `conversion_likely` to `verdict`; filter findings on `rule_id`.
- `Issue` constructor: new fields all default → third-party checkers
  unchanged.
- `apply_all()` keeps its signature; it is now transactional under the
  hood. New `run_fixers()` additionally reports `FixFailure`s.
- Exit codes unchanged by default (1 = blocked). `--severity` no longer
  influences the exit code. `--fix` now refuses invalid input models and
  declines unsound INT64 conversions it previously performed —
  intentional soundness changes, documented in the changelog.

## 7. Tests and final check results

Added ~65 tests across the six new suites plus updated existing suites.

Final results (this session, exact commands):

- `./scripts/run-tests.sh -q` → **445 passed, 1 skipped** (~1.7 s)
- `.venv/bin/mypy trtcheck/ --strict` → **Success: no issues in 26 files**
- `.venv/bin/black --check .` / `.venv/bin/isort --check-only .` → clean

## 8. Runtime / TensorRT checks actually performed

**None.** No TensorRT, `trtexec`, or GPU exists in this environment. The
runtime-verification module is tested exclusively through mocked
subprocess calls; the scorecard states explicitly that ground truth is
documented TRT behavior, not live builds. To verify for real:
`trtcheck model.onnx --verify-runtime` on a machine with TensorRT, or the
`bench/README.md` GPU protocol (`trtexec --onnx=<model>` per manifest
entry, recorded into `outcomes.json` under the `trtexec` key).

Network evidence used: the upstream onnx-tensorrt operator table
(https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md,
retrieved 2026-07-22) for TopK/Resize/Clip entries.

## 9. Remaining limitations / deferred work

- Conditional-support data covers 2 operators (TopK, Resize) by design —
  the evaluation infrastructure exists; converting more operators is
  data-entry work with the same evidence discipline.
- `graph_scope` is populated where the owning graph is cheaply known
  (operator support, initializer precision, control flow); dynamic-shape
  and graph-input findings leave it empty (top-level by construction).
- The dtype columns of the upstream table (per-op supported dtypes) are
  not yet modeled as conditions.
- Real trtexec leg of the scorecard: hardware-only, procedure documented.
- The GitHub Action still summarizes via `conversion_likely` (works, but
  could surface the four-state verdict in the PR comment).

## 10. Suggested next version and release checklist

Suggested: **v1.1.0** (additive schema, behavior corrections). See
`RELEASE_NOTES_DRAFT.md` for the draft notes and the pre-release
checklist. Do not release from this branch without the checklist.

## 11. Local commits (oldest first)

1. `1d97144` feat: stable rule IDs, four-state verdict model, schema-aware
   fixers, transactional fix pipeline
2. `7240911` test: regression suites for schema-aware INT64, transactional
   fixing, dropout training-mode, verdicts, conditions, --fix CLI, runtime verify
3. `02b702e` feat(bench): three-way outcome vocabulary with unverified
   coverage metrics
4. `c26b051` docs: verdict model, rule registry, fix-safety design doc,
   honest scorecard
5. (final) docs/assets/report commit containing this file

## 12. Five-minute interview explanation

**Architecture.** trtcheck is a plugin-composable static analyzer: a thin
`Analyzer` walks an ONNX protobuf (including every nested If/Loop/Scan
subgraph) through independent `Checker`s that each return typed `Issue`s;
pure `Reporter`s render the aggregate; `Fixer`s are the only components
allowed to rewrite the graph, and they run inside a transactional pipeline
that validates every candidate before committing it. Checkers never
format, reporters never analyze, fixers never guess.

**Hardest correctness issue.** The INT64 auto-fixer. TensorRT prefers
INT32, so "downcast every in-range INT64 initializer" looks obviously
right — and passes ONNX's default checker. But ONNX type constraints are
positional: `Reshape`'s shape input *requires* INT64, and elementwise ops
bind one type variable across both operands, so retyping a single operand
breaks the model in ways only strict type inference catches. The fix is a
whole-model use index (subgraphs can capture outer-scope names) plus an
allowlist of input positions whose type variable binds only that input.
Everything else — shadowed names, signature tensors, custom-domain
consumers — is refused. The design lesson: a graph rewrite is only as
safe as your model of every consumer's schema, and the honest fallback is
refusal, not a speculative Cast.

**Trust model.** Input models are untrusted (CI runs on arbitrary
protobufs): parse errors become domain errors, model-derived text is
sanitized before terminals and HTML, traversal depth is bounded. Plugins
are semi-trusted: a crashed checker becomes a visible "missing coverage"
finding, and a crashed fixer physically cannot corrupt output because it
only ever touched a discarded copy. And trtcheck does not trust *itself*:
static analysis reports four verdicts, where "unverified" is a
first-class answer meaning "I checked and I cannot know" — only a real
TensorRT build, run explicitly and recorded with its environment, is
allowed to claim "verified".
