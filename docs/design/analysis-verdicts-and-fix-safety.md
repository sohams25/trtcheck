# Design: analysis verdicts and fix safety

Status: implemented (schema 2.0, unreleased). This document records the
invariants the verdict model and the fixer pipeline are built on, so future
changes can be checked against them.

## Trust model

trtcheck runs on **untrusted input** (arbitrary ONNX protobufs in CI) and
loads **semi-trusted extensions** (entry-point plugins the user installed).

- Model-derived text (node names, producer strings, filenames) is sanitized
  before reaching a terminal or an HTML document (`trtcheck/_text.py`,
  reporter-level escaping).
- Built-in checker crashes propagate: they are trtcheck bugs and must fail
  tests. Third-party checker crashes are isolated into a
  `TRT-PLUGIN-CHECKER-ERROR` finding — a crashed checker means *missing
  coverage*, which must be visible in the report, not swallowed.
- Third-party fixers get no more trust than built-ins: every fixer runs
  inside the same transaction (below).
- Runtime verification shells out to `trtexec` with list-args (no shell),
  a timeout, and captured/truncated output.

## The verdict lattice

A single boolean ("conversion likely") could not express the difference
between "we checked and found nothing" and "we could not check". Schema 2.0
uses four states:

```
BLOCKED  >  UNVERIFIED  >  LIKELY  <  VERIFIED
```

- `BLOCKED`: >= 1 critical finding. Runtime success never overrides a
  static blocker — a contradiction there means the matrix is wrong and must
  be surfaced, not papered over.
- `UNVERIFIED`: no critical, but >= 1 finding with `verify_required=True`.
  Sources: unclassified operators, custom domains without a declared
  plugin, partial support, unresolvable conditional-support rules,
  crashed plugin checkers.
- `LIKELY`: no critical, nothing unresolved. Static prediction only.
- `VERIFIED`: `LIKELY`/`UNVERIFIED` + a successful `trtexec` parse/build
  for the user's actual environment. Only the runtime path sets it.

**Invariant:** no operator disappears silently. Every node is either
classified by the matrix, covered by an explicit uncertainty finding
(`TRT-OP-UNCLASSIFIED` / `TRT-OP-CUSTOM-DOMAIN`), or explicitly declared
plugin-backed by the user (`--plugin-domain`).

The deprecated `conversion_likely` boolean survives as
`verdict != BLOCKED` for 1.x JSON consumers.

## Evidence and confidence

Findings carry `confidence` (`high` = documented/tested, `medium` =
heuristic with known gaps, `low` = uncertainty marker) and matrix
conditional-support entries carry an `evidence` object
(`status`: `official_docs` | `empirically_verified` | `inferred` |
`unknown`, plus `source` URL and retrieval date). The drift checker
(`tools/check_matrix_drift.py`) compares the matrix against the upstream
onnx-tensorrt operator table on a schedule.

## Conditional support (matrix schema 2.x)

`operator_matrix.json` entries may carry `conditions`:

- `attribute_allowed` — a node attribute must be in an allowed set
  (`default_ok` covers the absent-attribute case);
- `constant_input_max` — an input must, when statically constant, be a
  scalar int <= `max_value`; when it is runtime-dynamic the condition is
  *unresolvable*, which produces an unverified finding rather than a pass
  or a guess.

Each condition evaluates to pass / violated (`TRT-OP-CONDITION`, severity
from the data) / unresolved (`TRT-OP-CONDITION-UNRESOLVED`,
`verify_required`). Unknown condition kinds (data newer than code) evaluate
to unresolved, never to pass. `applies_to` scopes a condition to the TRT
versions the evidence actually covers.

## Transactional fixing

`run_fixers()` (trtcheck/fixers/__init__.py) enforces:

1. The input model is never mutated (outer deep copy).
2. Each fixer runs against a fresh deep copy of the **last valid model**.
3. A candidate is committed only if the fixer returned well-formed
   `FixApplied` records *and* the candidate passes validation.
4. Any failure — exception (even after mutating), malformed return,
   invalid candidate, undeclared mutation — discards the candidate and
   records a `FixFailure`. Later fixers still run on the last valid state.

### Validation levels

A candidate is held to the strongest bar the *input* model meets:

- `full` — `onnx.checker.check_model(full_check=True)` (strict type/shape
  inference). This is the level that catches dtype rewrites the shallow
  checker misses (the Reshape INT64 regression).
- `basic` — structural check only; used for external-data models (full
  inference cannot read payloads from an in-memory proto) and inputs that
  fail full inference for pre-existing reasons.
- `none` — input fails even the basic check. Library callers may still run
  fixers; the CLI refuses to `--fix` such models.

### Schema-aware INT64 conversion

`Int64ToInt32Fixer` converts an initializer only when **every** use, across
all nested subgraphs, is at an input position whose ONNX type constraint
admits `tensor(int32)` *independently of other inputs/outputs* (allowlist:
Gather/GatherElements/ScatterElements `indices`, Cast/Shape/Size data
input). It refuses shadowed names, signature tensors (graph inputs and
outputs), custom-domain consumers, unknown positions (Reshape `shape`,
Slice `starts`, ...), overflow, and dead initializers. It never inserts
speculative Cast nodes to force a conversion through.

### Dropout removal

Removal requires provable inference mode: opset <= 6 `is_test=1`;
opset 7–11 unconditionally inference; opset >= 12 `training_mode` absent or
resolvable to a static scalar `False` (initializer or Constant, unambiguous
across scopes). True / dynamic / computed / ambiguous → refuse.

## The --fix pipeline

```
analyze(target) -> run_fixers (transactional) -> validate -> re-analyze(same target)
   -> diff findings by (rule_id, node_name, operator) -> resolved/remaining/introduced
   -> write only a validated candidate, never over the input
```

Dry-run performs everything except the write. `--format json` emits the
full machine-readable summary.

## Extension points

- Checkers/fixers/reporters via entry points (`docs/design/plugin-sdk.md`).
  New Issue fields all default, so 1.x-era plugin checkers keep working;
  their findings simply carry empty `rule_id`.
- New conditional-support kinds: add an evaluator in
  `checkers/operator_support.py`; unknown kinds are already fail-safe.
- Runtime verification is deliberately isolated in
  `trtcheck/runtime_verify.py`; static analysis never imports TensorRT.
