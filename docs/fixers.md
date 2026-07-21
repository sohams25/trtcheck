# Auto-fixers

`trtcheck --fix` runs an **audited, transactional** pipeline of conservative
ONNX rewrites targeting common TensorRT conversion failures:

1. analyze the input for the selected `--target-trt`;
2. run every fixer against an isolated deep-copy candidate — a fixer that
   crashes, returns malformed records, mutates without declaring it, or
   produces an ONNX-invalid model has its changes discarded (and the
   failure reported), while later fixers still run;
3. validate the result with `onnx.checker.check_model(full_check=True)`
   where the input supports it (external-data models fall back to the
   basic check);
4. re-analyze with the same target and report findings **resolved /
   remaining / introduced**, keyed by stable rule id + node identity;
5. write only a validated model, never over the input file.

Nothing is ever half-rewritten, and no speculative `Cast` nodes are
inserted to make a fix "succeed". When a rewrite is not provably safe,
the fixer skips it (skips are logged at INFO level on the
`trtcheck.fixers` logger).

See `docs/design/analysis-verdicts-and-fix-safety.md` for the invariants.

## Built-in fixers

### `int64_to_int32`

Casts an INT64 initializer to INT32 **only when every use of it — across
all nested subgraphs — is at an input position whose ONNX schema accepts
INT32 independently of the operator's other inputs**: `Gather` /
`GatherElements` / `ScatterElements` indices, and the data input of
`Cast` / `Shape` / `Size`.

Refuses, among others:

- consumers whose schema requires INT64 (`Reshape` shape, `Slice`
  starts/ends/axes, `Pad` pads, `Tile` repeats, ...) — converting those
  produces a model that passes the shallow checker but fails strict type
  inference;
- elementwise consumers (`Add`, `Mul`, ...) — int32 is an allowed dtype
  there, but the type variable binds both operands, so retyping one
  breaks the model;
- initializers that shadow a graph input/output (signature change),
  names defined in multiple scopes, custom-domain consumers, values
  outside INT32 range, empty or unconsumed initializers.

### `float64_to_float32`

Casts FLOAT64 initializers down to FLOAT32. Refuses on `NaN`, `+/-inf`,
empty initializers, or any value exceeding FP32 range.

### `uint8_input`

Promotes a `UINT8` graph input to `FLOAT` when its only consumer is a
`Cast(to=FLOAT)`. Removes the redundant Cast and rewires downstream
nodes. Refuses any other UINT8 consumption pattern.

### `drop_dropout`

Removes a Dropout node **only when it is provably in inference mode**:

- opset >= 12: the optional `training_mode` input is absent, or resolves
  to a static scalar `False` (initializer or Constant node, unambiguous
  across scopes). `True`, runtime-fed, computed, or ambiguous
  training_mode → the node is left alone;
- opset 7–11: Dropout has no training switch (inference = identity);
- opset <= 6: only with `is_test=1`.

Also refuses when the mask output is referenced anywhere, when the data
output is captured by another scope, or when rewiring would change the
graph signature.

### `upsample_to_resize`

Rewrites leftover deprecated `Upsample` ops to `Resize` (mode `nearest`
or `linear`) on opset-13+ graphs. Refuses below opset 13 (the 4-input
Resize form only validates from 13; run
`onnx.version_converter.convert_version` first).

## How to invoke

```bash
# preview what would change (nothing is written)
trtcheck model.onnx --fix --dry-run

# write the fixed file, report before/after findings
trtcheck model.onnx --fix --output model.fixed.onnx

# machine-readable fix summary
trtcheck model.onnx --fix --output model.fixed.onnx --format json
```

`--fix` requires `--output` (unless `--dry-run`), refuses to overwrite the
input file, refuses structurally invalid input models, and honors
`--target-trt` for both the before and after analysis. Use `--force` to
overwrite an existing output. Exit code follows the *fixed* model's
verdict (`--fail-on` applies).

## Third-party fixers

Plugin fixers (entry point `trtcheck.fixers`) run inside the same
transaction as built-ins: a crashing or misbehaving plugin is reported on
stderr and cannot corrupt the output model. `TRTCHECK_DEBUG=1` includes
the traceback.

## Adding a fixer

Each fixer lives in `trtcheck/fixers/`, implements the `Fixer` protocol
(`fix(model) -> list[FixApplied]`), and is registered in
`default_fixers()`. Contract: mutate the model you are handed (the
pipeline gives you a private candidate copy), return one `FixApplied` per
change, and skip anything not unambiguously safe. Write the failing test
first.
