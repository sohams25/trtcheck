# Auto-fixers

`trtcheck --fix` runs a pipeline of conservative ONNX rewrites that target
the most common TensorRT conversion failures. Each fixer either applies
the rewrite cleanly or refuses; nothing is half-rewritten.

## Built-in fixers

### `int64_to_int32`

Casts INT64 initializers down to INT32. Refuses if any value is outside
the INT32 range.

> **Why:** TensorRT casts INT64 to INT32 at engine build time anyway.
> Doing it at fix time surfaces overflow as a clear refusal rather than a
> silent build-time failure.

### `float64_to_float32`

Casts FLOAT64 initializers down to FLOAT32. Refuses on `NaN`, `+/-inf`,
empty initializers, or any value exceeding FP32 range.

### `uint8_input`

Promotes a `UINT8` graph input to `FLOAT` when its only consumer is a
`Cast(to=FLOAT)`. Removes the redundant Cast and rewires downstream
nodes.

> Refuses for any other UINT8 consumption pattern -- the right rewrite
> depends on what the model expected the UINT8 to mean (raw bytes,
> normalized image, indices).

### `drop_dropout`

Removes Dropout nodes and rewires their data consumers. Refuses if the
Dropout's mask output is referenced anywhere.

> TensorRT folds Dropout out of the engine anyway; removing it up front
> keeps the diagnostic report and any visualisation tooling cleaner.

### `upsample_to_resize`

Rewrites leftover deprecated `Upsample` ops to `Resize` (mode `nearest`
or `linear`) on opset-13+ graphs -- the shape some exporters still emit
even though the op stopped being legal after opset 9. Refuses below
opset 13: the 4-input Resize form with empty `roi`/`sizes` placeholders
only validates from 13, so a conformant opset-9 model needs a
whole-model opset bump (`onnx.version_converter`) first, then a re-run.

## How to invoke

```bash
# preview what would change
trtcheck model.onnx --fix --dry-run --output model.fixed.onnx

# actually write the fixed file
trtcheck model.onnx --fix --output model.fixed.onnx
```

`--fix` requires `--output` (unless `--dry-run` is set) and refuses to
overwrite the input file. Use `--force` to overwrite an existing output.

## Adding a fixer

Each fixer lives in `trtcheck/fixers/`, implements the `Fixer` protocol
(`fix(model) -> list[FixApplied]`), and is registered in
`default_fixers()`. The TDD discipline is the same as the checkers:
write a failing test first, then implement the minimum that passes.
