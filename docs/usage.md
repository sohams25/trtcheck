# Usage

## Basic check

```bash
trtcheck model.onnx
```

trtcheck is a **static** analyzer: it predicts, it does not guarantee. Every
report carries one of four verdicts:

| Verdict | Meaning |
|---|---|
| `blocked` | At least one known-critical incompatibility for the target TensorRT version. |
| `unverified` | No known blocker, but unresolved conditions remain: operators the support matrix does not classify, custom-domain ops that need a TensorRT plugin, or conditional support that cannot be settled statically (e.g. a runtime-dynamic TopK `K`). |
| `likely` | Every static check passed with nothing unresolved. Still a prediction — say "static analysis found no known blocker", never "guaranteed to convert". |
| `verified` | An optional real TensorRT build (`--verify-runtime`, requires `trtexec`) parsed the model and built an engine in *your* environment. |

## Exit codes

| Code | When |
|---|---|
| `0` | Verdict is `likely`, `verified`, or (by default) `unverified` |
| `1` | Verdict is `blocked`; with `--fail-on unverified`, also on `unverified`. Also fatal CLI errors. |
| `2` | Usage errors (bad flags/arguments) |

The exit code is computed from the **full** report: `--severity` trims the
displayed findings but never upgrades the verdict.

## Common flags

```bash
# target a specific TensorRT version
trtcheck model.onnx --target-trt 8.6

# machine-readable output for CI (schema 2.0: verdict, rule_id, confidence)
trtcheck model.onnx --format json --output report.json

# strict CI gate: also fail on unresolved/unverified conditions
trtcheck model.onnx --fail-on unverified

# declare that a TensorRT plugin implements a custom domain
trtcheck model.onnx --plugin-domain com.mycompany.ops

# self-contained HTML report
trtcheck model.onnx --format html --output report.html

# compare two ONNX files
trtcheck before.onnx after.onnx --diff --format html --output diff.html

# safe auto-fix (see docs/fixers.md): preview, then write
trtcheck model.onnx --fix --dry-run
trtcheck model.onnx --fix --output fixed.onnx

# optional runtime verification with a real TensorRT build
trtcheck model.onnx --verify-runtime --verify-timeout 900
```

## All flags

| Flag | Default | Meaning |
|---|---|---|
| `--target-trt` | `10.3` | TensorRT version to check against |
| `--format` | `console` | `console`, `json`, `html`, or a plugin reporter name |
| `--output` | (stdout) | Write the report to this path |
| `--severity` | `info` | Minimum severity to *display* (never changes the exit code) |
| `--fail-on` | `blocked` | `blocked` or `unverified`: which verdict fails the run |
| `--plugin-domain DOMAIN` | (none) | Declare a custom domain as plugin-backed; repeatable |
| `--diff` | off | Compare two ONNX files |
| `--force` | off | Allow `--output` to overwrite existing files |
| `--max-model-size` | `500` | Refuse to load ONNX files larger than this (MB) |
| `--fix` | off | Run the audited fix pipeline; writes to `--output` |
| `--dry-run` | off | With `--fix`, print changes without writing |
| `--verify-runtime` | off | Run `trtexec --onnx=MODEL` after static analysis |
| `--trtexec PATH` | (PATH lookup) | Explicit trtexec executable |
| `--verify-timeout` | `600` | Seconds before the trtexec run is killed |
| `--list-plugins` | off | Print discovered checkers, fixers, and reporters, then exit |
| `--disable-plugin NAME` | (none) | Exclude a checker, fixer, or reporter by name; repeatable |
| `-h`, `--help` | | Full CLI reference |

Set `TRTCHECK_DEBUG=1` to include tracebacks for third-party plugin
failures (hidden by default).

## Examples

### A model with a custom operator

```text
$ trtcheck detector.onnx
│ UNVERIFIED -- no known blocker, unresolved conditions remain  │
...
│ INFO │ TRT-OP-CUSTOM-DOMAIN │ <3 nodes> │ com.acme::DeformConv │ ...
$ echo $?        # 0 by default
$ trtcheck detector.onnx --fail-on unverified; echo $?   # 1
$ trtcheck detector.onnx --plugin-domain com.acme        # finding suppressed
```

### Safe fixing

```text
$ trtcheck model.onnx --fix --output fixed.onnx
  [int64_to_int32] cast initializer 'indices' from INT64 to INT32 ...
  [drop_dropout] removed Dropout node 'drop' ...

verdict: blocked -> likely (TensorRT 10.3); 2 finding(s) resolved, 0 remaining, 0 introduced

2 fix(es) applied. Wrote fixed.onnx.
```

Machine-readable summary: add `--format json` to the `--fix` invocation.

### Runtime verification (needs TensorRT + usually a GPU)

```text
$ trtcheck model.onnx --verify-runtime
runtime verification: success -- trtexec parsed the model and built an engine
│ VERIFIED -- TensorRT runtime build succeeded │
```

If `trtexec` is missing, times out, or cannot be spawned, the static
verdict is kept. If trtexec *ran and failed* (parser or engine-build
failure), an otherwise-`likely` report is demoted to `unverified` — runtime
evidence against the model is never hidden behind a clean static
prediction. Either way, the metadata (status, command, version, output
tails) is recorded in the JSON report under `runtime_verification`.

## JSON schema

Reports are schema `2.0`. Every 1.x key is still present (including the
deprecated boolean `conversion_likely`); new keys include `schema_version`,
`verdict`, `target_trt`, `runtime_verified`, `runtime_verification`, and
per-issue `rule_id` / `confidence` / `verify_required` / `target_trt` /
`graph_scope`. Filter CI on `rule_id` — the registry in
[docs/rules.md](rules.md) is covered by a stability test.

## CI integration

A composite GitHub Action ships in this repo. See
[the README](https://github.com/sohams25/trtcheck#use-it-as-a-github-action)
for the dual-workflow setup that posts a sticky PR comment when ONNX
files change.
