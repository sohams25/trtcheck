<p align="center">
  <img src="assets/banner.svg" alt="trtcheck: static pre-flight checks for ONNX to TensorRT conversion" width="100%">
</p>

<p align="center">
  <a href="https://github.com/sohams25/trtcheck/actions/workflows/ci.yml"><img alt="ci"     src="https://github.com/sohams25/trtcheck/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trtcheck/">                                              <img alt="pypi"   src="https://img.shields.io/pypi/v/trtcheck.svg"></a>
  <a href="https://www.python.org/">                                                         <img alt="python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="LICENSE">                                                                         <img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

trtcheck is a static analysis tool for the ONNX → TensorRT conversion
step. It reads an ONNX file, runs five independent checkers against a
per-version TensorRT operator matrix, and reports whether the model
will convert — with a specific remediation for each blocker it finds.
Five built-in fixers rewrite the common failure patterns. Analysis
runs in seconds and needs no TensorRT install, CUDA, or GPU, so it
works on a laptop and in CI.

<p align="center">
  <img src="assets/demo.svg" alt="trtcheck demo: the UINT8 case before and after --fix" width="100%">
</p>

## Install

```bash
pip install trtcheck
```

Or from source for development:

```bash
git clone https://github.com/sohams25/trtcheck.git
cd trtcheck
pip install -e ".[dev]"
```

Python 3.10–3.13, `onnx >= 1.15, < 2.0`. No platform dependencies beyond
`onnx` itself — analysis needs no TensorRT, no GPU. Modeled TensorRT
targets: 8.0, 8.6, 10.0, 10.3; each operator entry carries its own
evidence level (official documentation, inferred, or unknown — see
[`docs/rules.md`](docs/rules.md) and the operator pages).

## Quick start

```bash
$ trtcheck model.onnx
```

```
CONVERSION BLOCKED — 1 critical, 0 warning
CRITICAL  input  Input  Input 'input' has dtype UINT8; TensorRT
                        accepts only FP32, FP16, INT32, or INT8
                        as graph inputs.
                        → Move the UINT8 → FLOAT32 conversion (and
                          normalization) into your preprocessing
                          pipeline rather than the model body.

Estimated fix time: 15–30 minutes.
```

Every report carries one of four verdicts — **blocked** (a known critical
incompatibility), **unverified** (no known blocker, but unresolved
conditions: unclassified or custom-domain operators, conditional support
that static analysis cannot settle), **likely** (all static checks passed
— a prediction, not a guarantee), and **verified** (an optional real
`trtexec` build succeeded via `--verify-runtime`). The exit code is `1`
on `blocked` and `0` otherwise (`--fail-on unverified` tightens the CI
gate), so the same command works unchanged as a CI gate.
[`docs/case-studies/uint8-input.md`](docs/case-studies/uint8-input.md)
walks this exact case end to end, including the `--fix` rewrite that
turns it into a passing graph. Verdict accuracy is measured:
[`SCORECARD.md`](SCORECARD.md) publishes precision and recall against
a corpus with known conversion outcomes.

## Motivation

`trtexec` reports conversion problems at engine build time, one at a
time, as C++ log output. Some common failures and what they trace back
to:

| `trtexec` error | Actually means | Root cause |
|---|---|---|
| `UNSUPPORTED_NODE: SequenceEmpty` | Your model contains an ONNX sequence op | PyTorch `List[Tensor] = []` in `forward()` |
| `Assertion failed: convert_dtype: UINT8` | Graph input dtype is `uint8` | Image preprocessing with `np.uint8` |
| `at least 5 dimensions are required` | `MaxPool` sees a tensor that lost rank after shape inference | Dynamic batch combined with `reshape` |
| `INT64 weights detected … not natively supported` | A `Constant` or `Initializer` is `int64` | `torch.LongTensor` for argmax / indices |
| `Network must have at least one output` | Shape inference removed every output | `If` / `Loop` with dynamic shape |

trtcheck runs the equivalent compatibility checks statically, so every
issue in the model surfaces in one pass, as a named finding with a
fix, before an engine build is attempted.

## What it checks

| Checker | Catches |
|---|---|
| **operator support** | Ops missing or partial in the target TRT version (e.g. `SequenceEmpty`, `GroupNormalization` on TRT 8.x); documented conditional-support rules (e.g. TopK `sorted=0`, cubic `Resize`); honest `unverified` findings for operators the matrix does not classify and for custom-domain ops that need a TRT plugin |
| **precision** | `UINT8` / `INT64` / `FLOAT64` / `STRING` / `BFLOAT16` graph inputs, `INT64` weights, and `FLOAT64` introduced by a `Cast` or `Constant` anywhere in the graph |
| **dynamic shapes** | Two or more symbolic input dims, including dynamic dims encoded as a concrete `-1` |
| **control flow** | `Loop` with runtime trip count, nested `Loop`, `If`, `Scan` |
| **graph structure** | Empty outputs, duplicate node names, oversized constants |

Every check descends into `If` / `Loop` / `Scan` **subgraph bodies** — an
unsupported op buried in a branch is caught, not waved through. Each finding
includes a specific remediation. Not "this is bad" — what to change, where.

## What it auto-fixes

`--fix` runs an audited, **transactional** pipeline: every fixer works on
an isolated candidate copy, the result must pass full ONNX validation
(strict type/shape inference) before it is kept, and the report shows
which findings were resolved, which remain, and whether any were
introduced. A fixer that crashes — including a third-party plugin — cannot
leave a half-rewritten model. Use `--dry-run` to preview.

| Fixer | Rewrites |
|---|---|
| **`uint8_input`** | Promotes a `UINT8` graph input to `FLOAT` and drops the redundant downstream `Cast` |
| **`int64_to_int32`** | Casts `INT64` initializers to `INT32` only when every use is at a schema position that accepts INT32 (e.g. `Gather` indices) — never `Reshape`/`Slice` shape inputs, which require INT64 |
| **`float64_to_float32`** | Casts `FLOAT64` initializers to `FLOAT32` when no value is NaN, infinite, or out of FP32 range |
| **`drop_dropout`** | Removes `Dropout` nodes that are provably in inference mode (`training_mode` absent or statically false; mask unused) |
| **`upsample_to_resize`** | Rewrites leftover deprecated `Upsample` nodes as `Resize` on opset-13+ graphs (nearest / linear) |

```bash
trtcheck model.onnx --fix --dry-run                    # preview
trtcheck model.onnx --fix --output fixed.onnx          # apply
```

Refuses to overwrite the input or an existing output unless you pass
`--force`.

## Measured accuracy

The [`bench/`](bench/) harness scores trtcheck's verdicts against a
corpus with known conversion outcomes. Latest run against the TRT 10.3
matrix:

| Corpus | Blocker precision | Blocker recall | Unverified coverage | Total wall time |
|---|---|---|---|---|
| 12 models: 3 from the ONNX Model Zoo, 9 bundled fixtures | 1.000 | 1.000 | 0.250 | 2.3 s |

`unverified` predictions are never counted as successes — they are
reported separately, split by ground truth. Twelve models is a small
corpus and the failure cases are synthetic, so read this as "the checks
do what they claim on known patterns", not as a field-accuracy estimate.
Ground truth is documented TRT behavior, not a live `trtexec` run.
[`SCORECARD.md`](SCORECARD.md) has the per-model table, the methodology,
and what each run caught (the first run's false negative became the
`loop_runtime_trip_count` critical check; this run exposed a `Clip`
coverage gap in the matrix). To grow the corpus, add a model with a
known outcome to [`bench/manifest.yaml`](bench/manifest.yaml) and open a
PR.

## How it compares

| | trtcheck | [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) | [Netron](https://github.com/lutzroeder/netron) |
|---|---|---|---|
| Needs TensorRT / GPU | no | yes, for conversion checks | no |
| Time to a verdict | seconds | minutes (builds a real engine) | manual inspection |
| Fix suggestions | per-finding remediation + `--fix` rewrites | no | no |
| CI integration | exit code, JSON, GitHub Action | scriptable, needs a GPU runner | no |
| Verdict strength | predicts the build outcome (and says so: four-state verdict with explicit uncertainty; optional `--verify-runtime` runs trtexec when available) | proves it | n/a |

Use them together. Polygraphy building an engine is the ground truth;
if you have the GPU and the minutes, run it. Netron is for eyeballing
a graph once you know which node to look at. trtcheck is the
ten-second gate that runs before either: on a laptop, in CI, on every
PR.

## Usage

```bash
# basic check (defaults to TensorRT 10.3)
trtcheck model.onnx

# target a specific TensorRT version
trtcheck model.onnx --target-trt 8.6

# machine-readable output for CI
trtcheck model.onnx --format json --output report.json

# self-contained HTML report
trtcheck model.onnx --format html --output report.html

# filter to blockers only
trtcheck model.onnx --severity critical

# compare two versions of a model (before / after a fix)
trtcheck before.onnx after.onnx --diff

# auto-fix simple issues (transactional; reports resolved/remaining findings)
trtcheck model.onnx --fix --output model_fixed.onnx

# strict CI gate: also fail on unresolved conditions
trtcheck model.onnx --fail-on unverified

# optional: verify with a real TensorRT build (needs trtexec)
trtcheck model.onnx --verify-runtime
```

Exit code is `1` on a `blocked` verdict, `0` otherwise; `--fail-on
unverified` also fails on unresolved conditions. Findings carry stable
rule ids (`TRT-OP-UNSUPPORTED`, `TRT-DTYPE-UINT8-INPUT`, ...) for CI
filtering — see [`docs/rules.md`](docs/rules.md) and
[`docs/usage.md`](docs/usage.md).

Full CLI reference: `trtcheck --help`.

## Use it as a GitHub Action

Ships a composite Action that runs on PRs touching `*.onnx` files and
posts a sticky comment summarizing the report. The dual-workflow
pattern (analyze on PR head with read-only token, comment from base
repo with write token) keeps fork PRs safe.

`.github/workflows/trtcheck.yml`:

```yaml
name: trtcheck
on:
  pull_request:
    paths: ["**/*.onnx"]
permissions:
  contents: read
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - id: trtcheck
        uses: sohams25/trtcheck@v1
        with:
          target-trt: "10.3"
          fail-on: "critical"
      - if: always()
        run: |
          mkdir -p comment-artifact
          cp "${{ steps.trtcheck.outputs.comment-md }}" comment-artifact/body.md
          echo "${{ github.event.pull_request.number }}" > comment-artifact/pr-number.txt
      - if: always()
        uses: actions/upload-artifact@v4
        with: { name: trtcheck-comment, path: comment-artifact/ }
```

Pair with `trtcheck-comment.yml` to post the comment from the base
repo. Full template at
[`.github/workflows/example-consumer/trtcheck-comment.yml`](.github/workflows/example-consumer/trtcheck-comment.yml).

### Action inputs

| Input | Default | Purpose |
|---|---|---|
| `version` | `1.0.0` | trtcheck PyPI version to install |
| `target-trt` | `10.3` | `--target-trt` value |
| `severity` | `warning` | `--severity` filter |
| `fail-on` | `critical` | Exit policy: `critical`, `warning`, or `never` |
| `paths` | `**/*.onnx` | Glob of files to consider |
| `changed-only` | `true` | Only analyze PR-changed files |
| `base-ref` | (PR base sha) | Base ref to diff against when `changed-only` is set |
| `source-path` | (unset) | Install trtcheck from a local path instead of PyPI; used by the selftest workflow |

### Action outputs

`report-json`, `comment-md`, `critical-count`, `warning-count`,
`status` (`pass` / `fail`).

## Plugins

Third-party packages can ship checkers, fixers, and reporters via
Python entry-points:

```toml
[project.entry-points."trtcheck.fixers"]
strip_identity = "your_package.fixers:StripIdentityFixer"
```

The Protocols live in `trtcheck.plugins`. Worked example at
[`examples/trtcheck-extra-fixers/`](examples/trtcheck-extra-fixers/).
Confirm a plugin loaded with `trtcheck --list-plugins`; filter one out
without uninstalling with `trtcheck --disable-plugin NAME`.

Full surface at
[`docs/design/plugin-sdk.md`](docs/design/plugin-sdk.md). The public
extension API was frozen at v1.0 and follows semver from here.

## The operator matrix

`trtcheck/data/operator_matrix.json` is a hand-curated mapping from
ONNX operators to their support status across TRT 8.0, 8.6, 10.0, and
10.3. Refresh recipe:

```bash
# edit the generator
$EDITOR tools/build_operator_matrix.py

# regenerate the JSON
python tools/build_operator_matrix.py

# validate
pytest tests/test_data_files.py -v

# detect drift against the upstream onnx-tensorrt operators table
python tools/check_matrix_drift.py
```

Run the drift check before each release to keep the matrix honest.

## Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

./scripts/run-tests.sh        # full pytest suite
mypy trtcheck/                # strict type check
black . && isort .            # format
```

TDD is mandatory for new checkers, fixers, and reporters. The full
contribution guide — TDD cycle, operator-matrix refresh recipe, plugin
authoring layout — lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Security disclosures: [`SECURITY.md`](SECURITY.md).

## Roadmap

- Grow the bench corpus past nine models with real-world failing
  models (detection heads, transformer blocks) and publish a refreshed
  [`SCORECARD.md`](SCORECARD.md) per release.
- Run the `trtexec` leg of the harness on GPU hardware and reconcile
  the manifest's expected outcomes against live TRT behavior.
- Track new TensorRT releases in the operator matrix. The weekly
  [matrix-drift Action](.github/workflows/matrix-drift.yml) already
  files a tracking issue when the upstream operator table moves.

Shipped: the validation scorecard and the scheduled matrix-drift
Action. See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## Citation

If trtcheck saves your project some GPU hours, a citation is welcome:

```bibtex
@misc{trtcheck,
  title  = {trtcheck: a static pre-flight checker for ONNX to TensorRT conversion},
  author = {Soham},
  year   = {2026},
  url    = {https://github.com/sohams25/trtcheck}
}
```

Using it in CI? Open a PR adding your project to this section.

## License

MIT. See [`LICENSE`](LICENSE).
