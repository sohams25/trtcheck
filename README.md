<p align="center">
  <img src="assets/banner.svg" alt="trtcheck — catch ONNX to TensorRT failures before TensorRT does" width="100%">
</p>

<p align="center">
  <a href="https://github.com/sohams25/trtcheck/actions/workflows/ci.yml"><img alt="ci"     src="https://github.com/sohams25/trtcheck/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/trtcheck/">                                              <img alt="pypi"   src="https://img.shields.io/pypi/v/trtcheck.svg"></a>
  <a href="https://sohams25.github.io/trtcheck/">                                            <img alt="docs"   src="https://img.shields.io/badge/docs-mkdocs--material-blue"></a>
  <a href="https://www.python.org/">                                                         <img alt="python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue"></a>
  <a href="LICENSE">                                                                         <img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

> **`trtcheck`** — a static pre-flight checker for ONNX → TensorRT conversion.
> Five checkers, five auto-fixers, three report formats. Runs in seconds on
> any laptop. No TensorRT, no CUDA, no GPU.

---

## The 10-second pitch

The PyTorch → ONNX → TensorRT pipeline fails most of the time on the
last hop. The errors are cryptic, the iteration loop is slow, and the
fix is usually obvious in hindsight.

```bash
$ trtcheck model.onnx
```

```
CONVERSION WILL FAIL — 1 critical, 0 warning
CRITICAL  input  Input  Input 'input' has dtype UINT8; TensorRT
                        accepts only FP32, FP16, INT32, or INT8.
                        → Move the UINT8→FLOAT conversion into your
                          preprocessing pipeline rather than the model.

Estimated fix time: 15–30 minutes.
```

Exit code `1`. Wire it into CI to fail the PR before someone wastes an
afternoon on `trtexec`. See [`docs/case-studies/uint8-input.md`](docs/case-studies/uint8-input.md)
for the full before/after walkthrough — including the auto-fix that
rewrites this exact pattern into a passing graph.

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

Python 3.10+. No platform dependencies beyond `onnx` itself.

## Why the existing loop hurts

```
PyTorch ─► ONNX ─► trtexec ─► .engine
                       ▲
                       └── fails 80% of the time, on real workloads
```

A representative slice of what `trtexec` actually says, and what it
actually means:

| `trtexec` error | Actually means | Root cause |
|---|---|---|
| `UNSUPPORTED_NODE: SequenceEmpty` | Your model contains an ONNX sequence op | PyTorch `List[Tensor] = []` in `forward()` |
| `Assertion failed: convert_dtype: UINT8` | Graph input dtype is `uint8` | Image preprocessing with `np.uint8` |
| `at least 5 dimensions are required` | `MaxPool` sees a tensor that lost rank after shape inference | Dynamic batch combined with `reshape` |
| `INT64 weights detected … not natively supported` | A `Constant` or `Initializer` is `int64` | `torch.LongTensor` for argmax / indices |
| `Network must have at least one output` | Shape inference removed every output | `If` / `Loop` with dynamic shape |

The developer workflow today: run `trtexec`, wait 2–5 minutes, parse a
C++ traceback, guess, repeat. Average resolution time: **2–6 hours per
failure**.

The trtcheck workflow: run `trtcheck`, read a 10-second structured
report with a specific remediation per issue, fix once, then run
`trtexec` and have it pass.

## What it checks

| Checker | Catches |
|---|---|
| **operator support** | Ops missing or partial in the target TRT version (e.g. `SequenceEmpty`, `GroupNormalization` on TRT 8.x) |
| **precision** | `UINT8` / `FLOAT64` / `STRING` inputs, `INT64` weights, `BFLOAT16` on older targets |
| **dynamic shapes** | Multiple symbolic dims on graph inputs |
| **control flow** | `Loop` with runtime trip count, nested `Loop`, `If`, `Scan` |
| **graph structure** | Empty outputs, duplicate node names, oversized constants |

Each finding includes a specific remediation. Not "this is bad" — what
to change, where.

## What it auto-fixes

Pass `--fix` to apply built-in safe rewrites in place. Use `--dry-run`
to preview them first.

| Fixer | Rewrites |
|---|---|
| **`uint8_input`** | Promotes a `UINT8` graph input to `FLOAT` and drops the redundant downstream `Cast` |
| **`int64_to_int32`** | Casts `INT64` initializers to `INT32` when every value is in range |
| **`float64_to_float32`** | Casts `FLOAT64` initializers to `FLOAT32` when no value is NaN, infinite, or out of FP32 range |
| **`drop_dropout`** | Removes `Dropout` nodes and rewires consumers (skips nodes whose `mask` output is used) |
| **`upsample_to_resize`** | Rewrites the deprecated `Upsample` op as `Resize` when the mode and opset allow |

```bash
trtcheck model.onnx --fix --dry-run                    # preview
trtcheck model.onnx --fix --output fixed.onnx          # apply
```

Refuses to overwrite the input or an existing output unless you pass
`--force`.

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

# auto-fix simple issues
trtcheck model.onnx --fix --output model_fixed.onnx
```

Exit code is `1` if conversion is unlikely to succeed, `0` otherwise.

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
        uses: sohams25/trtcheck@v1.0.0
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
mkdocs serve                  # local docs site
```

TDD is mandatory for new checkers, fixers, and reporters. The full
contribution guide — TDD cycle, operator-matrix refresh recipe, plugin
authoring layout — lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Security disclosures: [`SECURITY.md`](SECURITY.md).

## Roadmap

- Run the `bench/` validation harness end-to-end against the public
  ONNX corpus and publish a `SCORECARD.md` with measured TPR / FPR.
- Scheduled GitHub Action that runs `tools/check_matrix_drift.py` and
  files a tracking issue when the upstream operator table drifts.

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.

## License

MIT. See [`LICENSE`](LICENSE).
