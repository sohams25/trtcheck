# trtcheck validation scorecard

Measured accuracy of trtcheck's **static** verdicts against the
`bench/manifest.yaml` corpus. Produced by the `bench/` harness; raw
predictions in [`bench/outcomes.json`](bench/outcomes.json).

- **trtcheck version:** 1.0.0 + the then-unreleased verdict-model changes (shipped in v1.1.0)
- **Target:** TensorRT 10.3, full report (verdict-based; the old `--severity critical` gate is no longer used because verdicts need the INFO-level uncertainty findings)
- **Corpus:** 12 models — 3 public (ONNX Model Zoo), 9 bundled fixtures
- **Run date:** 2026-07-22
- **Hardware:** none. Static analysis only; total wall time for all 12 models: **2.3 s** (per-model times below include Python interpreter startup).

Since the verdict model landed, predictions are three-way: `fail`
(blocked), `convert` (likely/verified), and `unverified` (no known
blocker, unresolved conditions). **Unverified predictions are never
counted as successes** — they are reported as coverage, split by ground
truth.

## Results (blocker confusion matrix, unverified excluded)

|            | expected: fail | expected: convert |
|---|---|---|
| **trtcheck: fail**    | 4 (TP) | 0 (FP) |
| **trtcheck: convert** | 0 (FN) | 5 (TN) |

| Metric | Value |
|---|---|
| Blocker precision | **1.000** |
| Blocker recall    | **1.000** |
| Blocker F1        | **1.000** |
| Unverified coverage | **0.250** (3 of 12: 1 real failure, 2 that actually convert) |

## Per-model outcomes

| Model | Source | Expected | trtcheck | Time |
|---|---|---|---|---|
| resnet50_v2 | ONNX Model Zoo | convert | convert | 0.29 s |
| mobilenetv2_1_0 | ONNX Model Zoo | convert | convert | 0.21 s |
| squeezenet1_1 | ONNX Model Zoo | convert | convert | 0.16 s |
| bundled_sequence_empty | fixture | fail | fail | 0.17 s |
| bundled_uint8_input | fixture | fail | fail | 0.21 s |
| bundled_int64_weights | fixture | convert | convert | 0.20 s |
| bundled_control_flow_loop | fixture | fail | fail | 0.17 s |
| bundled_fully_dynamic | fixture | convert | **unverified** | 0.17 s |
| bundled_clean_minimal | fixture | convert | convert | 0.21 s |
| bundled_topk_unsorted | fixture | fail | fail | 0.15 s |
| bundled_custom_domain | fixture | fail | **unverified** | 0.15 s |
| bundled_reshape_int64_shape | fixture | convert | convert | 0.16 s |

## What this run caught

- The honesty change immediately exposed a **matrix coverage gap**:
  `mobilenetv2_1_0` came back `unverified` because `Clip` (35 nodes) was
  absent from the operator matrix. `Clip` is now classified for TRT 10.x
  from the upstream onnx-tensorrt table (8.x left `unknown` — no evidence
  in hand), and the model classifies cleanly again. That is exactly the
  loop the unverified verdict exists to drive.
- `bundled_topk_unsorted` (TopK `sorted=0`) is caught by the new
  conditional-support rules (`TRT-OP-CONDITION`), not by an operator-level
  blanket status.
- `bundled_custom_domain` is *expected: fail* (no TensorRT plugin exists
  for it) and trtcheck reports `unverified` — honest: static analysis
  cannot know whether a plugin is installed in the deployment environment.
  With `--fail-on unverified`, a CI gate still fails it.
- `bundled_fully_dynamic` is `unverified` because a usable engine needs an
  optimization profile trtcheck cannot see; the manifest labels it
  `convert` since a build with profiles succeeds.

## Honest limitations

- **Small corpus.** Twelve models; the failure cases are synthetic
  fixtures built to exhibit specific TRT failure modes. Treat these
  numbers as "the checks do what they claim on known patterns," not a
  field-accuracy estimate.
- **Ground truth is the manifest, not live `trtexec`.** The `expected`
  labels encode documented TRT behavior. No TensorRT/GPU run was
  performed for this scorecard; the harness supports recording real
  `trtexec` outcomes and reporting drift (see `bench/README.md`), and
  `trtcheck --verify-runtime` now does the same per-model.
- **One TRT target.** Scored against the 10.3 operator matrix only.

## Reproduce

```bash
python bench/fetch.py                             # downloads + SHA-256 verifies the public models
python bench/predict.py                           # writes bench/outcomes.json
python bench/score.py --outcomes bench/outcomes.json
```

Environment for the numbers above: Linux 6.8, Python 3.10 venv,
onnx 1.21.0, no GPU, no TensorRT installed.

## Expand the corpus

Add entries to `bench/manifest.yaml` (stable URL + `expected` outcome +
`sha256`), validate with `python tools/validate_bench_manifest.py`, and
open a PR. Models that convert *and* models that fail are both wanted —
the FP column is only meaningful if the corpus contains models trtcheck
should stay quiet about.
