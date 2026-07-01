# trtcheck validation scorecard

Measured accuracy of trtcheck's conversion verdicts against the
`bench/manifest.yaml` corpus. Produced by the `bench/` harness; raw
predictions in [`bench/outcomes.json`](bench/outcomes.json).

- **trtcheck version:** 1.0.0
- **Target:** TensorRT 10.3, `--severity critical` (the CI gate configuration)
- **Corpus:** 9 models — 3 public (ONNX Model Zoo), 6 bundled fixtures
- **Run date:** 2026-07-02
- **Hardware:** none. Static analysis only; total wall time for all 9 models: **2.0 s**.

## Results

|            | expected: fail | expected: convert |
|---|---|---|
| **trtcheck: fail**    | 3 (TP) | 0 (FP) |
| **trtcheck: convert** | 0 (FN) | 6 (TN) |

| Metric | Value |
|---|---|
| Precision | **1.000** |
| Recall    | **1.000** |
| F1        | **1.000** |

## Per-model outcomes

| Model | Source | Expected | trtcheck | Time |
|---|---|---|---|---|
| resnet50_v2 | ONNX Model Zoo | convert | convert | 0.32 s |
| mobilenetv2_1_0 | ONNX Model Zoo | convert | convert | 0.24 s |
| squeezenet1_1 | ONNX Model Zoo | convert | convert | 0.19 s |
| bundled_sequence_empty | fixture | fail | fail | 0.18 s |
| bundled_uint8_input | fixture | fail | fail | 0.22 s |
| bundled_int64_weights | fixture | convert | convert | 0.22 s |
| bundled_control_flow_loop | fixture | fail | fail | 0.22 s |
| bundled_fully_dynamic | fixture | convert | convert | 0.19 s |
| bundled_clean_minimal | fixture | convert | convert | 0.20 s |

## What this run caught

The first pass of this harness scored 8/9: `bundled_control_flow_loop`
was a false negative. The checker flagged a Loop trip count fed from a
graph input — a pattern TensorRT always rejects at engine build — as
WARNING, so the `--severity critical` gate waved it through. That
finding became the `loop_runtime_trip_count` critical check (a trip
count *computed* inside the graph, which TRT may still shape-infer,
stays a warning). The harness exists precisely to surface this class of
misclassification.

## Honest limitations

- **Small corpus.** Nine models. The three public models are
  well-behaved classifiers; the failure cases are synthetic fixtures
  built to exhibit specific TRT failure modes. Treat these numbers as
  "the checks do what they claim on known patterns," not as a
  field-accuracy estimate.
- **Ground truth is the manifest, not live `trtexec`.** The `expected`
  labels encode documented TRT behavior. The harness supports a second
  leg — running `trtexec` on GPU hardware and recording drift between
  the manifest and reality — which has not been run yet. See
  [`bench/README.md`](bench/README.md) for the GPU protocol.
- **One TRT target.** Scored against the 10.3 operator matrix only.

## Reproduce

```bash
python bench/fetch.py                             # downloads + SHA-256 verifies the public models
python bench/predict.py                           # writes bench/outcomes.json
python bench/score.py --outcomes bench/outcomes.json
```

## Expand the corpus

Add entries to `bench/manifest.yaml` (stable URL + `expected` outcome +
`sha256`), validate with `python tools/validate_bench_manifest.py`, and
open a PR. Models that convert *and* models that fail are both wanted —
the FP column is only meaningful if the corpus contains models trtcheck
should stay quiet about.
