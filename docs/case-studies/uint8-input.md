# Case study: the UINT8 input trap

A common, silent failure mode for the PyTorch → ONNX → TensorRT pipeline.
Walks through what trtcheck flags, what `--fix` rewrites, and what you
save versus the `trtexec` retry loop.

## The setup

You're deploying an image model to a Jetson. Your preprocessing reads
pixels as `uint8` (the natural type for image bytes) and the cast to
float happens inside the network's forward pass. PyTorch exports this
faithfully. ONNX is happy. Everything looks fine on disk.

Here is the resulting graph in full (the bundled fixture is a miniature
of the same shape — a Relu standing in for the network body):

```
input  : tensor(uint8)   shape=[1, 3, 32, 32]
   └── Cast(to=FLOAT) ── Relu ──►  output
```

One input, a Cast, the network. Three bytes of disagreement with
TensorRT.

## What `trtexec` does

You run `trtexec --onnx=model.onnx`. Two to five minutes later, you get
something like:

```
[E] [TRT] ModelImporter.cpp: ERROR: builtin_op_importers.cpp:...
[E] [TRT] Assertion failed: convert_dtype: TensorRT does not support
    UINT8 inputs. Supported: FP32, FP16, INT32, INT8.
[E] Engine could not be created from network
```

The fix is obvious in hindsight — move the cast out of the model and
into preprocessing — but the failure mode is opaque the first time you
hit it, and every iteration of the `export → trtexec → read C++ trace →
google → patch` loop costs another two minutes plus context-switch.

## What trtcheck does

```bash
$ trtcheck tests/fixtures/failing/uint8_input.onnx
```

```
╭──────────────────────── trtcheck report ────────────────────────╮
│ CONVERSION BLOCKED -- known critical incompatibilities          │
│ file: tests/fixtures/failing/uint8_input.onnx  target: TRT 10.3 │
│ opset: 17  producer: trtcheck-fixtures  nodes: 2                │
│ 1 critical  0 warning  0 info                                   │
╰─────────────────────────────────────────────────────────────────╯
                          Detected issues
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Severity ┃ Rule                  ┃ Node  ┃ Operator ┃ Issue         ┃ Fix           ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ CRITICAL │ TRT-DTYPE-UINT8-INPUT │ input │ Input    │ Input 'input' │ Apply the     │
│          │                       │       │          │ has dtype     │ UINT8 ->      │
│          │                       │       │          │ UINT8.        │ FLOAT32       │
│          │                       │       │          │ TensorRT      │ conversion    │
│          │                       │       │          │ inputs must   │ (and          │
│          │                       │       │          │ be FLOAT32,   │ normalization)│
│          │                       │       │          │ FLOAT16,      │ in your       │
│          │                       │       │          │ INT32, or     │ preprocessing │
│          │                       │       │          │ INT8.         │ pipeline.     │
└──────────┴───────────────────────┴───────┴──────────┴───────────────┴───────────────┘
Estimated fix time: 15-30 minutes.
Address critical issues first; warnings can often wait.
```

`echo $?` → `1`. Exits non-zero so CI can fail the PR.

Total wall time: well under a second on this fixture. The bottleneck is
how fast you can read.

## What `--fix` rewrites

For this exact pattern — a leading `Cast` from a UINT8 input feeding
the rest of the network — the diagnosis comes with a built-in safe
rewrite:

```bash
$ trtcheck tests/fixtures/failing/uint8_input.onnx \
      --fix --output model_fixed.onnx
  [uint8_input] promote input 'input' from UINT8 to FLOAT and drop the
                redundant Cast node 'cast_1'

verdict: blocked -> likely (TensorRT 10.3); 1 finding(s) resolved,
0 remaining, 0 introduced

1 fix(es) applied. Wrote model_fixed.onnx.
```

What changed in the graph:

| | Before | After |
|---|---|---|
| input dtype | UINT8 | FLOAT |
| nodes | `Cast(uint8 → float)` → `Relu` | `Relu` |
| Relu reads | the Cast's output | the input directly |

Re-running trtcheck against the rewritten file:

```
╭──────────────────────── trtcheck report ────────────────────────╮
│ LIKELY -- static analysis found no known blocker                │
│ file: model_fixed.onnx  target: TensorRT 10.3                   │
│ opset: 17  producer: trtcheck-fixtures  nodes: 1                │
│ 0 critical  0 warning  0 info                                   │
╰─────────────────────────────────────────────────────────────────╯
No issues detected.
```

`echo $?` → `0`. Green light to `trtexec`.

`--dry-run` prints the same one-line description without writing
anything, in case you want to inspect the proposed rewrite before
committing it.

## What you actually pay attention to

The honest accounting:

| | trtexec-only loop | trtcheck loop |
|---|---|---|
| time to first signal | 2–5 min per attempt | < 1 second |
| signal quality | C++ traceback, no remediation | severity, node, exact fix |
| feedback for the next iteration | re-run, hope | report or apply `--fix` |
| dependencies on the laptop | TensorRT, CUDA, GPU | Python only |

The savings compound when the model has more than one issue. Many real
exports trip three or four checks at once (a UINT8 input, an INT64
embedding index, a fully-dynamic shape, a Loop with a dynamic trip
count). trtcheck surfaces all of them in the first run instead of
unmasking them one painful retry at a time.

## Reproducing this locally

```bash
git clone https://github.com/sohams25/trtcheck && cd trtcheck
pip install -e .

trtcheck tests/fixtures/failing/uint8_input.onnx
trtcheck tests/fixtures/failing/uint8_input.onnx --fix --output fixed.onnx
trtcheck fixed.onnx
```

No GPU. No TensorRT install. Three commands.
