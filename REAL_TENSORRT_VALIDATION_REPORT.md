# Real TensorRT validation report

Date: 2026-07-22. Branch: `claude/trtcheck-real-tensorrt-validation`
(from `claude/trtcheck-release-readiness` @ `063f071`).

## Status: **BLOCKED — no TensorRT on this host**

The runtime-verification leg could not run: this machine has an NVIDIA
GPU and driver but **no TensorRT installation of any kind**. Nothing in
this report is runtime evidence; no runtime result has been fabricated.
Everything that *can* be validated without TensorRT was validated and is
recorded below.

## 1. Environment

| Item | Value |
|---|---|
| OS | Linux 6.8.0-106-generic (x86_64) |
| Python | 3.13.9 (repo venv) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB (5697 MiB free) |
| NVIDIA driver | 580.126.20 (`nvidia-smi` works) |
| CUDA toolkit | not installed (`nvcc` absent; no `/usr/local/cuda*`) |
| TensorRT | **not installed** |
| trtexec | **not found** |
| onnx | 1.21.0 |
| Git state | clean tree at branch creation |

## 2. Blocker evidence (exact checks)

Every discovery avenue was exhausted:

```text
$ which trtexec                     -> not on PATH
$ ls /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt*/bin/trtexec \
     /opt/tensorrt*/bin/trtexec /opt/nvidia/tensorrt*/bin/trtexec
                                    -> no such files
$ timeout 60 find / -xdev -name trtexec   -> no match before timeout
$ ldconfig -p | grep -i nvinfer     -> empty (no TensorRT libraries)
$ apt list --installed | grep -iE 'tensorrt|nvinfer'  -> empty
$ python3 -c "import tensorrt"      -> ModuleNotFoundError
$ ls /opt/nvidia                    -> sdkmanager only (Jetson flashing tool)
$ docker images                     -> 39 images; none contain TensorRT
                                       (ROS/Supabase/build tooling; the only
                                       ML images are third-party-owned aarch64
                                       Jetson images — out of scope by policy
                                       and the wrong architecture)
```

The repository's override mechanism (`trtcheck --trtexec PATH`) was
inspected and works, but there is no executable to point it at.

Not attempted, deliberately: `pip install tensorrt` (multi-GB CUDA wheel
stack that still ships **no trtexec binary** — the CLI path under test),
pulling an NGC TensorRT container (~8 GB), or apt-installing TensorRT +
CUDA system-wide. All are the heavyweight automatic installations this
task prohibits.

## 3. Corpus and what WAS validated

Bounded 7-model corpus (all repository-owned/deterministic, plus one
public ONNX Model Zoo model already cached with SHA-256 verification).
Machine-readable results: [`bench/real_tensorrt_smoke_results.json`](bench/real_tensorrt_smoke_results.json).

| Model | Expected real outcome | Static verdict (TRT 10.3 target) | `--verify-runtime` on this host |
|---|---|---|---|
| clean_minimal | build success | likely | `missing_trtexec` (controlled) |
| squeezenet1_1 (public zoo) | build success | likely | `missing_trtexec` |
| sequence_empty | parser failure | **blocked** | `missing_trtexec` |
| fully_dynamic | needs optimization profile | **unverified** | `missing_trtexec` |
| custom_domain | parser failure w/o plugin | **unverified** | `missing_trtexec` |
| uint8_input after `--fix` | build success | likely (was blocked) | `missing_trtexec` |
| reshape_int64_shape | build success | likely | `missing_trtexec` |

Invariants verified live on this host (the subset that does not require
TensorRT):

- static analysis alone **never** produced `verified` (asserted for all 7);
- a missing verifier is a **controlled** `missing_trtexec` status — no
  uncaught exception, exit codes unchanged;
- `--fix` on `uint8_input.onnx` produced a fully-valid model
  (`blocked -> likely`) through the installed pipeline;
- the Int64 fixer **refused** the Reshape shape-input conversion on the
  regression fixture (`--fix --dry-run` applies no `int64_to_int32` fix);
- trtexec discovery, argument construction, timeout, and the
  parser/build/timeout/missing classification remain covered by the
  deterministic mocked suite (`tests/test_runtime_verify.py`, 9 tests),
  including the invariant that a real parser/build failure demotes a
  `likely` verdict to `unverified`.

## 4. Static-to-runtime comparison

**Not possible on this host.** No direct trtexec runs, no agreement
analysis, no elapsed build times. The table above records expectations,
not outcomes.

## 5. Defects discovered

None attributable to trtcheck on this leg. (Classification A–D not
exercised for runtime behavior; the environment result is category C:
environmental, documented, no static rules changed to compensate.)

## 6. Fixes made / tests added

None required by this leg. No code changed on this branch; the only new
artifacts are this report and the results JSON.

## 7. Remaining limitations

- Runtime verification remains validated **only** through mocked
  subprocess tests plus the live missing-verifier path. No real TensorRT
  parser/build has ever been executed against this codebase.
- The scorecard's ground truth remains documented TRT behavior, not live
  builds (already stated in `SCORECARD.md` and the README).

## 8. Exact reproduction commands (for a machine WITH TensorRT)

```bash
# 0) confirm the verifier
trtexec --version

# 1) per-model: static, then real verification (expect VERIFIED on clean models)
trtcheck tests/fixtures/clean_minimal.onnx --target-trt 10.3 --format json
trtcheck tests/fixtures/clean_minimal.onnx --target-trt 10.3 --verify-runtime --verify-timeout 900

# 2) expected failure (SequenceEmpty): verdict stays blocked; trtexec parser fails
trtcheck tests/fixtures/failing/sequence_empty.onnx --verify-runtime
trtexec --onnx=tests/fixtures/failing/sequence_empty.onnx   # independent check

# 3) dynamic shapes: trtexec needs explicit profiles
trtexec --onnx=tests/fixtures/failing/fully_dynamic.onnx \
  --minShapes=input:1x3x32x32 --optShapes=input:4x3x224x224 --maxShapes=input:8x3x512x512

# 4) custom domain without plugin: must NOT become verified
trtcheck tests/fixtures/custom_domain.onnx --verify-runtime

# 5) fix-then-verify
trtcheck tests/fixtures/failing/uint8_input.onnx --fix --output /tmp/fixed.onnx
trtcheck /tmp/fixed.onnx --verify-runtime

# 6) Reshape INT64 regression model builds as-is
trtcheck tests/fixtures/reshape_int64_shape.onnx --verify-runtime

# 7) refresh the machine-readable evidence honestly
#    (replace direct_trtexec.status entries with real outcomes)
```

Record the TensorRT version printed by trtexec in the results file; do
not generalize outcomes to other TensorRT versions.

## 9. Does the evidence justify a v1.1.0 release?

The **static, packaging, safety, and mocked-integration** evidence is
complete and strong. The **real-runtime smoke is still outstanding** and
is the one named external check in `RELEASE_READINESS_REPORT.md`.
Recommendation unchanged: **release v1.1.0 after the real trtexec smoke
passes on a TensorRT machine** (expected: one genuine `verified`, one
correctly-classified failure). Releasing without it would ship an
integration path that has never touched the real tool it wraps.
