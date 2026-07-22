# Real TensorRT validation report

Date: 2026-07-22. Shipped in v1.1.0 via PR #23 (this run supersedes the
earlier same-day BLOCKED host-level attempt, commit `7527404`).

## Status: **COMPLETE — real runtime smoke passed**

Real runtime integration was smoke-tested on **TensorRT 10.3.0** using 7
representative generated/public fixtures inside the official NGC
container. This validates the verification integration and the selected
cases, **not universal model compatibility**.

## 1. Environment

| Item | Value |
|---|---|
| Host OS | Zorin OS 17.3 (Ubuntu 22.04 base), Linux 6.8.0-106-generic, x86_64 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) |
| NVIDIA driver | 580.126.20 |
| Docker | Engine 29.3.0 (native, default context) |
| NVIDIA Container Toolkit | **not installed** — see GPU-access strategy below |
| Container image | `nvcr.io/nvidia/tensorrt:24.08-py3` |
| Image digest | `sha256:9507e5f248fc61a5b2c985ce6e386ecf2576c3d96112ceb38cf88b240d4ca072` (14.6 GB local) |
| TensorRT (in container) | **10.3.0** (`trtexec` at `/opt/tensorrt/bin/trtexec`, banner `TensorRT v100300`) |
| Container Python | 3.10 |
| trtcheck under test | **installed wheel** `trtcheck-1.0.0-py3-none-any.whl` (never editable) |

### Why this image

The repository's support targets are TensorRT 8.0 / 8.6 / 10.0 / 10.3.
Newer NGC images carry TensorRT versions (10.16, 11.0) that are **not**
repository targets, and claiming them would violate the evidence policy.
`24.08-py3` is the official NGC release that ships **exactly TensorRT
10.3**, matching the default target the whole matrix is scored against.

### GPU-access strategy (no host changes)

`sudo` on this host requires an interactive password, so the NVIDIA
Container Toolkit could not be installed. Instead of modifying the host,
GPU access uses manual passthrough — strictly less invasive:

- `--device /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools`
- read-only mounts of the driver's user-space libraries only
  (`libcuda*`, `libcudadebugger*`, `libnvidia-ml*`, `libnvidia-cfg*`,
  `libnvidia-nvvm*`, `libnvidia-ptxjitcompiler*`, `libnvidia-gpucomp*`)
  staged into a temp dir, exposed via `LD_LIBRARY_PATH=/nvlibs`
- read-only mount of the host `nvidia-smi` binary (NGC images don't ship it)

Verified end-to-end before the corpus: `nvidia-smi` sees the RTX 4050
inside the container and a real engine build PASSED. **Zero host
packages/configuration were installed or changed** (nothing to roll back).
`scripts/real-smoke-container.sh` automates all of this and prefers
`--gpus all` automatically on hosts where the toolkit exists.

## 2. Corpus and results

Runner: `scripts/real-smoke-container.sh` → `scripts/real_tensorrt_smoke.py`.
Machine-readable results: [`bench/real_tensorrt_smoke_results.json`](bench/real_tensorrt_smoke_results.json).
Per model: static analysis (target 10.3), `trtcheck --verify-runtime
--trtexec <real>`, and an **independent direct trtexec run**, all with
timeouts and bounded output capture.

| Model | Static verdict | Wrapper runtime | Direct trtexec | Agree | Expected |
|---|---|---|---|---|---|
| clean_minimal | likely | success → **verified** | build PASSED (~6 s) | yes | yes |
| squeezenet1_1 (public zoo) | likely | success → **verified** | build PASSED | yes | yes |
| sequence_empty | **blocked** | parser_failure → stays blocked | parser FAILED | yes | yes |
| fully_dynamic | unverified | success → verified¹ | build PASSED¹ | yes | yes¹ |
| custom_domain | unverified | parser_failure → stays **unverified** | parser FAILED | yes | yes |
| uint8_input after `--fix` | likely (was blocked) | success → **verified** | build PASSED | yes | yes |
| reshape_int64_shape | likely | success → **verified** | build PASSED | yes | yes |

**Summary: 7/7 run, 5 genuine engine builds, 2 genuine parser failures,
0 wrapper/direct disagreements, 0 unexpected outcomes.**

¹ See dynamic-shape analysis below.

Key invariants confirmed with real execution:

- static analysis alone never yields `verified`; only the real trtexec
  success path set it;
- real parser failures were classified `parser_failure` (not lost, not
  misread from incidental log text) and never upgraded the verdict;
- the custom-domain model did **not** become verified without its plugin
  — trtexec's parser rejects it and trtcheck stays `unverified`;
- the `--fix` output of the UINT8 fixture — produced by the installed
  wheel inside the container — genuinely builds an engine;
- the Reshape INT64 regression model builds as-is, and the fixer's
  refusal to convert its shape initializer was re-confirmed in-container;
- trtexec is invoked as an argument list with a timeout; the recorded
  commands round-trip exactly.

## 3. Dynamic-shape / profile analysis

Observed, version-specific tool behavior on **TensorRT 10.3 trtexec**: a
model whose input is fully dynamic (`[batch, channels, h, w]`) does *not*
fail without shape flags — trtexec warns
`Dynamic dimensions required for input: input, but no shapes were
provided. Automatically overriding shape to: 1x1x1x1` and builds a
degenerate engine. With an explicit profile
(`--minShapes=input:1x1x8x8 --optShapes=input:1x3x64x64
--maxShapes=input:2x3x128x128`) the build also passes (5.2 s).

Consequences, honestly stated:

- a missing profile is **not** misclassified as an unsupported-operator
  failure — it isn't a failure at all on this trtexec version;
- trtcheck's static `TRT-SHAPE-PROFILE-MISSING` (unverified) finding
  remains the right warning: the no-profile "success" builds an engine
  fixed at 1×1×1×1, which is not a usable dynamic deployment;
- the wrapper reports `verified` for that no-profile success because a
  build genuinely succeeded in this environment; the auto-override
  warning is preserved in the recorded output tail. This nuance is
  listed under limitations.

This is recorded as TensorRT-10.3-specific evidence only; no
generalization to other versions and no matrix changes were made from it.

## 4. Defects discovered

- **trtcheck product defects: none.** All wrapper classifications matched
  independent trtexec behavior; JSON stayed schema-valid (2.0) under real
  runs; the installed wheel worked end-to-end in-container.
- **Smoke-runner defects (fixed in `scripts/real_tensorrt_smoke.py`
  before the final run):** (1) it parsed trtcheck JSON from a truncated
  output tail; (2) it mis-extracted the TensorRT version because
  `trtexec --version` exits non-zero on this build; (3) its initial
  no-profile expectation for dynamic models didn't match real TRT 10.3
  behavior. These are test-harness fixes, not product changes, and the
  runner is deterministic and repo-owned.

## 5. Installed-wheel validation

The corpus ran the **installed console script** (`pip install
/wheel/trtcheck-1.0.0-py3-none-any.whl` inside the container, repo
mounted read-only): packaged matrix/remediation data loaded, entry point
worked, explicit `--trtexec` configuration worked, JSON reports remained
schema 2.0 with rule ids throughout.

## 6. Cleanup

- No engines were saved (`--saveEngine` never passed); container work
  dirs live under `/tmp` outside the repo; staged driver libs are removed
  by the wrapper's trap.
- Temporary containers were `--rm` (none remain).
- `.gitignore` covers `*.engine`, `*.plan`, `*.trt`.
- The NGC image (14.6 GB) is retained in the local Docker cache — 150+ GB
  remain free. Optional removal: `docker rmi nvcr.io/nvidia/tensorrt:24.08-py3`.
- Unrelated images/containers untouched. No host packages or
  configuration were changed (nothing to back up or roll back).

## 7. Limitations

- Bounded smoke: 7 models, one TensorRT version (10.3.0), one GPU
  (RTX 4050 Laptop). This validates the verification integration and the
  selected cases — it is not a compatibility benchmark, and results are
  not generalized to TensorRT 8.x/10.0 or other hardware.
- `verified` means "trtexec parsed and built an engine in this
  environment"; for dynamic models without profiles, TRT 10.3 builds a
  degenerate 1×1×1×1 engine (see §3) — the static unverified findings
  remain the actionable signal for dynamic deployments.
- The public-model leg used the already-cached, SHA-256-verified
  squeezenet1_1 from the ONNX Model Zoo.

## 8. Reproduction

```bash
# one-time: docker pull nvcr.io/nvidia/tensorrt:24.08-py3
python -m build                       # produce dist/trtcheck-*.whl
scripts/real-smoke-container.sh       # runs the whole corpus, prints the summary
# results JSON is written to the printed temp dir; the committed copy is
# bench/real_tensorrt_smoke_results.json
```

On a host with the NVIDIA Container Toolkit the script uses `--gpus all`
automatically; otherwise it falls back to the no-root manual passthrough
described above.

## 9. Release recommendation

The one named external check from `RELEASE_READINESS_REPORT.md` — a real
trtexec smoke with at least one genuine success and one genuine failure —
has now passed, with full wrapper/direct agreement, using the installed
wheel. **Recommendation: release v1.1.0** (after the routine version-bump
and changelog-roll checklist in `RELEASE_NOTES_DRAFT.md`).
