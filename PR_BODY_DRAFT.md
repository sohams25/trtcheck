# PR draft: honest verdicts, safe fixes, release readiness

> Draft only — no PR has been opened. Branches:
> `claude/trtcheck-hardening` (5 commits) + `claude/trtcheck-release-readiness`
> (verification & hardening of the hardening), targeting `main`.

## What this PR does

1. **Four-state verdicts** (`blocked` / `unverified` / `likely` /
   `verified`). Unknown and custom-domain operators no longer pass
   silently; a failed real trtexec run can no longer hide behind a clean
   static prediction. `conversion_likely` stays as a deprecated alias.
2. **Stable diagnostics, JSON schema 2.0** — per-finding `rule_id`,
   `confidence`, `verify_required`, `target_trt`, `graph_scope`; the
   registry (docs/rules.md) is pinned by a stability test; every 1.x JSON
   key is preserved.
3. **Safe, transactional `--fix`** — per-fixer isolated candidates
   validated with strict type/shape inference; use-aware INT64
   conversion (the Reshape shape-input corruption is a pinned regression);
   training-mode-aware Dropout removal; before/after findings diff with
   the same `--target-trt`.
4. **Conditional operator support** — evidence-backed per-op conditions
   (TopK, Resize) sourced from the upstream onnx-tensorrt table.
5. **Optional runtime verification** — `--verify-runtime` (trtexec,
   list-args, timeout); only a real successful build yields `verified`.
6. **Evidence discipline** — bench harness with three-way outcomes and
   unverified coverage (never counted as success), machine-readable
   summary, refreshed honest SCORECARD; package smoke test from a fresh
   venv; SECURITY_REVIEW.md.

## Compatibility

- Public API additive; JSON 2.0 is a superset of 1.x.
- Exit codes unchanged by default; `--severity` is now display-only.
- Deliberate behavior changes (documented in CHANGELOG): unclassified /
  custom ops report `unverified`; `--fix` refuses invalid inputs and
  declines previously-unsound INT64 conversions.

## Test plan

- [x] 456 tests, 1 opt-in skip (`./scripts/run-tests.sh`)
- [x] `mypy trtcheck/ --strict` clean; black + isort clean
- [x] `python -m build` + `twine check` pass
- [x] `scripts/package-smoke.sh` — wheel install in fresh venv, CLI
      analyze / JSON / fix / missing-trtexec paths
- [x] Real `trtexec` smoke — TensorRT 10.3.0 (NGC `24.08-py3` container,
      RTX 4050): 7-model corpus, 5 genuine builds, 2 genuine failures,
      0 wrapper/direct disagreements, installed wheel used throughout.
      Evidence: `REAL_TENSORRT_VALIDATION_REPORT.md`
