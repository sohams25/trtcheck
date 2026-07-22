# Public release report — trtcheck v1.1.0

Date: 2026-07-22. Status: **COMPLETE**.

## Release record

| Item | Value |
|---|---|
| Repository | https://github.com/sohams25/trtcheck (public, owner verified) |
| PR | https://github.com/sohams25/trtcheck/pull/23 — merged (merge commit) |
| Merge commit | `8bfc3ee` |
| Tagged commit | `de2f27a` (merge commit + one doc-banner fix; identical package contents) |
| Tag | `v1.1.0` (annotated, pushed) |
| GitHub Release | https://github.com/sohams25/trtcheck/releases/tag/v1.1.0 |
| PyPI | **trtcheck 1.1.0 live** — wheel + sdist, published by the tag-triggered trusted-publishing workflow (run 29919281432, success, with digital attestations) |
| Public install | `pip install trtcheck==1.1.0` in a fresh venv outside the repo: import/version OK, `trtcheck --help` + `python -m trtcheck` OK, packaged data schema 2.0, JSON report schema 2.0 with rule ids on a generated ONNX model |

## CI and tests

- PR CI: 5/5 required-in-practice checks green (test matrix py3.10–3.13 +
  dogfood action). `main` has no branch protection; nothing was merged red.
- Local: 456 passed / 1 skipped; mypy `--strict` clean; Black + isort
  clean; `python -m build` + `twine check` PASSED for both 1.1.0
  artifacts; fresh-venv wheel smoke PASS.

## Release-only commits

1. `ab64714` chore(release): prepare v1.1.0 — version bumps
   (pyproject / `__init__` / action.yml default), CHANGELOG roll to
   `[1.1.0] - 2026-07-22`, release-notes finalization.
2. `de2f27a` docs: PR body draft reflects merged PR #23 (stale
   "no PR has been opened" banner corrected on main before tagging).

No hardening/validation commits were rewritten; no force-pushes; the
existing v1.1.0-free tag/release/PyPI state was confirmed before creating
any of them.

## Defects fixed during release

None — CI was green on the first run; the only correction was the stale
draft banner above.

## Evidence carried by this release (conservative claims)

- Real runtime evidence is for **TensorRT 10.3.0** (official NGC
  `tensorrt:24.08-py3` container): 7 bounded generated/public fixtures,
  5 genuine engine builds, 2 expected parser failures, 7/7
  wrapper/direct-trtexec agreement. This validates the verification
  integration and those cases, not universal compatibility.
- TensorRT 10.16 / 11.0 are not modeled targets and no support is claimed.
- On TRT 10.3, trtexec auto-builds a degenerate 1×1×1×1 engine for
  dynamic models without shape flags; static missing-profile findings
  remain the actionable signal.

## Remaining limitations / maintenance recommendation

- Scorecard ground truth is still documentation-based for the 12-model
  bench corpus; the real-runtime leg covers the 7-fixture smoke.
- Recommend a short **feature freeze**: watch the `matrix-drift` weekly
  action, PyPI install reports, and issue tracker for regressions before
  starting new feature work. Next natural work items: extend
  conditional-support coverage from upstream evidence, and re-run the
  container smoke when a new repo-supported TensorRT target is added.
