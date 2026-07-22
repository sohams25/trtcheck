# Release readiness report — trtcheck

Date: 2026-07-22.

## 1. Executive status: **COMPLETE**

All critical hardening claims were independently re-verified from the
repository state; four real gaps found during the audit were fixed with
regression tests; the package installs and works from a fresh venv outside
the source tree; security review is documented; no claim in the repo
depends on unavailable hardware evidence. Remote state untouched.

## 2. Branches and commits

- Starting point: `claude/trtcheck-hardening` @ `01b3f43`
  (5 commits on top of `main` @ `374fe48`) — clean tree, confirmed.
- Work branch: `claude/trtcheck-release-readiness`, created from that HEAD.
  Hardening commits were not rewritten, squashed, or rebased.

## 3. Environment

Linux 6.8.0-106-generic; repo venv Python **3.13.9** (Anaconda build; note:
mypy checks against `python_version = "3.10"` per pyproject, and the
package declares 3.10–3.13); onnx 1.21.0, click 8.4.1, numpy 2.4.6,
rich 15.0.0. **TensorRT: not installed. trtexec: not on PATH. GPU: not
used.** Network was available (used only for pip installs of dev tools).

## 4. Claims independently verified

| Handoff claim | Verification |
|---|---|
| 5 commits on main@374fe48, clean tree | `git log/status` — confirmed |
| 445 passed / 1 skipped; mypy strict, black, isort clean | Re-run at baseline — reproduced exactly |
| Reshape INT64 regression pinned; blind cast breaks full validation | `tests/test_fixers_int64_schema.py` proves the failure mode and the refusal; shared/mixed consumers, nested captures, shadowing, custom domains, graph-boundary tensors, overflow, empty and dead initializers all covered |
| Transactional fixers | Adversarial fixtures: mutate-then-raise, emit-invalid, undeclared mutation, malformed return — none can affect output; failures identify the fixer; later fixers run |
| Dropout training-mode safety | absent/false/true/dynamic/computed/ambiguous + is_test + mask + cross-scope cases all tested |
| Four-state verdicts + precedence | blocked > verified > unverified > likely tested, incl. combined conditions; static analysis cannot set `verified` (only the runtime path flips `runtime_verified`); missing trtexec is a controlled status |
| Stable rule ids / schema 2.0 | Registry pinned by test; every emitted finding carries an id; 1.x keys preserved; rendering deterministic (new test) |
| Unknown/custom ops cannot pass silently | Fixtures per category; `--plugin-domain` opt-out tested |
| Target-aware `--fix` diff, safe writes, exit codes | `tests/test_cli_fix.py`; invalid outputs never written; overwrite guards tested |
| Runtime verification isolation | list-args, timeout, output truncation, five failure states — mocked tests |
| Packaging data files, entry points | Verified in built artifacts + external smoke |

## 5. Defects discovered during this audit (and fixed)

1. **Runtime failure could hide behind static `likely`.** A recorded
   trtexec parser/build failure left the verdict `likely`. Now demoted to
   `unverified` (statuses that merely could not run — missing binary,
   timeout, spawn error — leave the static verdict). Unit + CLI tests.
2. **Lying fix records.** A fixer returning `FixApplied` records without
   modifying the model had its records committed. Now rejected with an
   explicit `FixFailure`; later fixers unaffected. Test added.
3. **Blank plugin rule ids.** Plugin checker findings without a `rule_id`
   reached reports empty. Now assigned a namespaced `PLUGIN-<NAME>`
   fallback. Test added.
4. **Diff identity ignored graph scope.** Same-named nodes in different
   subgraphs could alias in `--fix` before/after comparisons.
   `Issue.identity()` now includes `graph_scope`. Test added.

## 6. Test results (final)

- `./scripts/run-tests.sh` → **456 passed, 1 skipped** (opt-in benchmark), ~1.5 s
- `mypy trtcheck/ --strict` → clean (26 files)
- `black --check .` / `isort --check-only .` → clean
- No linter (ruff/flake8) is configured for this repo; none was added.

## 7. Packaging results

- Stale untracked `build/`, `dist/`, `*.egg-info` removed; rebuilt with
  `python -m build` → `trtcheck-1.0.0.tar.gz` + `trtcheck-1.0.0-py3-none-any.whl`
- `twine check dist/*` → PASSED for both artifacts
- sdist verified to include `LICENSE`, `README.md`, `trtcheck/data/*.json`
- **External smoke (`scripts/package-smoke.sh`, added this pass): PASS** —
  fresh venv in `/tmp`, wheel (non-editable) install, then from outside the
  repo: import + version, packaged data files, `trtcheck --help`,
  `python -m trtcheck --version`, analyze a generated model (console +
  schema-2.0 JSON), `--fix` producing a fully-valid model, and
  missing-trtexec behavior under an empty `PATH`.

## 8. Security / privacy findings

See `SECURITY_REVIEW.md`. Summary: no secrets, personal paths, employer
identifiers, or large/generated binaries in tracked files; `yaml.safe_load`
only; no pickle/eval/shell=True/archive extraction; subprocess use is
list-args with timeouts; downloads are https + SHA-256 verified; model-
derived text is sanitized for terminal and HTML; plugin trust boundary
documented. No security-critical fixes were required this pass.

## 9. Runtime evidence

**Real smoke passed (2026-07-22).** On branch
`claude/trtcheck-ngc-runtime-validation`, the 7-model corpus ran against
genuine TensorRT **10.3.0** (official NGC container
`nvcr.io/nvidia/tensorrt:24.08-py3`, RTX 4050 Laptop GPU, driver
580.126.20) using the installed wheel: 5 genuine engine builds, 2 genuine
parser failures, 0 wrapper/direct-trtexec disagreements. Full evidence:
`REAL_TENSORRT_VALIDATION_REPORT.md` +
`bench/real_tensorrt_smoke_results.json`. This validates the verification
integration and the selected cases, not universal model compatibility.

(An earlier same-day host-level attempt on
`claude/trtcheck-real-tensorrt-validation` was BLOCKED — no host TensorRT
existed; the container route above resolved it without any host changes.
Reproduce with `scripts/real-smoke-container.sh`.)

## 10. Benchmark evidence and limitations

12-model corpus (3 ONNX Model Zoo + 9 deterministic bundled fixtures);
blocker precision/recall 1.000 with unverified coverage 0.250; unverified
predictions are never counted as success. Ground truth is **documented
TRT behavior, not live builds** — stated in `SCORECARD.md` and the README.
Machine-readable summary: `bench/summary.json` (new `score.py --json`).
Fixtures are generated by `tests/fixtures/generate_broken.py`
(deterministic, byte-stable — verified by regeneration).

## 11. Public API / schema changes (this branch, on top of hardening)

- `AnalysisReport.verdict`: new demotion rule for recorded runtime
  failures (documented in docstring, docs/usage.md, design doc).
- `Issue.identity()` returns a 4-tuple (adds `graph_scope`) — internal
  diffing helper, not part of the JSON schema.
- `run_fixers` rejects claimed-but-absent changes (new failure mode
  string; `FixFailure` shape unchanged).
- `bench/score.py` gains `--json`; `ScoreResult.to_dict()`.
- JSON schema remains 2.0 — no key added or removed this pass.

## 12. Migration instructions

For 1.x JSON consumers: switch `conversion_likely` → `verdict`; filter
findings on `rule_id` (`docs/rules.md`). All 1.x keys still emitted.
CI gates that must treat unresolved conditions as failures: add
`--fail-on unverified`. No Python API removals.

## 13. Files changed (this branch: 16 files, +400/−13)

`trtcheck/types.py`, `trtcheck/analyzer.py`, `trtcheck/fixers/__init__.py`
(audit fixes); `bench/score.py`, `bench/summary.json`;
`scripts/package-smoke.sh`; tests (`test_verdicts`, `test_runtime_verify`,
`test_fixers_transactional`, `test_plugins_module`, `test_bench_score`);
docs (`usage.md`, design doc, README, CHANGELOG, RELEASE_NOTES_DRAFT);
new `SECURITY_REVIEW.md`, `PR_BODY_DRAFT.md`, this report.

## 14. Local commits (this branch)

1. `a335d6f` fix: close four audit gaps from independent hardening review
2. `2921dbe` chore: package smoke script (fresh-venv wheel install + CLI exercise)
3. (branch HEAD) docs: security review, PR draft, release readiness report, README/CHANGELOG updates — the commit containing this file; see `git log -1`

(Plus the five untouched hardening commits beneath.)

## 15. Version recommendation

**1.1.0.** Additive public API and JSON schema (2.0 is a superset of 1.x);
deliberate, documented behavior corrections (unclassified ops →
unverified; sounder `--fix` refusals; runtime-failure demotion). Not a
patch (user-visible behavior changes); not 2.0.0 (nothing removed, no key
repurposed). Bump `trtcheck/__init__.py.__version__` and
`pyproject.toml` together at release time — do not bump on this branch.

## 16. Remaining external actions

1. ~~Run the real trtexec smoke on a TensorRT machine~~ — **DONE**
   (2026-07-22, NGC TensorRT 10.3 container; see
   `REAL_TENSORRT_VALIDATION_REPORT.md`). No external checks remain.
2. Push branches, open the PR (draft body in `PR_BODY_DRAFT.md`), let CI
   run on all supported Pythons.
3. Optional: refresh `assets/demo.svg` from live CLI output (text updated;
   layout not re-rendered).

## 17. Commands for later (do not run automatically)

```bash
# inspect
git log --oneline main..claude/trtcheck-release-readiness
git diff main...claude/trtcheck-release-readiness

# push + PR
git push -u origin claude/trtcheck-hardening claude/trtcheck-release-readiness
gh pr create --base main --head claude/trtcheck-release-readiness \
  --title "Honest verdicts, safe fixes, release readiness" \
  --body-file PR_BODY_DRAFT.md

# release (after merge + version bump + CHANGELOG roll)
git tag -a v1.1.0 -m "trtcheck 1.1.0"
git push origin v1.1.0
python -m build && python -m twine check dist/* && python -m twine upload dist/*
```
