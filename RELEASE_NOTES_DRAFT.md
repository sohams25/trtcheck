# trtcheck v1.1.0 — draft release notes (not published)

Theme: **honest verdicts, safe fixes.**

## Highlights

- **Four-state verdicts.** Reports now conclude `blocked`, `unverified`,
  `likely`, or `verified` instead of a single boolean. Unknown operators
  and custom-domain ops no longer disappear silently — they make the
  verdict `unverified`, with per-op findings. `--fail-on unverified`
  turns that into a CI failure; `--plugin-domain` declares plugin-backed
  domains.
- **Stable rule ids (report schema 2.0).** Filter CI on
  `TRT-OP-UNSUPPORTED`, `TRT-DTYPE-UINT8-INPUT`, `TRT-CONTROL-LOOP-*`,
  ... Registry in `docs/rules.md`, guarded by a stability test. All 1.x
  JSON keys (including `conversion_likely`) are still emitted.
- **Safe `--fix`.** The fix pipeline is transactional (a crashing fixer —
  including third-party plugins — cannot leave a half-rewritten model),
  validates with strict type/shape inference, honors `--target-trt`, and
  reports findings resolved / remaining / introduced. Two correctness
  fixes ship with it:
  - `int64_to_int32` no longer corrupts `Reshape`/`Slice`-style
    schema-required-INT64 inputs; it converts only provably safe uses.
  - `drop_dropout` no longer removes Dropouts whose `training_mode` is
    true, dynamic, or unresolvable.
- **Conditional operator support.** The matrix can now express documented
  per-operator conditions (first: TopK `sorted=1` / `K < 3840`, Resize
  mode + coordinate-transform + antialias restrictions, sourced from the
  upstream onnx-tensorrt table).
- **Optional runtime verification.** `trtcheck model.onnx
  --verify-runtime` runs `trtexec` when available and upgrades the
  verdict to `verified` on a successful build; a recorded parser/build
  *failure* demotes an otherwise-clean report to `unverified`. Static
  analysis still needs no TensorRT, no GPU.

## Compatibility

- Public API: additive. `conversion_likely` is deprecated but present.
- JSON schema 2.0 is a superset of 1.x; consumers should migrate from
  `conversion_likely` to `verdict`.
- Exit codes unchanged by default. `--severity` no longer influences the
  exit code (it was display-only in intent; now in behavior too).
- Behavior change: models with unclassified/custom operators previously
  reported clean; they now report `unverified` findings (INFO severity,
  exit 0 unless `--fail-on unverified`).
- Behavior change: `--fix` refuses structurally invalid input models and
  converts fewer INT64 initializers than before (only provably safe
  uses). This is deliberate: the removed conversions were unsound.

## Release checklist (do not publish from this branch)

- [ ] bump `__version__` and `pyproject.toml` to 1.1.0
- [ ] move CHANGELOG Unreleased → 1.1.0 with date
- [ ] full suite + mypy strict + black/isort on a clean checkout
- [ ] regenerate operator docs + matrix (`tools/`) and verify drift check
- [ ] rerun `bench/` scorecard, refresh SCORECARD.md date/numbers
- [ ] run `scripts/package-smoke.sh` against the release wheel
- [ ] tag + build wheel/sdist, verify `pip install` in a fresh venv
