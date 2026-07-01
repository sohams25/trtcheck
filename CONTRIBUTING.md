# Contributing to trtcheck

Two paths: improve the core, or ship a plugin. Pick the one that fits.

---

## Improving the core

Bug fixes, new built-in checkers, new built-in fixers, reporter
tweaks, operator-matrix updates — all land in this repo via PR.

### Setup

```bash
git clone https://github.com/sohams25/trtcheck.git
cd trtcheck
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Python 3.10+ is required. CI runs against 3.10 / 3.11 / 3.12 / 3.13.

### TDD is mandatory

Every checker, every fixer, every reporter, every utility ships with
its tests authored *first*. The cycle:

1. Write the test under `tests/test_<thing>.py` covering the happy
   path, a boundary, and the failure mode you're trying to catch.
2. Run it. Confirm it fails for the right reason.
3. Implement just enough to make it pass.
4. Refactor.
5. Commit as two changes: `test: …` then `feat: …` (or `fix: …`).

Test fixtures are deterministic. Build them with `onnx.helper` inside
the test file or extend `tests/fixtures/generate_broken.py` if a fix
needs a saved `.onnx`.

### The local gate

Before pushing, run what CI runs:

```bash
./scripts/run-tests.sh         # pytest with coverage
mypy trtcheck/                 # strict, no incremental
black --check . && isort --check-only .

# if you touched data files
python -c "import json; json.load(open('trtcheck/data/operator_matrix.json'))"
python -c "import json; json.load(open('trtcheck/data/remediation_db.json'))"
```

All four must be green. CI on the PR runs the same set on every commit.

### Updating the operator matrix

When NVIDIA drops a new TensorRT release:

```bash
$EDITOR tools/build_operator_matrix.py         # add the new column
python tools/build_operator_matrix.py          # regenerates the JSON
pytest tests/test_data_files.py -v             # validates
python tools/check_matrix_drift.py             # checks vs onnx-tensorrt
```

Commit the generator change and the regenerated JSON together. Drift
check should be clean (or, if it isn't, that's the next PR).

### Commit style

Lowercase subject, terse, imperative, no trailing period. Group
related changes into a single PR with one logical commit per change.
No AI co-author lines, no boilerplate footers. Look at the existing
`git log --oneline` for the voice.

---

## Shipping a plugin

You can publish your own checkers, fixers, or reporters as a separate
PyPI package — no fork required. The extension surface is frozen at
v1.0 and follows semver from there.

### Pick a kind

| Kind | Protocol | Entry-point group | Returns |
|---|---|---|---|
| Checker | `trtcheck.plugins.Checker` | `trtcheck.checkers` | `list[Issue]` |
| Fixer | `trtcheck.plugins.Fixer` | `trtcheck.fixers` | `list[FixApplied]` |
| Reporter | `trtcheck.plugins.Reporter` | `trtcheck.reporters` | `str` |

All three Protocols are `@runtime_checkable` and require a `name: str`
attribute. The loader instantiates with no arguments (zero-arg
constructors only, in v1.0).

### Minimal layout

Copy [`examples/trtcheck-extra-fixers/`](examples/trtcheck-extra-fixers/)
and adapt:

```
your-plugin/
├── pyproject.toml          # declares the entry-point
├── src/
│   └── your_pkg/
│       ├── __init__.py
│       └── fixers/         # or checkers/ or reporters/
│           ├── __init__.py
│           └── your_thing.py
└── README.md
```

### Declare the entry-point

In your `pyproject.toml`:

```toml
[project]
dependencies = [
    "trtcheck>=1.0",
    "onnx>=1.15,<2.0",          # if your plugin touches ONNX directly
]

[project.entry-points."trtcheck.fixers"]
my_fixer = "your_pkg.fixers.your_thing:YourFixer"
```

The entry-point name (`my_fixer`) should match the class's `name`
attribute. That way `trtcheck --disable-plugin my_fixer` works.

### Verify

```bash
pip install -e .
trtcheck --list-plugins
```

Your fixer should appear under `Fixers:` alongside the built-ins. If
it doesn't, the loader logged a WARNING — run trtcheck with
`PYTHONVERBOSE=1` or check Python logging configuration.

### Error contract

A plugin that raises during `check` / `fix` / `render` is caught and
surfaced as a `WARNING`-severity Issue. The rest of the pipeline keeps
running. One broken plugin should never kill an analysis.

Don't rely on this — handle your own errors. But know it's the safety
net.

### Versioning

If your plugin works against a specific `trtcheck` minor, pin it
(`trtcheck>=1.0,<2.0`). The public Protocols and `AnalyzerConfig`
fields are stable inside a major version; new ones are additive.

---

## Filing issues

Bugs: include the trtcheck version (`trtcheck --version`), the ONNX
file or a minimal reproducer, and the report's output.

Feature requests: describe the failure mode you want to catch, with a
concrete trtexec error or ONNX export pattern. Speculative checks are
hard to land without an example.

Security: `SECURITY.md` for the disclosure policy.

---

## License

By contributing, you agree your contribution is licensed under the
[MIT License](LICENSE).
