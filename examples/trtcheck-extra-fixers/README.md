# trtcheck-extra-fixers

Example out-of-tree plugin for [`trtcheck`](https://pypi.org/project/trtcheck/).

This package adds one extra fixer (`strip_identity`) that removes
redundant `Identity` nodes from an ONNX graph. It is not published to
PyPI; the value here is showing the plugin layout so you can copy it for
your own fixers, checkers, or reporters.

## Layout

```
trtcheck-extra-fixers/
├── pyproject.toml                # declares the entry-point
├── src/
│   └── trtcheck_extras/
│       ├── __init__.py
│       └── fixers/
│           ├── __init__.py
│           └── strip_identity.py # the plugin
└── README.md
```

The entry-point declaration in `pyproject.toml`:

```toml
[project.entry-points."trtcheck.fixers"]
strip_identity = "trtcheck_extras.fixers.strip_identity:StripIdentityFixer"
```

- The group `trtcheck.fixers` matches the constant used by trtcheck's
  plugin loader. Use `trtcheck.checkers` or `trtcheck.reporters` for the
  other two plugin kinds.
- The entry-point name (`strip_identity`) should match the class's
  `name` attribute so `--disable-plugin strip_identity` works.

## Try it

```bash
# from this directory, after `pip install trtcheck`:
pip install -e .

# the plugin should now show up in the list:
trtcheck --list-plugins
# ...
# Fixers:
#   - int64_to_int32
#   - float64_to_float32
#   - drop_dropout
#   - upsample_to_resize
#   - uint8_input
#   - strip_identity     <-- here

# and `trtcheck --fix` will include it in the pipeline:
trtcheck model.onnx --fix --output model.fixed.onnx
```

## Writing your own

1. Pick a plugin kind: checker, fixer, or reporter.
2. Implement the corresponding Protocol from `trtcheck.plugins`. Each
   protocol requires a `name` attribute plus one method.
3. Declare an entry-point under the right group in `pyproject.toml`.
4. `pip install -e .` to make it discoverable.

`trtcheck --list-plugins` is the fastest way to confirm your plugin
loaded. If it doesn't show up, run with `--verbose` to see why -- the
loader logs at WARNING when a plugin fails Protocol check, fails to
import, or fails to construct.
