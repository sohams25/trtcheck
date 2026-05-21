# Plugin SDK design

Locks the public extension surface for v1.0. The goals:

1. Third parties can publish their own checkers, fixers, and reporters as
   separately installed Python packages. No need to fork `trtcheck`.
2. The existing in-tree checkers and fixers keep working untouched; they
   just become discoverable via the same mechanism.
3. The API is documented and semver-stable from v1.0 onward.

## Scope (v1.0 vs later)

In:

- Public `Checker`, `Fixer`, `Reporter` Protocols, frozen.
- Entry-point discovery for installed plugins.
- CLI flags to list and disable plugins.
- A worked example out-of-tree plugin.

Out (deferred to a later minor):

- Plugin config (per-plugin options on the CLI).
- A plugin registry / index.
- Per-target-trt plugin filtering (plugins decide their own applicability).

## Protocols

Live in `trtcheck.plugins`. The existing module-level Protocols
(`trtcheck.checkers.Checker`, `trtcheck.reporters.Reporter`, and the
implicit Fixer protocol in `trtcheck.fixers`) get re-exported from the new
home for backwards compatibility:

```python
# trtcheck/plugins.py
from __future__ import annotations
from typing import Protocol, runtime_checkable

import onnx

from trtcheck.types import AnalysisReport, Issue


@runtime_checkable
class Checker(Protocol):
    name: str
    def check(self, model: onnx.ModelProto) -> list[Issue]: ...


@runtime_checkable
class Fixer(Protocol):
    name: str
    def fix(self, model: onnx.ModelProto) -> list["FixApplied"]: ...


@runtime_checkable
class Reporter(Protocol):
    name: str
    def render(self, report: AnalysisReport) -> str: ...
```

`@runtime_checkable` lets the loader `isinstance`-validate a discovered
class before adding it to the pipeline. The cost is a small runtime check
at load time; the benefit is a clear error when a plugin doesn't satisfy
the contract.

`Issue`, `AnalysisReport`, `FixApplied`, `Severity`, `CheckCategory`
remain in `trtcheck.types` and are part of the public API.

The legacy import paths (`trtcheck.checkers.Checker` etc.) keep working
via a re-export so out-of-tree code written against v0.x does not break.

## Entry-point groups

Three groups, one per plugin type, namespaced under the package name:

| Group | Object kind |
|---|---|
| `trtcheck.checkers` | `Checker`-shaped class (zero-arg constructor) |
| `trtcheck.fixers` | `Fixer`-shaped class (zero-arg constructor) |
| `trtcheck.reporters` | `Reporter`-shaped class (zero-arg constructor) |

A plugin package declares them in its `pyproject.toml`:

```toml
[project.entry-points."trtcheck.fixers"]
strip_quant_dequant = "trtcheck_extras.fixers.strip_qd:StripQuantDequantFixer"
```

## Discovery semantics

`trtcheck.plugins.load_plugins()` walks
`importlib.metadata.entry_points(group=...)` once at analyzer
construction. It:

1. Loads each entry-point's referenced object.
2. Instantiates it with no arguments. (No kwargs in v1.0; deferred.)
3. Validates against the matching Protocol via `isinstance`. Misses are
   logged and skipped, not raised.
4. Returns three lists: `(checkers, fixers, reporters)`.

The `Analyzer` then prepends or appends discovered plugins to the
existing built-in pipeline. Built-ins run first so plugin behavior cannot
shadow a core check.

Discovery happens once per process. No hot-reload.

## Error handling

A plugin can fail at three points:

1. **Import** (its entry-point string doesn't resolve). Log at WARNING
   level with the entry-point name and the underlying exception class.
   Continue with the remaining plugins.
2. **Construction** (the class constructor raises). Same: log + skip.
3. **Execution** (the plugin's `check` / `fix` / `render` raises). The
   analyzer catches the exception, records it as an Issue with severity
   `WARNING`, category `OPERATOR_SUPPORT` (the closest pre-existing
   bucket), `node_name="<plugin: NAME>"`, and a remediation that names
   the failing plugin. The rest of the pipeline continues; one broken
   plugin should not kill the whole analysis.

Rationale: plugins are third-party code we don't control. Failing closed
(refusing to analyze anything if one plugin throws) is hostile to users
who installed an unrelated broken plugin from PyPI.

## CLI surface

Two new flags on the top-level command:

- `--list-plugins`: prints the resolved built-ins plus discovered
  entry-point plugins, grouped by type. Useful when debugging plugin
  installs.
- `--disable-plugin NAME` (repeatable): excludes a plugin by its `name`
  attribute. Applies to both built-ins and entry-point plugins so users
  can silence a noisy in-tree check without forking.

## Backwards compatibility

The v0.x callers we currently know about:

- The CLI -- internal, ours to update.
- `trtcheck.Analyzer` and `trtcheck.analyze()` -- public, must keep
  current signatures.
- The Protocols at their current import paths -- public, must keep
  importable.

The `AnalyzerConfig` dataclass grows two optional fields
(`disable_plugins: list[str]`, `discover_entry_point_plugins: bool =
True`) with defaults that preserve current behavior.

`recompute_counts()` / `derive_verdict()` were removed in v0.1.1 already;
no further removals at v1.0.

## Example plugin

`examples/trtcheck-extra-fixers/`: a fully separate Python package with
its own `pyproject.toml` declaring one fixer via entry-point. The package
imports `trtcheck.plugins` for the `Fixer` Protocol and `trtcheck.types`
for `FixApplied`. The example covers:

- Project layout
- `pyproject.toml` snippet
- A trivial fixer (e.g. removing unused `Identity` chains)
- `pip install -e .` then `trtcheck --list-plugins` showing the new entry

The example is not on PyPI; it's documentation.

## Migration guide

For v0.x callers who imported the Protocols from their old locations,
nothing changes -- the old imports still work and re-export from
`trtcheck.plugins`. The CHANGELOG entry for v1.0 calls out the new
preferred import path.

For anyone who patched the `Analyzer.checkers` list at runtime, the same
trick still works because the `Analyzer` still stores the resolved
checker list as `self.checkers`. We do not enforce immutability at v1.0;
that becomes an option at v1.1+.

## Open questions

1. Should entry-point plugins be allowed to register *additional*
   `CheckCategory` values? Today the enum is closed. Position for v1.0:
   keep the enum closed; plugins use the closest existing bucket and put
   plugin-specific detail in the `message` field. Revisit if a real
   plugin needs it.
2. Should `--list-plugins` ship as JSON-only or render a rich table?
   Position: emit a plain table by default, `--format json` if specified
   on the same command.

## Acceptance

The v1.0 release blocks on all of:

- Protocols live in `trtcheck.plugins` and are re-exported from current
  locations.
- `load_plugins()` exists, is covered by tests, and is called by
  `Analyzer.__init__`.
- `--list-plugins` and `--disable-plugin` work and are covered by CLI
  tests.
- The example plugin under `examples/trtcheck-extra-fixers/` is
  installable via `pip install -e .` and shows up in `--list-plugins`.
- The CHANGELOG calls out the public API surface and the migration
  story.
