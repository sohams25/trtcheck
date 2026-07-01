"""Click-based command line interface for trtcheck."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import onnx

from trtcheck import __version__
from trtcheck.analyzer import _BUILTIN_CHECKER_NAMES, Analyzer, AnalyzerConfig, safe_load
from trtcheck.fixers import FixApplied, apply_all, default_fixers
from trtcheck.plugins import Fixer, Reporter, load_plugins
from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter
from trtcheck.types import AnalysisReport, Severity

_FORMATS = ["console", "json", "html"]
_SEVERITIES = ["critical", "warning", "info"]
_KNOWN_TARGETS = ["8.0", "8.6", "10.0", "10.3"]


class _SafePluginFixer:
    """Isolate a third-party fixer the way the analyzer isolates plugin
    checkers: a crash is reported on stderr and skipped, never a traceback.
    A partial mutation from a crashed fixer is caught downstream by the
    onnx.checker validation before anything is written."""

    def __init__(self, inner: Fixer) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)

    def fix(self, model: onnx.ModelProto) -> list[FixApplied]:
        try:
            return self._inner.fix(model)
        except Exception as exc:
            click.echo(
                f"warning: plugin fixer {self.name!r} raised "
                f"{exc.__class__.__name__}: {exc}; skipped "
                "(--disable-plugin to silence)",
                err=True,
            )
            return []


def _plugin_reporters(disable_plugins: tuple[str, ...] = ()) -> dict[str, Reporter]:
    _, _, reporters = load_plugins()
    disabled = set(disable_plugins)
    return {
        name: r
        for r in reporters
        if (name := getattr(r, "name", r.__class__.__name__)) not in disabled
    }


def _warn_unknown_disables(disable_plugins: tuple[str, ...]) -> None:
    """A typo in --disable-plugin must not silently leave the target enabled."""
    if not disable_plugins:
        return
    checkers, fixers, reporters = load_plugins()
    known = {getattr(x, "name", "") for group in (checkers, fixers, reporters) for x in group}
    known |= _BUILTIN_CHECKER_NAMES
    known |= {f.name for f in default_fixers()}
    known |= set(_FORMATS)
    for name in disable_plugins:
        if name not in known:
            click.echo(
                f"warning: --disable-plugin {name!r} matches no known checker, "
                "fixer, or reporter (see --list-plugins)",
                err=True,
            )


def _filter_issues(report: AnalysisReport, minimum: str) -> AnalysisReport:
    """Return a copy of `report` keeping only issues at or above the threshold.

    Threshold semantics:
      --severity critical -> only critical
      --severity warning  -> critical + warning
      --severity info     -> everything (default, no filter)
    """
    if minimum == "info":
        return report
    threshold = Severity(minimum)
    cutoff = Severity.rank(threshold)
    kept = [i for i in report.issues if Severity.rank(i.severity) <= cutoff]
    # Don't mutate the original -- make a shallow copy of the report.
    filtered = AnalysisReport(
        filename=report.filename,
        onnx_ir_version=report.onnx_ir_version,
        opset_version=report.opset_version,
        producer=report.producer,
        total_nodes=report.total_nodes,
        issues=kept,
    )
    return filtered


def _render(
    report: AnalysisReport,
    fmt: str,
    *,
    color: bool = True,
    disable_plugins: tuple[str, ...] = (),
) -> str:
    if fmt == "json":
        return JSONReporter().render(report)
    if fmt == "html":
        return HTMLReporter().render(report)
    if fmt == "console":
        # color=False when writing to a file so the artifact is plain text,
        # not a soup of raw ANSI escape codes.
        return ConsoleReporter(color=color).render(report)
    plugin = _plugin_reporters(disable_plugins).get(fmt)
    if plugin is None:
        available = ", ".join([*_FORMATS, *_plugin_reporters(disable_plugins)])
        raise click.BadParameter(f"unknown format {fmt!r}; available: {available}")
    return plugin.render(report)


def _emit(text: str, output_path: Path | None, force: bool = False) -> None:
    if output_path is None:
        click.echo(text, nl=False)
        if not text.endswith("\n"):
            click.echo()
        return
    if output_path.exists() and not force:
        raise click.ClickException(
            f"refusing to overwrite existing file: {output_path} (use --force)"
        )
    try:
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"could not write to {output_path}: {exc}") from exc


@click.command(name="trtcheck", context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="trtcheck")
@click.argument(
    "models",
    nargs=-1,
    required=False,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
)
@click.option(
    "--target-trt",
    type=click.Choice(_KNOWN_TARGETS),
    default="10.3",
    show_default=True,
    help="TensorRT version to check compatibility against.",
)
@click.option(
    "--format",
    "fmt",
    metavar="[console|json|html|PLUGIN]",
    default="console",
    show_default=True,
    help="Output format: a built-in or a discovered plugin reporter name.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the report to this path instead of stdout.",
)
@click.option(
    "--severity",
    type=click.Choice(_SEVERITIES),
    default="info",
    show_default=True,
    help="Minimum severity to include in the report.",
)
@click.option(
    "--diff",
    is_flag=True,
    default=False,
    help="Compare two ONNX files. Requires exactly two model arguments.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --output even if it already exists.",
)
@click.option(
    "--max-model-size",
    type=int,
    default=500,
    show_default=True,
    metavar="MB",
    help="Refuse to load ONNX files larger than this (in MB).",
)
@click.option(
    "--fix",
    "fix_mode",
    is_flag=True,
    default=False,
    help="Apply built-in fixers and write a corrected ONNX file to --output.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="With --fix, print what would change without writing anything.",
)
@click.option(
    "--list-plugins",
    is_flag=True,
    default=False,
    help="Print discovered checkers, fixers, and reporters and exit.",
)
@click.option(
    "--disable-plugin",
    "disable_plugins",
    multiple=True,
    metavar="NAME",
    help="Exclude a plugin by its name. May be passed multiple times.",
)
def main(
    models: tuple[Path, ...],
    target_trt: str,
    fmt: str,
    output: Path | None,
    severity: str,
    diff: bool,
    force: bool,
    max_model_size: int,
    fix_mode: bool,
    dry_run: bool,
    list_plugins: bool,
    disable_plugins: tuple[str, ...],
) -> None:
    """Run trtcheck against one or two ONNX models.

    \b
    Examples:
      trtcheck model.onnx
      trtcheck model.onnx --target-trt 8.6
      trtcheck model.onnx --format json --output report.json
      trtcheck before.onnx after.onnx --diff
    """
    _warn_unknown_disables(disable_plugins)

    if list_plugins:
        _print_plugin_listing(target_trt, max_model_size, list(disable_plugins))
        return

    if diff:
        if len(models) != 2:
            raise click.BadParameter("--diff requires exactly two model arguments")
        _run_diff(models, target_trt, fmt, output, severity, force, max_model_size, disable_plugins)
        return

    if len(models) != 1:
        raise click.BadParameter("Pass one ONNX file, or use --diff with exactly two files.")

    path = models[0]

    if fix_mode:
        _run_fix(path, output, force, dry_run, max_model_size, disable_plugins)
        return

    if not path.exists():
        raise click.ClickException(f"ONNX file not found: {path}")

    analyzer = Analyzer(
        AnalyzerConfig(
            target_trt=target_trt,
            max_model_size_mb=max_model_size,
            disable_plugins=list(disable_plugins),
        )
    )
    try:
        report = analyzer.analyze_path(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    report = _filter_issues(report, severity)

    text = _render(report, fmt, color=output is None, disable_plugins=disable_plugins)
    _emit(text, output, force=force)

    if not report.conversion_likely:
        sys.exit(1)


def _run_diff(
    models: tuple[Path, ...],
    target_trt: str,
    fmt: str,
    output: Path | None,
    severity: str,
    force: bool,
    max_model_size: int,
    disable_plugins: tuple[str, ...] = (),
) -> None:
    before, after = models
    if not before.exists():
        raise click.ClickException(f"ONNX file not found: {before}")
    if not after.exists():
        raise click.ClickException(f"ONNX file not found: {after}")

    analyzer = Analyzer(
        AnalyzerConfig(
            target_trt=target_trt,
            max_model_size_mb=max_model_size,
            disable_plugins=list(disable_plugins),
        )
    )
    try:
        report_before = _filter_issues(analyzer.analyze_path(before), severity)
        report_after = _filter_issues(analyzer.analyze_path(after), severity)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if fmt == "json":
        payload = json.dumps(
            {"before": report_before.to_dict(), "after": report_after.to_dict()},
            indent=2,
        )
        _emit(payload, output, force=force)
    else:
        if fmt == "html":
            combined = HTMLReporter().render_diff(report_before, report_after)
            _emit(combined, output, force=force)
        else:
            # console and plugin reporters: render each side, join with a rule.
            before_text = _render(
                report_before, fmt, color=output is None, disable_plugins=disable_plugins
            )
            after_text = _render(
                report_after, fmt, color=output is None, disable_plugins=disable_plugins
            )
            sep = "\n" + ("=" * 80) + "\n"
            _emit(
                f"BEFORE: {before}\n{before_text}\n{sep}AFTER: {after}\n{after_text}",
                output,
                force=force,
            )

    # The point of --diff is "did my fix work?". Exit based on the *after*
    # model only -- a passing 'after' must be a passing CI signal even when
    # the 'before' fixture still fails.
    if not report_after.conversion_likely:
        sys.exit(1)


def _run_fix(
    path: Path,
    output: Path | None,
    force: bool,
    dry_run: bool,
    max_model_size: int,
    disable_plugins: tuple[str, ...] = (),
) -> None:
    if not path.exists():
        raise click.ClickException(f"ONNX file not found: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_model_size:
        raise click.ClickException(
            f"ONNX file is {size_mb:.1f} MB, above the {max_model_size} MB limit. "
            "Raise the limit with --max-model-size."
        )

    # Validate destination before doing work / printing a fix list, so a user
    # who forgets --output is told immediately rather than after the report.
    if not dry_run and output is None:
        raise click.ClickException("--fix requires --output to write the fixed model")

    try:
        model = safe_load(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # Built-ins first, then discovered plugin fixers (isolated so a broken
    # plugin can't take down the run), minus anything disabled by name.
    _, plugin_fixers, _ = load_plugins()
    disabled = set(disable_plugins)
    fixers = [
        f
        for f in [*default_fixers(), *(_SafePluginFixer(pf) for pf in plugin_fixers)]
        if getattr(f, "name", "") not in disabled
    ]
    new_model, applied = apply_all(model, fixers)

    if not applied:
        click.echo("no fixes applied -- model unchanged")
        return

    for fix in applied:
        click.echo(f"  [{fix.fixer}] {fix.description}")

    if dry_run:
        click.echo(f"\n{len(applied)} fix(es) would be applied (dry run).")
        return

    if output is None:  # unreachable: guarded above, but keep mypy + -O happy
        raise click.ClickException("--fix requires --output to write the fixed model")
    if output.resolve() == path.resolve():
        raise click.ClickException(
            "refusing to overwrite the input file; choose a different --output"
        )
    if output.exists() and not force:
        raise click.ClickException(f"refusing to overwrite existing file: {output} (use --force)")
    try:
        onnx.checker.check_model(new_model)
    except Exception as exc:  # onnx ValidationError et al.
        raise click.ClickException(f"applying fixes produced an invalid ONNX model: {exc}") from exc
    onnx.save(new_model, str(output))
    click.echo(f"\n{len(applied)} fix(es) applied. Wrote {output}.")


def _print_plugin_listing(target_trt: str, max_model_size: int, disable_plugins: list[str]) -> None:
    """Print built-in plus discovered plugins, grouped by type."""
    analyzer = Analyzer(
        AnalyzerConfig(
            target_trt=target_trt,
            max_model_size_mb=max_model_size,
            disable_plugins=disable_plugins,
        )
    )
    click.echo("Checkers:")
    for c in analyzer.checkers:
        click.echo(f"  - {getattr(c, 'name', c.__class__.__name__)}")
    click.echo("\nFixers:")
    built_in_fixers = list(default_fixers())
    _, discovered_fixers, discovered_reporters = load_plugins()
    disabled = set(disable_plugins)
    for f in built_in_fixers + discovered_fixers:
        name = getattr(f, "name", f.__class__.__name__)
        if name not in disabled:
            click.echo(f"  - {name}")
    click.echo("\nReporters:")
    built_in_reporters = ["console", "json", "html"]
    for name in built_in_reporters:
        click.echo(f"  - {name}")
    for r in discovered_reporters:
        rname = getattr(r, "name", r.__class__.__name__)
        if rname not in disabled:
            click.echo(f"  - {rname}")


if __name__ == "__main__":
    main()
