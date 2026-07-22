"""Click-based command line interface for trtcheck."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import onnx

from trtcheck import __version__
from trtcheck.analyzer import _BUILTIN_CHECKER_NAMES, Analyzer, AnalyzerConfig, safe_load
from trtcheck.fixers import (
    default_fixers,
    run_fixers,
    validate_model,
    validation_level_for,
)
from trtcheck.plugins import Reporter, load_plugins
from trtcheck.reporters.console import ConsoleReporter
from trtcheck.reporters.html import HTMLReporter
from trtcheck.reporters.json import JSONReporter
from trtcheck.types import REPORT_SCHEMA_VERSION, AnalysisReport, Severity, Verdict

_FORMATS = ["console", "json", "html"]
_SEVERITIES = ["critical", "warning", "info"]
_KNOWN_TARGETS = ["8.0", "8.6", "10.0", "10.3"]
_FAIL_ON = ["blocked", "unverified"]


# Exit codes (documented in docs/usage.md):
#   0  verdict is LIKELY or VERIFIED (or UNVERIFIED unless --fail-on unverified)
#   1  verdict is BLOCKED, or a fatal CLI error
#   2  usage error (Click)
def _exit_code(report: AnalysisReport, fail_on: str) -> int:
    if report.verdict is Verdict.BLOCKED:
        return 1
    if fail_on == "unverified" and report.verdict is Verdict.UNVERIFIED:
        return 1
    return 0


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
        target_trt=report.target_trt,
        runtime_verified=report.runtime_verified,
        runtime_verification=report.runtime_verification,
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
@click.option(
    "--plugin-domain",
    "plugin_domains",
    multiple=True,
    metavar="DOMAIN",
    help=(
        "Declare a custom ONNX domain as backed by an installed TensorRT "
        "plugin, suppressing its TRT-OP-CUSTOM-DOMAIN findings."
    ),
)
@click.option(
    "--fail-on",
    type=click.Choice(_FAIL_ON),
    default="blocked",
    show_default=True,
    help=(
        "Exit non-zero on this verdict or worse: 'blocked' fails only on "
        "known blockers; 'unverified' also fails when unresolved conditions "
        "remain."
    ),
)
@click.option(
    "--verify-runtime",
    is_flag=True,
    default=False,
    help=(
        "After static analysis, run 'trtexec --onnx=MODEL' to verify with a "
        "real TensorRT build. Requires trtexec (and usually a GPU)."
    ),
)
@click.option(
    "--trtexec",
    "trtexec_path",
    type=click.Path(path_type=str),
    default=None,
    help="Path to the trtexec executable (default: search PATH).",
)
@click.option(
    "--verify-timeout",
    type=int,
    default=600,
    show_default=True,
    metavar="SECONDS",
    help="Timeout for the trtexec run started by --verify-runtime.",
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
    plugin_domains: tuple[str, ...],
    fail_on: str,
    verify_runtime: bool,
    trtexec_path: str | None,
    verify_timeout: int,
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
        _run_fix(
            path,
            output,
            force,
            dry_run,
            max_model_size,
            disable_plugins,
            target_trt=target_trt,
            fmt=fmt,
            plugin_domains=plugin_domains,
            fail_on=fail_on,
        )
        return

    if not path.exists():
        raise click.ClickException(f"ONNX file not found: {path}")

    analyzer = Analyzer(
        AnalyzerConfig(
            target_trt=target_trt,
            max_model_size_mb=max_model_size,
            disable_plugins=list(disable_plugins),
            plugin_domains=list(plugin_domains),
        )
    )
    try:
        report = analyzer.analyze_path(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if verify_runtime:
        _attach_runtime_verification(report, path, trtexec_path, verify_timeout)

    display = _filter_issues(report, severity)

    text = _render(display, fmt, color=output is None, disable_plugins=disable_plugins)
    _emit(text, output, force=force)

    # Exit code comes from the UNfiltered report: --severity only trims the
    # display, it must not upgrade an unverified model to a passing one.
    sys.exit(_exit_code(report, fail_on))


def _attach_runtime_verification(
    report: AnalysisReport, path: Path, trtexec_path: str | None, timeout_s: int
) -> None:
    """Run trtexec against `path` and fold the outcome into `report`.

    VERIFIED is only ever set on a successful build with no static blocker;
    a runtime failure or an unavailable trtexec leaves the static verdict
    untouched (the metadata still records what happened).
    """
    from trtcheck.runtime_verify import verify_model

    result = verify_model(path, trtexec_path=trtexec_path, timeout_s=timeout_s)
    report.runtime_verification = result.to_dict()
    if result.verified and report.verdict is not Verdict.BLOCKED:
        report.runtime_verified = True
    click.echo(f"runtime verification: {result.status.value} -- {result.detail}", err=True)


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
    *,
    target_trt: str = "10.3",
    fmt: str = "console",
    plugin_domains: tuple[str, ...] = (),
    fail_on: str = "blocked",
) -> None:
    """The --fix pipeline: analyze -> fix transactionally -> validate ->
    re-analyze with the same target -> report resolved/remaining/new findings.
    """
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
    if output is not None:
        if output.resolve() == path.resolve():
            raise click.ClickException(
                "refusing to overwrite the input file; choose a different --output"
            )
        if not dry_run and output.exists() and not force:
            raise click.ClickException(
                f"refusing to overwrite existing file: {output} (use --force)"
            )

    try:
        model = safe_load(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # --fix only operates on structurally valid ONNX: fixers correct TensorRT
    # incompatibilities, not broken protobufs.
    try:
        validate_model(model, level="basic")
    except Exception as exc:
        raise click.ClickException(
            f"input model failed ONNX validation; --fix needs a valid model: {exc}"
        ) from exc

    analyzer = Analyzer(
        AnalyzerConfig(
            target_trt=target_trt,
            max_model_size_mb=max_model_size,
            disable_plugins=list(disable_plugins),
            plugin_domains=list(plugin_domains),
        )
    )
    before = analyzer.analyze_model(model, filename=str(path))

    # Built-ins first, then discovered plugin fixers, minus anything disabled
    # by name. run_fixers() is transactional: a fixer that crashes or emits an
    # invalid model has its changes discarded and later fixers still run.
    _, plugin_fixers, _ = load_plugins()
    disabled = set(disable_plugins)
    fixers = [
        f for f in [*default_fixers(), *plugin_fixers] if getattr(f, "name", "") not in disabled
    ]
    outcome = run_fixers(model, fixers)

    for failure in outcome.failures:
        click.echo(f"warning: fixer {failure.fixer!r}: {failure.reason}", err=True)

    after = analyzer.analyze_model(
        outcome.model, filename=str(output) if output else f"{path} (fixed)"
    )
    before_ids = {i.identity() for i in before.issues}
    after_ids = {i.identity() for i in after.issues}
    resolved = [i for i in before.issues if i.identity() not in after_ids]
    remaining = [i for i in after.issues if i.identity() in before_ids]
    introduced = [i for i in after.issues if i.identity() not in before_ids]

    if fmt == "json":
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "target_trt": target_trt,
            "input": str(path),
            "output": str(output) if output else None,
            "dry_run": dry_run,
            "validation": outcome.validation,
            "fixes_applied": [f.to_dict() for f in outcome.applied],
            "fixer_failures": [f.to_dict() for f in outcome.failures],
            "resolved": [i.to_dict() for i in resolved],
            "remaining": [i.to_dict() for i in remaining],
            "introduced": [i.to_dict() for i in introduced],
            "verdict_before": before.verdict.value,
            "verdict_after": after.verdict.value,
        }
        click.echo(json.dumps(payload, indent=2))
    else:
        if not outcome.applied and not outcome.failures:
            click.echo("no fixes applied -- model unchanged")
        for fix in outcome.applied:
            click.echo(f"  [{fix.fixer}] {fix.description}")
        click.echo(
            f"\nverdict: {before.verdict.value} -> {after.verdict.value} "
            f"(TensorRT {target_trt}); "
            f"{len(resolved)} finding(s) resolved, {len(remaining)} remaining, "
            f"{len(introduced)} introduced"
        )
        for issue in introduced:
            click.echo(f"  introduced: [{issue.rule_id}] {issue.message}")

    if not outcome.applied:
        # Nothing changed; never write an output file that is byte-identical
        # in content but pretends to be "fixed".
        if not dry_run:
            click.echo("nothing to write -- no fixer made a change", err=True)
        sys.exit(_exit_code(after, fail_on))

    if dry_run:
        if fmt != "json":
            click.echo(f"\n{len(outcome.applied)} fix(es) would be applied (dry run).")
        sys.exit(_exit_code(after, fail_on))

    assert output is not None  # guarded above
    # run_fixers validated every committed candidate; validate once more at
    # the write boundary as a belt-and-braces invariant.
    try:
        validate_model(outcome.model, level=outcome.validation)
    except Exception as exc:
        raise click.ClickException(f"applying fixes produced an invalid ONNX model: {exc}") from exc
    onnx.save(outcome.model, str(output))
    if fmt != "json":
        click.echo(f"\n{len(outcome.applied)} fix(es) applied. Wrote {output}.")
    sys.exit(_exit_code(after, fail_on))


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
