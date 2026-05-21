"""Generate one markdown page per operator from operator_matrix.json.

Idempotent: running twice produces byte-identical output, and any
previously-generated page whose operator left the matrix is removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MATRIX = _REPO_ROOT / "trtcheck" / "data" / "operator_matrix.json"
_DEFAULT_OUT = _REPO_ROOT / "docs" / "operators"

_INDEX_PAGE_NAME = "index.md"


def render_operator(name: str, entry: dict[str, Any], *, versions: list[str]) -> str:
    """Render a single operator's markdown page."""
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append("## TensorRT support")
    lines.append("")
    lines.append("| Version | Status |")
    lines.append("| --- | --- |")
    support = entry.get("support", {})
    for v in versions:
        lines.append(f"| {v} | {support.get(v, 'unknown')} |")
    lines.append("")

    notes = entry.get("notes")
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
        lines.append("")

    limitations = entry.get("limitations", [])
    if limitations:
        lines.append("## Limitations")
        lines.append("")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    remediation = entry.get("remediation")
    if remediation:
        lines.append("## Remediation")
        lines.append("")
        lines.append(remediation)
        lines.append("")

    github_issue = entry.get("github_issue")
    if github_issue:
        lines.append("## See also")
        lines.append("")
        lines.append(f"- {github_issue}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_index(matrix: dict[str, Any]) -> str:
    """Render the operators/index.md landing page."""
    ops = sorted(matrix.get("operators", {}).keys())
    versions = matrix.get("target_trt_versions", [])
    lines: list[str] = []
    lines.append("# Operators")
    lines.append("")
    if versions:
        lines.append(f"Per-operator TensorRT support against versions {', '.join(versions)}.")
        lines.append("")
    lines.append(f"`{len(ops)}` operators tracked.")
    lines.append("")
    for op in ops:
        lines.append(f"- [{op}]({op}.md)")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build(matrix_path: Path, out_dir: Path) -> int:
    """Generate the operator pages. Returns the number of operator pages."""
    matrix = json.loads(Path(matrix_path).read_text())
    operators: dict[str, dict[str, Any]] = matrix.get("operators", {})
    versions: list[str] = matrix.get("target_trt_versions", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-operator pages
    expected_files = {f"{name}.md" for name in operators}
    expected_files.add(_INDEX_PAGE_NAME)
    for name, entry in operators.items():
        page = out_dir / f"{name}.md"
        page.write_text(render_operator(name, entry, versions=versions))

    # Write the landing page
    (out_dir / _INDEX_PAGE_NAME).write_text(render_index(matrix))

    # Remove stale generated files (anything *.md not in expected_files)
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix == ".md" and path.name not in expected_files:
            path.unlink()

    return len(operators)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=_DEFAULT_MATRIX,
        help="Path to operator_matrix.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Directory to write per-operator pages into.",
    )
    args = parser.parse_args(argv)
    try:
        count = build(args.matrix, args.out)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"build_operator_docs: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} operator pages plus index to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
