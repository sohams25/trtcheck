"""Compare our bundled operator_matrix.json against upstream ONNX-TensorRT docs.

The upstream docs live at:
  https://raw.githubusercontent.com/onnx/onnx-tensorrt/main/docs/operators.md

Run with no arguments to fetch and diff; pass --local PATH for offline use
(e.g. when adapting this script in CI without GitHub access).

The output is intentionally simple text -- one line per drift item -- so it
can be diffed across runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

_DEFAULT_URL = "https://raw.githubusercontent.com/onnx/onnx-tensorrt/main/docs/operators.md"
_MATRIX_PATH = Path(__file__).parent.parent / "trtcheck" / "data" / "operator_matrix.json"

# Match a markdown table row that starts and ends with '|' and has at least
# one '|' in the middle. Cells are split on '|'.
_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
# Skip the header-separator row (cells contain only dashes and colons).
_SEPARATOR_RE = re.compile(r"^[\s\-:]+$")
# Extract the major TRT version a status column describes, e.g. the headers
# "TensorRT 10.x" / "TRT 10.x" / "TRT v10" -> "10". Used so a --target the table
# doesn't cover is not diffed against the wrong column (which would manufacture
# false mismatches), and to locate the status column by header rather than
# assuming it is always the second cell.
_COLUMN_VERSION_RE = re.compile(r"(?:tensor\s*rt|trt)\s*v?\s*(\d+)", re.IGNORECASE)


def major_of(version: str) -> str:
    """Major component of a TRT version string: '10.3' -> '10', '8' -> '8'."""
    return version.split(".")[0].strip()


def parse_upstream_markdown(text: str) -> dict[str, dict[str, str]]:
    """Extract operator -> {status, notes, version} from upstream markdown tables.

    Status normalizes 'Y' -> 'supported', 'N' -> 'not_supported', 'partial'
    or 'P' -> 'partial'. Unrecognized values pass through as 'unknown'.

    The status column is located by header (the first/only cell matching a
    "TensorRT N" / "TRT N" pattern), not by fixed position. ``version`` is that
    column's major TRT version (e.g. "10"), or "" if the header gives no version
    -- tagging each row lets ``compare`` refuse to diff it against a different
    target major. If the table ever carries multiple version columns a warning
    is emitted (the last is used); make this multi-column-aware if that happens.
    """
    out: dict[str, dict[str, str]] = {}
    column_version = ""
    status_idx = 1  # default: second cell, until the header tells us otherwise
    for line in text.splitlines():
        match = _ROW_RE.match(line)
        if not match:
            continue
        cells = [c.strip() for c in match.group(1).split("|")]
        if not cells or len(cells) < 2:
            continue
        op = cells[0]
        # The header row identifies itself ("Operator") and names the TRT
        # version column(s). Locate the status column BY HEADER rather than
        # assuming it is cells[1] -- the upstream table has had extra columns
        # before the status one. If several version columns exist we can't know
        # which the single-status model means, so warn loudly and take the last.
        if op.lower() == "operator":
            version_cols = [
                (i, m.group(1)) for i, c in enumerate(cells) if (m := _COLUMN_VERSION_RE.search(c))
            ]
            if len(version_cols) > 1:
                cols = ", ".join(f"col{i}=TRT{v}" for i, v in version_cols)
                print(
                    f"warning: upstream table has multiple TRT version columns ({cols}); "
                    "using the last. Make this tool multi-column-aware if that is wrong.",
                    file=sys.stderr,
                )
            if version_cols:
                status_idx, column_version = version_cols[-1]
            continue
        # Skip blank names and separators
        if not op or _SEPARATOR_RE.match(op):
            continue
        if status_idx >= len(cells):
            continue  # row narrower than the header promised; skip defensively
        status_cell = cells[status_idx].strip()
        if _SEPARATOR_RE.match(status_cell):
            continue
        status = _normalize_status(status_cell)
        notes = cells[status_idx + 1] if len(cells) > status_idx + 1 else ""
        out[op] = {"status": status, "notes": notes, "version": column_version}
    return out


def _normalize_status(cell: str) -> str:
    upper = cell.strip().upper()
    if upper in ("Y", "YES", "SUPPORTED"):
        return "supported"
    if upper in ("N", "NO", "NOT SUPPORTED", "NOT_SUPPORTED"):
        return "not_supported"
    if upper in ("P", "PARTIAL"):
        return "partial"
    return "unknown"


def compare(
    upstream: dict[str, dict[str, str]],
    matrix: dict[str, Any],
    target_version: str,
) -> list[str]:
    """Return a list of human-readable drift lines.

    Empty list means upstream and matrix agree on everything they both
    cover. Lines start with a category in square brackets so they can be
    grepped:
      [new upstream] -- op upstream supports that the matrix doesn't list
      [mismatch X]   -- the matrix's status for target_version disagrees
                        with upstream
    """
    drift: list[str] = []
    matrix_ops: dict[str, Any] = matrix["operators"]
    target_major = major_of(target_version)

    for op, info in upstream.items():
        # If the upstream table describes a different TRT major than the target,
        # its column says nothing about the target version -- diffing it would
        # invent mismatches. Skip rather than compare across majors.
        op_version = info.get("version", "")
        if op_version and op_version != target_major:
            continue
        upstream_status = info["status"]
        if op not in matrix_ops:
            drift.append(f"[new upstream] {op} (upstream={upstream_status})")
            continue
        matrix_status = matrix_ops[op]["support"].get(target_version, "unknown")
        if upstream_status == "unknown" or matrix_status == "unknown":
            continue  # nothing actionable
        if upstream_status != matrix_status:
            drift.append(
                f"[mismatch] {op}: upstream={upstream_status} "
                f"matrix[{target_version}]={matrix_status}"
            )
    return drift


def _load_upstream(local: Path | None) -> str:
    if local is not None:
        return local.read_text()
    with urllib.request.urlopen(_DEFAULT_URL, timeout=15) as resp:
        body: bytes = resp.read()
    return body.decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="Read upstream markdown from this local file instead of fetching.",
    )
    parser.add_argument(
        "--target",
        default="10.3",
        help="TRT version column in operator_matrix.json to compare against.",
    )
    args = parser.parse_args(argv)

    upstream_text = _load_upstream(args.local)
    upstream = parse_upstream_markdown(upstream_text)
    matrix = json.loads(_MATRIX_PATH.read_text())

    # Warn loudly if the upstream table describes a different TRT major than the
    # requested target -- otherwise an empty drift result could be mistaken for
    # "in sync" when really nothing was comparable.
    table_versions = {info.get("version", "") for info in upstream.values()} - {""}
    target_major = major_of(args.target)
    if table_versions and target_major not in table_versions:
        cov = ", ".join(sorted(table_versions))
        print(
            f"warning: upstream table covers TRT major {cov}; nothing to compare "
            f"for --target {args.target}. Use a target in that major, or update the "
            "upstream source."
        )
        return 0

    drift = compare(upstream, matrix, target_version=args.target)
    if not drift:
        print(f"no drift detected against {args.target} ({len(upstream)} upstream ops parsed)")
        return 0
    print(f"{len(drift)} drift item(s) found:")
    for line in drift:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
