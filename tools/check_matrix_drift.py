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


def parse_upstream_markdown(text: str) -> dict[str, dict[str, str]]:
    """Extract operator -> {status, notes} from upstream markdown tables.

    Status normalizes 'Y' -> 'supported', 'N' -> 'not_supported', 'partial'
    or 'P' -> 'partial'. Unrecognized values pass through as 'unknown'.
    """
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = _ROW_RE.match(line)
        if not match:
            continue
        cells = [c.strip() for c in match.group(1).split("|")]
        if not cells or len(cells) < 2:
            continue
        op = cells[0]
        # Skip headers and separators
        if not op or op.lower() == "operator" or _SEPARATOR_RE.match(op):
            continue
        status_cell = cells[1].strip()
        if _SEPARATOR_RE.match(status_cell):
            continue
        status = _normalize_status(status_cell)
        notes = cells[2] if len(cells) > 2 else ""
        out[op] = {"status": status, "notes": notes}
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

    for op, info in upstream.items():
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
