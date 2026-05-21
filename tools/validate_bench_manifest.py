"""Validate bench/manifest.yaml against the expected schema.

Run as a script for ad-hoc checks. Also importable: `validate(path)` raises
ManifestError with a short message on the first problem, otherwise returns
the parsed entry list.

Rules:
  - top-level: `models:` is a non-empty list
  - per-entry required keys: name, source, expected, reason
  - `expected` in {convert, fail}
  - `reason` in the trtcheck category vocabulary plus "none"
  - `source` is either an http(s) url or a repo-relative path that exists
    on disk (so the bundled-fixture entries can't drift)
  - `name` is unique across the manifest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

_VALID_EXPECTED = {"convert", "fail"}
_VALID_REASONS = {
    "operator_support",
    "precision",
    "dynamic_shapes",
    "control_flow",
    "graph_structure",
    "none",
}
_REQUIRED_KEYS = {"name", "source", "expected", "reason"}


class ManifestError(ValueError):
    """Raised when bench/manifest.yaml does not satisfy the schema."""


def _is_url(source: str) -> bool:
    return source.startswith("https://") or source.startswith("http://")


def validate(
    manifest_path: Path,
    *,
    repo_root: Path = _REPO_ROOT,
) -> list[dict[str, Any]]:
    """Parse and validate `manifest_path`. Returns the entry list."""
    try:
        with open(manifest_path) as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {manifest_path}") from exc
    if not isinstance(doc, dict):
        raise ManifestError(f"manifest root must be a mapping, got {type(doc).__name__}")
    entries = doc.get("models")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("'models' must be a non-empty list")

    seen_names: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"entry {i}: must be a mapping")
        missing = _REQUIRED_KEYS - entry.keys()
        if missing:
            raise ManifestError(f"entry {i}: missing keys {sorted(missing)}")
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ManifestError(f"entry {i}: 'name' must be a non-empty string")
        if name in seen_names:
            raise ManifestError(f"duplicate name: {name!r}")
        seen_names.add(name)

        if entry["expected"] not in _VALID_EXPECTED:
            raise ManifestError(
                f"entry '{name}': expected must be one of {sorted(_VALID_EXPECTED)}"
            )
        if entry["reason"] not in _VALID_REASONS:
            raise ManifestError(f"entry '{name}': reason must be one of {sorted(_VALID_REASONS)}")

        source = entry["source"]
        if not isinstance(source, str) or not source:
            raise ManifestError(f"entry '{name}': 'source' must be a non-empty string")
        if not _is_url(source):
            # bundled path: file must exist
            candidate = (repo_root / source).resolve()
            if not candidate.exists():
                raise ManifestError(f"entry '{name}': bundled source '{source}' does not exist")

    return entries


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    path = Path(argv[0]) if argv else _REPO_ROOT / "bench" / "manifest.yaml"
    try:
        entries = validate(path)
    except ManifestError as exc:
        print(f"manifest invalid: {exc}", file=sys.stderr)
        return 1
    print(f"manifest ok ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
