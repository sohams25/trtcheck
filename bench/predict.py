"""Run trtcheck against every manifest entry and write an outcomes file.

This is the trtcheck leg of the validation harness: for each model in
bench/manifest.yaml it invokes the CLI with ``--format json`` and records the verdict as
``convert``, ``unverified``, or ``fail``. The result feeds bench/score.py.

URL-sourced entries are read from bench/cache/ -- run bench/fetch.py
first. Bundled-fixture entries are read in place, so the pipeline works
offline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "bench" / "manifest.yaml"
_OUTCOMES = _REPO_ROOT / "bench" / "outcomes.json"


def resolve_model_path(entry: dict[str, Any], root: Path) -> Path:
    """Cache path for URL entries, repo-relative path for bundled ones."""
    source = entry["source"]
    if source.startswith(("http://", "https://")):
        return root / "bench" / "cache" / f"{entry['name']}.onnx"
    return root / source


def verdict_from_report(report: dict[str, Any]) -> str:
    """Map trtcheck's JSON report to the outcomes vocabulary.

    Schema 2.x reports carry a four-state ``verdict``; it maps to three
    outcome buckets: blocked -> "fail", unverified -> "unverified",
    likely/verified -> "convert". Schema 1.x reports (no ``verdict`` key)
    fall back to the boolean ``conversion_likely``.
    """
    verdict = report.get("verdict")
    if verdict is not None:
        if verdict == "blocked":
            return "fail"
        if verdict == "unverified":
            return "unverified"
        return "convert"
    return "convert" if report["conversion_likely"] else "fail"


def predict(
    entries: list[dict[str, Any]],
    root: Path,
    trtcheck_cmd: list[str] | None = None,
    out_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Run trtcheck on every entry; return (and optionally write) predictions."""
    # Default to the interpreter running this script, not a PATH lookup --
    # `python bench/predict.py` then works in any env with trtcheck installed.
    cmd = trtcheck_cmd or [sys.executable, "-m", "trtcheck"]
    predictions: dict[str, dict[str, str]] = {}
    for entry in entries:
        model = resolve_model_path(entry, root)
        if not model.exists():
            raise FileNotFoundError(
                f"{entry['name']}: {model} not found -- run bench/fetch.py first?"
            )
        proc = subprocess.run(
            [*cmd, str(model), "--format", "json"],
            capture_output=True,
            text=True,
        )
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{entry['name']}: trtcheck produced no JSON "
                f"(exit {proc.returncode}): {proc.stderr.strip()[:200]}"
            ) from exc
        predictions[entry["name"]] = {"trtcheck": verdict_from_report(report)}
    if out_path is not None:
        out_path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n")
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--out", type=Path, default=_OUTCOMES)
    args = parser.parse_args(argv)

    with open(args.manifest) as f:
        entries = yaml.safe_load(f)["models"]
    predictions = predict(entries, _REPO_ROOT, out_path=args.out)
    for name, row in predictions.items():
        print(f"{name:30s} {row['trtcheck']}")
    print(f"wrote {args.out} ({len(predictions)} predictions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
