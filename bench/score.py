"""Score trtcheck predictions against an outcomes file.

Inputs:
  - `bench/manifest.yaml` (each entry's `expected` is the ground truth)
  - an outcomes JSON file. Shape:
        {
          "predictions": {
            "<model_name>": {
              "trtcheck": "convert" | "fail",     # required
              "trtexec":  "convert" | "fail"      # optional
            },
            ...
          }
        }

The scoring treats "fail" as the positive class. A "fail" prediction by
trtcheck against a manifest entry whose expected is also "fail" is a true
positive (trtcheck caught the real failure).

The pure-function entry point `score()` returns a `ScoreResult` so tests
can pin specific values. The CLI prints a confusion matrix plus
precision/recall/F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

_VALID_OUTCOMES = {"convert", "fail"}
# trtcheck predictions may additionally be "unverified": no known blocker but
# unresolved conditions remain. Unverified predictions are excluded from the
# blocker confusion matrix and reported as coverage instead -- an unverified
# call is neither a caught failure nor a clean bill of health.
_VALID_PREDICTIONS = _VALID_OUTCOMES | {"unverified"}


@dataclass
class ScoreResult:
    """Confusion matrix + summary metrics with "fail" as the positive class."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    skipped: list[str] = field(default_factory=list)
    drift: list[str] = field(default_factory=list)
    # Entries trtcheck declined to classify (prediction == "unverified"),
    # split by what the ground truth says they actually do.
    unverified_on_fail: list[str] = field(default_factory=list)
    unverified_on_convert: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def unverified_total(self) -> int:
        return len(self.unverified_on_fail) + len(self.unverified_on_convert)

    @property
    def unverified_coverage(self) -> float:
        """Fraction of all classified-or-unverified entries trtcheck declined
        to classify. High coverage with low blocker recall means the tool is
        honest but not yet informative; low coverage with high recall is the
        goal."""
        denom = self.total + self.unverified_total
        return self.unverified_total / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def score(
    manifest_entries: list[dict[str, Any]],
    outcomes: dict[str, dict[str, str]],
) -> ScoreResult:
    """Compute the confusion matrix for trtcheck against ground-truth labels.

    Each manifest entry's `expected` is the ground truth; the matching
    `outcomes[name].trtcheck` is the trtcheck prediction. When an entry has
    no matching prediction it's added to `skipped`. When the manifest and a
    provided trtexec outcome disagree, the entry is added to `drift`
    (informational; does not affect the matrix).
    """
    result = ScoreResult()
    for entry in manifest_entries:
        name = entry["name"]
        expected = entry["expected"]
        if expected not in _VALID_OUTCOMES:
            raise ValueError(f"manifest entry '{name}' has invalid expected={expected!r}")

        pred_block = outcomes.get(name)
        if not pred_block or "trtcheck" not in pred_block:
            result.skipped.append(name)
            continue

        trtcheck_pred = pred_block["trtcheck"]
        if trtcheck_pred not in _VALID_PREDICTIONS:
            raise ValueError(f"outcomes['{name}'].trtcheck has invalid value {trtcheck_pred!r}")

        trtexec_pred = pred_block.get("trtexec")
        if trtexec_pred and trtexec_pred != expected:
            result.drift.append(name)

        if trtcheck_pred == "unverified":
            # Never counted as success: tracked separately, both ways.
            if expected == "fail":
                result.unverified_on_fail.append(name)
            else:
                result.unverified_on_convert.append(name)
            continue

        # "fail" is the positive class
        if trtcheck_pred == "fail" and expected == "fail":
            result.true_positive += 1
        elif trtcheck_pred == "fail" and expected == "convert":
            result.false_positive += 1
        elif trtcheck_pred == "convert" and expected == "convert":
            result.true_negative += 1
        else:  # trtcheck_pred == "convert" and expected == "fail"
            result.false_negative += 1
    return result


def format_report(s: ScoreResult) -> str:
    lines: list[str] = []
    lines.append("trtcheck validation score")
    lines.append("-" * 40)
    lines.append(f"  scored:        {s.total}")
    lines.append(f"  true positive: {s.true_positive}   (trtcheck=fail, expected=fail)")
    lines.append(f"  false positive:{s.false_positive}   (trtcheck=fail, expected=convert)")
    lines.append(f"  true negative: {s.true_negative}   (trtcheck=convert, expected=convert)")
    lines.append(f"  false negative:{s.false_negative}   (trtcheck=convert, expected=fail)")
    lines.append("")
    lines.append(f"  blocker precision: {s.precision:.3f}")
    lines.append(f"  blocker recall:    {s.recall:.3f}")
    lines.append(f"  blocker f1:        {s.f1:.3f}")
    if s.unverified_total:
        lines.append("")
        lines.append(
            f"  unverified: {s.unverified_total} "
            f"(coverage {s.unverified_coverage:.3f}; "
            f"{len(s.unverified_on_fail)} were real failures, "
            f"{len(s.unverified_on_convert)} actually convert)"
        )
        for name in s.unverified_on_fail + s.unverified_on_convert:
            lines.append(f"    - {name}")
    if s.skipped:
        lines.append("")
        lines.append(f"  skipped (no prediction): {len(s.skipped)}")
        for name in s.skipped:
            lines.append(f"    - {name}")
    if s.drift:
        lines.append("")
        lines.append(f"  manifest/trtexec drift: {len(s.drift)} (re-label these)")
        for name in s.drift:
            lines.append(f"    - {name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "bench" / "manifest.yaml",
        help="Path to bench/manifest.yaml.",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        required=True,
        help="Path to outcomes.json produced by the validation runner.",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.manifest) as f:
            manifest = yaml.safe_load(f).get("models", [])
        outcomes_doc = json.loads(args.outcomes.read_text())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"score: {exc}", file=sys.stderr)
        return 2

    outcomes = outcomes_doc.get("predictions", {})
    if not isinstance(outcomes, dict):
        print("score: outcomes.predictions must be an object", file=sys.stderr)
        return 2

    result = score(manifest, outcomes)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
