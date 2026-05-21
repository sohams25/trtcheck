"""Tests for bench/score.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.score import ScoreResult, format_report, main, score


def _manifest(*entries: tuple[str, str]) -> list[dict[str, str]]:
    return [{"name": n, "expected": e} for n, e in entries]


class TestScore:
    def test_all_correct_predictions(self) -> None:
        manifest = _manifest(("a", "fail"), ("b", "convert"))
        outcomes = {
            "a": {"trtcheck": "fail"},
            "b": {"trtcheck": "convert"},
        }
        r = score(manifest, outcomes)
        assert (r.true_positive, r.false_positive, r.true_negative, r.false_negative) == (
            1, 0, 1, 0,
        )
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_false_positive_and_false_negative(self) -> None:
        manifest = _manifest(
            ("tp", "fail"),
            ("fp", "convert"),
            ("tn", "convert"),
            ("fn", "fail"),
        )
        outcomes = {
            "tp": {"trtcheck": "fail"},
            "fp": {"trtcheck": "fail"},
            "tn": {"trtcheck": "convert"},
            "fn": {"trtcheck": "convert"},
        }
        r = score(manifest, outcomes)
        assert (r.true_positive, r.false_positive, r.true_negative, r.false_negative) == (
            1, 1, 1, 1,
        )
        assert r.precision == 0.5
        assert r.recall == 0.5
        assert r.f1 == 0.5

    def test_entry_without_prediction_is_skipped(self) -> None:
        manifest = _manifest(("a", "fail"), ("b", "fail"))
        outcomes = {"a": {"trtcheck": "fail"}}
        r = score(manifest, outcomes)
        assert r.skipped == ["b"]
        assert r.total == 1
        # the skipped row must not affect the matrix
        assert r.true_positive == 1

    def test_trtexec_drift_is_recorded_without_affecting_matrix(self) -> None:
        manifest = _manifest(("a", "fail"), ("b", "convert"))
        outcomes = {
            "a": {"trtcheck": "fail", "trtexec": "fail"},
            "b": {"trtcheck": "convert", "trtexec": "fail"},  # drift: manifest=convert
        }
        r = score(manifest, outcomes)
        assert "b" in r.drift
        # Matrix uses manifest's expected, not the diverging trtexec
        assert r.true_negative == 1

    def test_empty_inputs_yield_zero_metrics(self) -> None:
        r = score([], {})
        assert r.total == 0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_only_negatives_gives_zero_precision_recall(self) -> None:
        """No 'fail' predictions and no 'fail' ground truth -> degenerate
        positive class; precision and recall must not blow up."""
        manifest = _manifest(("a", "convert"), ("b", "convert"))
        outcomes = {
            "a": {"trtcheck": "convert"},
            "b": {"trtcheck": "convert"},
        }
        r = score(manifest, outcomes)
        assert r.true_negative == 2
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0

    def test_invalid_expected_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid expected"):
            score(_manifest(("a", "maybe")), {"a": {"trtcheck": "fail"}})

    def test_invalid_trtcheck_prediction_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid value"):
            score(_manifest(("a", "fail")), {"a": {"trtcheck": "huh"}})


class TestReport:
    def test_report_includes_all_metrics(self) -> None:
        r = ScoreResult(true_positive=2, false_positive=1, true_negative=3, false_negative=1)
        text = format_report(r)
        assert "true positive: 2" in text
        assert "false positive:1" in text
        assert "true negative: 3" in text
        assert "false negative:1" in text
        assert "precision:" in text and "recall:" in text and "f1:" in text

    def test_report_surfaces_skipped(self) -> None:
        r = ScoreResult(skipped=["nope"])
        assert "nope" in format_report(r)
        assert "skipped" in format_report(r).lower()

    def test_report_surfaces_drift(self) -> None:
        r = ScoreResult(drift=["b"])
        text = format_report(r)
        assert "drift" in text.lower()
        assert "b" in text


class TestMain:
    def test_cli_reads_manifest_and_outcomes(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            "models:\n"
            "  - name: a\n    source: bundled.onnx\n    expected: fail\n"
        )
        outcomes_path = tmp_path / "out.json"
        outcomes_path.write_text(
            json.dumps({"predictions": {"a": {"trtcheck": "fail"}}})
        )
        rc = main(["--manifest", str(manifest_path), "--outcomes", str(outcomes_path)])
        assert rc == 0

    def test_cli_errors_on_bad_outcomes_shape(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("models: []\n")
        outcomes_path = tmp_path / "out.json"
        outcomes_path.write_text(json.dumps({"predictions": []}))  # list, not dict
        rc = main(["--manifest", str(manifest_path), "--outcomes", str(outcomes_path)])
        assert rc == 2
