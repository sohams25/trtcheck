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
            1,
            0,
            1,
            0,
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
            1,
            1,
            1,
            1,
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
            "models:\n" "  - name: a\n    source: bundled.onnx\n    expected: fail\n"
        )
        outcomes_path = tmp_path / "out.json"
        outcomes_path.write_text(json.dumps({"predictions": {"a": {"trtcheck": "fail"}}}))
        rc = main(["--manifest", str(manifest_path), "--outcomes", str(outcomes_path)])
        assert rc == 0

    def test_cli_errors_on_bad_outcomes_shape(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("models: []\n")
        outcomes_path = tmp_path / "out.json"
        outcomes_path.write_text(json.dumps({"predictions": []}))  # list, not dict
        rc = main(["--manifest", str(manifest_path), "--outcomes", str(outcomes_path)])
        assert rc == 2


class TestUnverifiedPredictions:
    def test_unverified_is_never_counted_as_success(self) -> None:
        manifest = _manifest(("real_fail", "fail"), ("real_convert", "convert"))
        outcomes = {
            "real_fail": {"trtcheck": "unverified"},
            "real_convert": {"trtcheck": "unverified"},
        }
        r = score(manifest, outcomes)
        # Nothing lands in the confusion matrix -- and nothing reads as a pass.
        assert r.total == 0
        assert r.unverified_on_fail == ["real_fail"]
        assert r.unverified_on_convert == ["real_convert"]
        assert r.unverified_total == 2
        assert r.unverified_coverage == 1.0

    def test_unverified_coverage_mixes_with_classified(self) -> None:
        manifest = _manifest(("a", "fail"), ("b", "convert"), ("c", "fail"))
        outcomes = {
            "a": {"trtcheck": "fail"},
            "b": {"trtcheck": "convert"},
            "c": {"trtcheck": "unverified"},
        }
        r = score(manifest, outcomes)
        assert r.true_positive == 1 and r.true_negative == 1
        assert r.unverified_coverage == pytest.approx(1 / 3)
        assert "unverified" in format_report(r)

    def test_bogus_prediction_value_still_raises(self) -> None:
        manifest = _manifest(("a", "fail"))
        with pytest.raises(ValueError):
            score(manifest, {"a": {"trtcheck": "maybe"}})


def test_score_to_dict_and_json_flag(tmp_path: Path) -> None:
    manifest = _manifest(("a", "fail"), ("b", "convert"), ("c", "fail"))
    outcomes = {
        "a": {"trtcheck": "fail"},
        "b": {"trtcheck": "convert"},
        "c": {"trtcheck": "unverified"},
    }
    r = score(manifest, outcomes)
    d = r.to_dict()
    assert d["blocker_precision"] == 1.0
    assert d["blocker_recall"] == 1.0
    assert d["unverified_on_fail"] == ["c"]
    assert d["unverified_coverage"] == pytest.approx(1 / 3)

    # CLI round trip
    mpath = tmp_path / "manifest.yaml"
    opath = tmp_path / "outcomes.json"
    jpath = tmp_path / "summary.json"
    import yaml

    mpath.write_text(yaml.safe_dump({"models": manifest}))
    opath.write_text(json.dumps({"predictions": outcomes}))
    assert main(["--manifest", str(mpath), "--outcomes", str(opath), "--json", str(jpath)]) == 0
    assert json.loads(jpath.read_text())["scored"] == 2
