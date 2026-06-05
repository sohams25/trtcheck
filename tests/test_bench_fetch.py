"""Tests for bench/fetch.py cache-path safety.

A manifest 'name' is attacker-influenced data turned into a filesystem path, so
a hostile name must not be able to write outside bench/cache/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.fetch import _safe_dest, fetch_entry


def test_safe_dest_accepts_clean_name(tmp_path: Path) -> None:
    dest = _safe_dest(tmp_path, "resnet50")
    assert dest == (tmp_path / "resnet50.onnx").resolve()
    assert tmp_path.resolve() in dest.parents


@pytest.mark.parametrize(
    "bad",
    [
        "../evil",
        "../../etc/passwd",
        "a/b",
        "..",
        ".",
        "...",
        ".hidden",
        "x/../../y",
        "with space",
    ],
)
def test_safe_dest_rejects_traversal_and_separators(tmp_path: Path, bad: str) -> None:
    with pytest.raises(RuntimeError):
        _safe_dest(tmp_path, bad)


def test_fetch_entry_rejects_malicious_name_before_network(tmp_path: Path) -> None:
    # A URL source would normally be downloaded; the name is validated first, so
    # this raises without any network access and writes nothing outside the cache.
    entry = {"name": "../../pwned", "source": "https://example.invalid/model.onnx"}
    with pytest.raises(RuntimeError):
        fetch_entry(entry, cache_dir=tmp_path)
    assert not (tmp_path.parent / "pwned.onnx").exists()


def test_fetch_entry_skips_bundled_entries(tmp_path: Path) -> None:
    # Non-URL (bundled) entries are left in place and never build a cache path.
    entry = {"name": "clean_minimal", "source": "tests/fixtures/clean_minimal.onnx"}
    assert fetch_entry(entry, cache_dir=tmp_path) is None
