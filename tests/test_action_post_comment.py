"""Tests for action/post_comment.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

_PC_PATH = Path(__file__).parent.parent / "action" / "post_comment.py"


def _import_post_comment():
    spec = importlib.util.spec_from_file_location("post_comment", _PC_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pc():
    return _import_post_comment()


class TestFindStickyComment:
    def test_returns_id_when_marker_first(self, pc) -> None:
        comments = [
            {"id": 11, "body": "some random comment"},
            {"id": 22, "body": f"{pc.MARKER}\n\n## trtcheck report\n..."},
        ]
        with patch.object(pc, "_request", return_value=comments) as mock:
            result = pc.find_sticky_comment_id("o/r", 7, "tok")
        assert result == 22
        mock.assert_called_once()

    def test_returns_none_when_no_marker(self, pc) -> None:
        comments = [{"id": 1, "body": "nope"}, {"id": 2, "body": "still nope"}]
        with patch.object(pc, "_request", return_value=comments):
            assert pc.find_sticky_comment_id("o/r", 7, "tok") is None

    def test_returns_none_when_empty(self, pc) -> None:
        with patch.object(pc, "_request", return_value=[]):
            assert pc.find_sticky_comment_id("o/r", 7, "tok") is None

    def test_paginates_when_full_page(self, pc) -> None:
        full_page = [{"id": i, "body": "nope"} for i in range(100)]
        next_page = [{"id": 999, "body": f"{pc.MARKER}\n..."}]
        with patch.object(pc, "_request", side_effect=[full_page, next_page]) as mock:
            result = pc.find_sticky_comment_id("o/r", 7, "tok")
        assert result == 999
        assert mock.call_count == 2


class TestUpsert:
    def test_updates_when_existing(self, pc) -> None:
        with patch.object(pc, "find_sticky_comment_id", return_value=42):
            with patch.object(pc, "_request", return_value={}) as mock:
                action, cid = pc.upsert_comment("o/r", 7, f"{pc.MARKER}\nhi", "tok")
        assert action == "updated"
        assert cid == 42
        args, kwargs = mock.call_args
        assert args[0] == "PATCH"

    def test_creates_when_absent(self, pc) -> None:
        with patch.object(pc, "find_sticky_comment_id", return_value=None):
            with patch.object(pc, "_request", return_value={"id": 100}) as mock:
                action, cid = pc.upsert_comment("o/r", 7, f"{pc.MARKER}\nhi", "tok")
        assert action == "created"
        assert cid == 100
        args, kwargs = mock.call_args
        assert args[0] == "POST"


class TestMain:
    def test_refuses_body_without_marker(self, pc, tmp_path: Path) -> None:
        body = tmp_path / "body.md"
        body.write_text("no marker here")
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            rc = pc.main(["--repo", "o/r", "--pr", "1", "--body-file", str(body)])
        assert rc == 2

    def test_errors_without_token(self, pc, tmp_path: Path) -> None:
        body = tmp_path / "body.md"
        body.write_text(f"{pc.MARKER}\nhi")
        with patch.dict("os.environ", {}, clear=True):
            rc = pc.main(["--repo", "o/r", "--pr", "1", "--body-file", str(body)])
        assert rc == 2
