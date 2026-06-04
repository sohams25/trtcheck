"""Post or update the sticky trtcheck comment on a pull request.

Used by the second workflow (`workflow_run`) in the dual-workflow safe-PR
pattern. Reads the rendered markdown from --body-file, finds an existing
comment by the marker, and either PATCHes or POSTs.

Talks to the GitHub REST API via stdlib urllib so the action has no
runtime dependencies beyond Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

MARKER = "<!-- trtcheck-sticky:v1 -->"


def _request(
    method: str, url: str, token: str, payload: dict[str, Any] | None = None
) -> dict[str, Any] | list[dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
            if not body:
                return {}
            result: dict[str, Any] | list[dict[str, Any]] = json.loads(body)
            return result
    except urllib.error.HTTPError as exc:
        # Surface status + reason only. Never let the request object (which
        # carries the Authorization header) hit a traceback in CI logs.
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {exc.reason}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API {method} {url} unreachable: {exc.reason}") from None


def find_sticky_comment_id(repo: str, pr_number: int, token: str) -> int | None:
    """Return the id of an existing trtcheck sticky comment, or None."""
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
            f"?per_page=100&page={page}"
        )
        comments = _request("GET", url, token)
        if not isinstance(comments, list) or not comments:
            return None
        for c in comments:
            body = c.get("body") or ""
            if body.lstrip().startswith(MARKER):
                return int(c["id"])
        if len(comments) < 100:
            return None
        page += 1


def upsert_comment(repo: str, pr_number: int, body: str, token: str) -> tuple[str, int]:
    """Create or update the sticky comment. Returns (action, comment_id)."""
    existing = find_sticky_comment_id(repo, pr_number, token)
    if existing is not None:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing}"
        _request("PATCH", url, token, payload={"body": body})
        return "updated", existing
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    created = _request("POST", url, token, payload={"body": body})
    if isinstance(created, dict):
        return "created", int(created["id"])
    return "created", 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--pr", required=True, type=int, help="pull request number")
    parser.add_argument("--body-file", required=True, help="path to markdown body to post")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("post_comment: GITHUB_TOKEN not set", file=sys.stderr)
        return 2

    with open(args.body_file, encoding="utf-8") as fh:
        body = fh.read()
    if not body.lstrip().startswith(MARKER):
        # Defense in depth: refuse to post a body that lacks the marker.
        # Otherwise we could create comments we can't later find and update.
        print("post_comment: body is missing sticky marker", file=sys.stderr)
        return 2

    action, comment_id = upsert_comment(args.repo, args.pr, body, token)
    print(f"{action} comment id={comment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
