"""Download URL-based manifest entries into bench/cache/.

Reads bench/manifest.yaml, fetches every entry whose `source` starts with
http(s)://, and writes it to bench/cache/<name>.onnx. Skips files that
already exist. Verifies SHA-256 when the manifest lists one.

Manifest entries with a repo-relative `source` (bundled fixtures) are
left alone -- they're already in the tree.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _REPO_ROOT / "bench" / "manifest.yaml"
_CACHE_DIR = _REPO_ROOT / "bench" / "cache"

# A manifest 'name' becomes a cache filename, so it must not contain path
# separators or traversal sequences -- otherwise a hostile manifest could write
# outside bench/cache/. Require a leading alphanumeric so dot-only names (".",
# "..", "...") and leading-dot names are rejected, not just bare "."/"..".
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _safe_dest(cache_dir: Path, name: str) -> Path:
    """Build the cache path for `name`, refusing anything that could escape it."""
    if not _SAFE_NAME_RE.match(name):
        raise RuntimeError(
            f"unsafe manifest entry name {name!r}: must start alphanumeric and "
            "contain only [A-Za-z0-9._-]"
        )
    dest = (cache_dir / f"{name}.onnx").resolve()
    if cache_dir.resolve() not in dest.parents:
        raise RuntimeError(f"manifest entry name {name!r} resolves outside the cache dir")
    return dest


def load_manifest(path: Path = _MANIFEST) -> list[dict[str, Any]]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    models = doc.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"{path}: 'models' must be a list, got {type(models).__name__}")
    return models


def is_url(source: str) -> bool:
    return source.startswith("https://") or source.startswith("http://")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_entry(entry: dict[str, Any], cache_dir: Path = _CACHE_DIR) -> Path | None:
    """Fetch one entry. Returns the cached path, or None if no fetch needed
    (bundled-path entry).
    """
    source = entry["source"]
    if not is_url(source):
        return None

    dest = _safe_dest(cache_dir, entry["name"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        expected_hash = entry.get("sha256")
        if expected_hash:
            actual = sha256_of(dest)
            if actual != expected_hash:
                raise RuntimeError(
                    f"{entry['name']}: cached file hash mismatch "
                    f"(expected {expected_hash[:16]}..., got {actual[:16]}...)"
                )
        return dest

    print(f"fetching {entry['name']} from {source}")
    try:
        with urllib.request.urlopen(source, timeout=60) as resp:
            data = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{entry['name']}: download failed: {exc.reason}") from None

    expected_hash = entry.get("sha256")
    if expected_hash:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_hash:
            raise RuntimeError(
                f"{entry['name']}: downloaded hash mismatch "
                f"(expected {expected_hash[:16]}..., got {actual[:16]}...)"
            )

    dest.write_bytes(data)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST,
        help="Path to bench/manifest.yaml.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_CACHE_DIR,
        help="Where downloaded models go.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Only fetch entries with this name. Can be passed multiple times.",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"fetch: {exc}", file=sys.stderr)
        return 2

    if args.name:
        wanted = set(args.name)
        entries = [e for e in entries if e.get("name") in wanted]

    fetched = 0
    skipped = 0
    for entry in entries:
        try:
            result = fetch_entry(entry, args.cache_dir)
        except RuntimeError as exc:
            print(f"  error: {exc}", file=sys.stderr)
            return 1
        if result is None:
            skipped += 1
        else:
            fetched += 1

    print(f"fetched/cached: {fetched}, skipped (bundled): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
