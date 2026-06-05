"""Keep the version string in lockstep across the package, pyproject, and Action.

A consumer who adopts the GitHub Action without pinning installs its default
version; if that drifts from the released package, they silently run a stale
analyzer. These tests fail the build when the three sources disagree.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from trtcheck import __version__

_REPO = Path(__file__).resolve().parent.parent


def test_pyproject_version_matches_package() -> None:
    text = (_REPO / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "could not find version in pyproject.toml"
    assert (
        match.group(1) == __version__
    ), f"pyproject version {match.group(1)} != trtcheck.__version__ {__version__}"


def test_action_default_version_matches_package() -> None:
    action = yaml.safe_load((_REPO / "action.yml").read_text())
    default = action["inputs"]["version"]["default"]
    assert default == __version__, (
        f"action.yml inputs.version.default {default!r} != trtcheck.__version__ "
        f"{__version__!r}; bump action.yml on every release."
    )
