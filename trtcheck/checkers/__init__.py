"""Checker plugins.

The `Checker` Protocol lives in `trtcheck.plugins`; this module re-exports
it for back-compat. New code should import directly from
`trtcheck.plugins`.
"""

from __future__ import annotations

from trtcheck.plugins import Checker

__all__ = ["Checker"]
