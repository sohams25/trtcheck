"""Reporter plugins.

The `Reporter` Protocol lives in `trtcheck.plugins`; this module
re-exports it for back-compat.
"""

from __future__ import annotations

from trtcheck.plugins import Reporter

__all__ = ["Reporter"]
