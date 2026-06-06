"""Shared sanitization for model-derived strings.

A node name, operator, message, remediation, filename, or producer comes
straight from an untrusted ONNX file. Before any of it reaches a terminal or an
HTML document we strip two classes of dangerous characters:

* **ASCII / C1 control characters** (keeping tab and newline). A hostile model
  could otherwise smuggle ANSI escape sequences (cursor moves, screen clears)
  into a terminal, or a NUL byte that makes an HTML file byte-invalid and breaks
  downstream HTML/XML consumers.
* **Unicode bidirectional / formatting overrides** (Trojan-Source). U+202A-E,
  U+2066-9, U+200E/200F visually reorder text so a malicious "fix" command can
  be made to read as something benign.

Both the console and HTML reporters route every model-derived field through
:func:`strip_unsafe` so the two output formats can never drift apart on what
counts as safe. (The JSON reporter relies on ``json.dumps`` escaping, which is
already lossless and safe for control characters.)
"""

from __future__ import annotations

import re

# Keep \t (0x09) and \n (0x0a). Strip the rest of C0, DEL (0x7f), the C1 block
# (0x80-0x9f -- 0x9b is the single-byte CSI ANSI introducer), and the Unicode
# bidi/format overrides used in Trojan-Source display-spoofing attacks
# (U+200E/200F marks, U+202A-202E embeddings/overrides, U+2066-2069 isolates).
_UNSAFE_CHARS = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f" "\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def strip_unsafe(text: str) -> str:
    """Remove control and bidi-override characters from untrusted text."""
    return _UNSAFE_CHARS.sub("", text)
