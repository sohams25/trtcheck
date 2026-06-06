"""Shared sanitization for model-derived strings.

A node name, operator, message, remediation, filename, or producer comes
straight from an untrusted ONNX file. Before any of it reaches a terminal or an
HTML document we strip two classes of dangerous characters:

* **ASCII / C1 control characters** (keeping tab and newline). A hostile model
  could otherwise smuggle ANSI escape sequences (cursor moves, screen clears)
  into a terminal, or a NUL byte that makes an HTML file byte-invalid and breaks
  downstream HTML/XML consumers.
* **Unicode bidirectional / zero-width formatting characters** (Trojan-Source,
  CVE-2021-42574). Bidi overrides visually reorder text so a malicious "fix"
  command can be made to read as something benign; zero-width chars splice or
  hide content. Neither has any legitimate use in a model-derived string.

Both the console and HTML reporters route every model-derived field through
:func:`strip_unsafe` so the two output formats can never drift apart on what
counts as safe. (The JSON reporter relies on ``json.dumps`` escaping, which is
already lossless and safe for control characters.)
"""

from __future__ import annotations

import re

# One character class -- a single literal; the brackets span the whole pattern.
# Keep \t (U+0009) and \n (U+000A). Strip:
#   U+0000-0008, U+000B-001F  rest of the C0 control block
#   U+007F-009F               DEL + C1 block (U+009B is the single-byte CSI
#                             ANSI introducer)
#   U+200B-200F               zero-width space/(non-)joiner + LRM/RLM marks
#   U+202A-202E               bidi embeddings/overrides (Trojan-Source)
#   U+2066-2069               bidi isolates (Trojan-Source)
#   U+FEFF                    BOM / zero-width no-break space (stego carrier)
_UNSAFE_CHARS = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)


def strip_unsafe(text: str) -> str:
    """Remove control, bidi-override, and zero-width characters from untrusted text."""
    return _UNSAFE_CHARS.sub("", text)
