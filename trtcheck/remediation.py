"""Single source of truth for finding remediation text.

``data/remediation_db.json`` holds, per issue id, the human-facing
``summary`` / ``explanation`` / ``remediation`` / ``docs_link`` and the
``severity`` / ``category`` for every finding the non-operator checkers
(precision, control_flow, dynamic_shapes, graph_structure) can emit. This
module loads it once and hands back typed, frozen entries so the checkers no
longer hard-code (and silently drift from) that prose.

Scope: this is single-source for ``explanation`` / ``remediation`` /
``docs_link`` / ``severity`` only. The *per-node* prefix of a message ("Input
'x' has dtype UINT8") is necessarily built in the checker from node context.
``operator_support`` intentionally does NOT use this file -- it owns
``operator_matrix.json``, which carries per-operator, per-TRT-version data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from trtcheck.types import CheckCategory, Confidence, Issue, Severity


@dataclass(frozen=True)
class RemediationEntry:
    """One issue class's canonical metadata, loaded from remediation_db.json."""

    category: CheckCategory
    severity: Severity
    summary: str
    explanation: str
    remediation: str
    docs_link: str | None = None
    rule_id: str = ""
    confidence: Confidence = Confidence.HIGH
    verify_required: bool = False


def _to_entry(key: str, raw: dict[str, Any]) -> RemediationEntry:
    """Build a typed entry from a raw JSON dict, failing loudly on bad data.

    ``Severity(...)`` / ``CheckCategory(...)`` raise ``ValueError`` if the JSON
    carries a value outside the enum -- we surface that with the offending key
    so a bad edit to remediation_db.json points at itself.
    """
    try:
        return RemediationEntry(
            category=CheckCategory(raw["category"]),
            severity=Severity(raw["severity"]),
            summary=raw["summary"],
            explanation=raw["explanation"],
            remediation=raw["remediation"],
            docs_link=raw.get("docs_link"),
            rule_id=raw["rule_id"],
            confidence=Confidence(raw.get("confidence", "high")),
            verify_required=bool(raw.get("verify_required", False)),
        )
    except (ValueError, KeyError) as exc:
        raise ValueError(f"remediation_db.json entry {key!r} is invalid: {exc}") from exc


def _parse(text: str) -> dict[str, RemediationEntry]:
    """Parse the DB JSON text into typed entries, failing loudly on bad data."""
    try:
        issues = json.loads(text)["issues"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"remediation_db.json is malformed: {exc}") from exc
    return {key: _to_entry(key, entry) for key, entry in issues.items()}


def _load() -> dict[str, RemediationEntry]:
    text = resources.files("trtcheck.data").joinpath("remediation_db.json").read_text()
    return _parse(text)


_DB: dict[str, RemediationEntry] = _load()


def get(issue_id: str) -> RemediationEntry:
    """Return the entry for ``issue_id``. Raises ``KeyError`` if unknown."""
    return _DB[issue_id]


def known_ids() -> frozenset[str]:
    """Every issue id defined in remediation_db.json."""
    return frozenset(_DB)


def rule_ids() -> frozenset[str]:
    """Every stable rule id defined in remediation_db.json."""
    return frozenset(e.rule_id for e in _DB.values())


def make_issue(
    issue_id: str,
    *,
    node_name: str,
    operator: str,
    prefix: str,
    graph_scope: str = "",
    target_trt: str | None = None,
) -> Issue:
    """Build an :class:`Issue` for ``issue_id``.

    The per-node ``prefix`` (built by the checker from node context, e.g.
    "Input 'x' has dtype UINT8") is joined with the entry's generic
    ``explanation``; ``severity`` / ``category`` / ``remediation`` /
    ``docs_link`` come verbatim from the DB so they live in exactly one place.
    """
    entry = get(issue_id)
    return Issue(
        severity=entry.severity,
        category=entry.category,
        node_name=node_name,
        operator=operator,
        message=f"{prefix}. {entry.explanation}",
        remediation=entry.remediation,
        docs_link=entry.docs_link,
        rule_id=entry.rule_id,
        confidence=entry.confidence,
        verify_required=entry.verify_required,
        graph_scope=graph_scope,
        target_trt=target_trt,
    )
