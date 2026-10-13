"""Findings: the contract, the deterministic claim checker, and the rule-assigned status.

The three modules are deliberately separable. `schema.py` is the contract the PRD's insight
registry can later absorb; `checker.py` is arithmetic and text, with no model in it; and
`status_rules.py` is a pure function of already-computed numbers, so the one word a reader is most
likely to over-read (`supported`) is decided by a rule that can be unit-tested.
"""

from __future__ import annotations

from .checker import Verdict, check_draft, check_numbers_in_text
from .schema import Evidence, Finding, FindingDraft, FindingEffect, FindingScope, FindingsFile
from .status_rules import assign_status, status_badge, summarise_status_counts

__all__ = [
    "Evidence", "Finding", "FindingDraft", "FindingEffect", "FindingScope", "FindingsFile",
    "Verdict", "assign_status", "check_draft", "check_numbers_in_text", "status_badge",
    "summarise_status_counts",
]
