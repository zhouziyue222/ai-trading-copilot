"""Deterministic rechecks for untrusted analyst evidence.

The keyword scan is deliberately conservative and only blocks an action that
an injected report tried to hide; it never unlocks anything.
# ponytail: regex over free text; upgrade to structured evidence extraction
when analyst outputs carry per-item normalized risk fields.
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple


_SEVERE_PATTERN = re.compile(
    r"(?:going concern|sec investigation|class action|accounting restatement|"
    r"delisting|bankruptcy|fraud|material weakness|accounting irregularity|"
    r"securities lawsuit)",
    re.IGNORECASE,
)


def severe_evidence_hits(evidence: str | None, limit: int = 5) -> list[str]:
    if not evidence:
        return []
    return sorted(
        {
            match.group(0).lower()
            for match in _SEVERE_PATTERN.finditer(str(evidence))
        }
    )[:limit]


def force_material_risk_from_evidence(
    *,
    evidence: str | None,
    material_risk: bool,
    risk_flags: Iterable[str],
    source: str,
) -> Tuple[bool, list[str], list[str]]:
    """Return (material_risk, risk_flags, guard_notes) after evidence recheck."""

    flags = list(risk_flags)
    notes: list[str] = []
    hits = severe_evidence_hits(evidence)
    if not material_risk and hits:
        material_risk = True
        flags = [*flags, *(f"deterministic_{source}_recheck:{hit}" for hit in hits)]
        notes.append(
            f"deterministic {source} recheck forced material_risk=true: "
            + ", ".join(hits)
        )
    if material_risk and not flags:
        flags = ["unspecified_material_risk"]
        notes.append("material_risk=true carried no risk_flags; added unspecified flag.")
    elif not material_risk and flags:
        material_risk = True
        notes.append("risk_flags present without material_risk; forced material_risk=true.")
    return material_risk, flags, notes
