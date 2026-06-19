"""Evidence-requirement loading and injection.

`evidence_requirements.csv` is a checklist keyed by (claim_object, applies_to
issue-family). We don't try to deterministically guess the issue family from
the free-text claim — that's brittle. Instead we inject ALL requirement rows
relevant to the claim's object (plus rows where claim_object == 'all') into the
Stage 2 prompt and instruct the model to select the matching family and judge
whether `minimum_image_evidence` is satisfied. The list per object is small,
so the token cost is negligible and the model gets the full rubric.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


class EvidenceRequirements:
    def __init__(self, rows: List[Dict[str, str]]):
        self._rows = rows

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceRequirements":
        rows: List[Dict[str, str]] = []
        p = Path(path)
        if not p.exists():
            return cls(rows)
        with open(p, "r", encoding="utf-8", newline="") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)]
        return cls(rows)

    def for_object(self, claim_object: str) -> List[Dict[str, str]]:
        obj = str(claim_object).strip().lower()
        return [
            r for r in self._rows
            if (r.get("claim_object", "").strip().lower() in {obj, "all"})
        ]

    def rubric_text(self, claim_object: str) -> str:
        """Render the applicable requirements as a compact checklist block."""
        rows = self.for_object(claim_object)
        if not rows:
            return "(no specific evidence requirements provided for this object)"
        lines = []
        for r in rows:
            applies = r.get("applies_to", "").strip()
            req = r.get("minimum_image_evidence", "").strip()
            rid = r.get("requirement_id", "").strip()
            lines.append(f"- [{rid}] issue family \"{applies}\": minimum evidence = {req}")
        return "\n".join(lines)
