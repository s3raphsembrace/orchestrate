"""User history loading, compact summarization, and risk-flag derivation.

History adds *risk context* only. Per the spec it must not override clear
visual evidence on its own, so the pipeline injects a short summary into the
Stage 2 prompt and separately derives deterministic risk flags that are merged
into the final risk_flags.

The dataset already encodes the intended history signal in the `history_flags`
column (observed values: `none`, `user_history_risk`,
`user_history_risk;manual_review_required`, `manual_review_required`). We
propagate those recognized tokens verbatim rather than re-deriving them from raw
counts — this reproduces the labeled `user_history_risk` / `manual_review_required`
flags exactly on the sample set, where a counts-based heuristic disagreed on the
user whose `history_flags` was `manual_review_required` (not `user_history_risk`).
The numeric `is_risky` heuristic is retained as a secondary, conservative signal
for users whose `history_flags` are absent or `none`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

# Risk tokens the history file is allowed to assert into the output directly.
HISTORY_RISK_TOKENS = {"user_history_risk", "manual_review_required"}


class UserHistory:
    def __init__(self, rows_by_user: Dict[str, Dict[str, str]]):
        self._rows = rows_by_user

    @classmethod
    def load(cls, path: str | Path) -> "UserHistory":
        rows: Dict[str, Dict[str, str]] = {}
        p = Path(path)
        if not p.exists():
            return cls(rows)
        with open(p, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows[row["user_id"].strip()] = row
        return cls(rows)

    def get(self, user_id: str) -> Optional[Dict[str, str]]:
        return self._rows.get(str(user_id).strip())

    def summary(self, user_id: str) -> str:
        """One-line, model-friendly summary. Empty string if no record."""
        row = self.get(user_id)
        if not row:
            return "No prior history on file for this user."
        flags = (row.get("history_flags") or "").strip()
        summ = (row.get("history_summary") or "").strip()
        parts = [
            f"past_claims={row.get('past_claim_count', '0')}",
            f"accepted={row.get('accept_claim', '0')}",
            f"manual_review={row.get('manual_review_claim', '0')}",
            f"rejected={row.get('rejected_claim', '0')}",
            f"last_90d={row.get('last_90_days_claim_count', '0')}",
        ]
        line = ", ".join(parts)
        if flags and flags.lower() != "none":
            line += f"; flags: {flags}"
        if summ:
            line += f"; note: {summ}"
        return line

    def risk_flags(self, user_id: str) -> List[str]:
        """Recognized risk tokens asserted by the user's `history_flags` column.

        This is the primary, exact history signal: the column already carries
        the intended `user_history_risk` / `manual_review_required` tokens, so
        we surface them directly (closed vocabulary, order-preserving, deduped).
        Returns [] when there is no record or the column is `none`/empty.
        """
        row = self.get(user_id)
        if not row:
            return []
        raw = (row.get("history_flags") or "").strip().lower()
        if not raw or raw == "none":
            return []
        out: List[str] = []
        for tok in raw.replace(",", ";").split(";"):
            t = tok.strip()
            if t in HISTORY_RISK_TOKENS and t not in out:
                out.append(t)
        return out

    def is_risky(self, user_id: str) -> bool:
        """Deterministic risk signal used to set the user_history_risk flag.

        Triggers on explicit history flags, a high recent-claim burst, or a
        rejection-heavy track record. Tunable; intentionally conservative.
        """
        row = self.get(user_id)
        if not row:
            return False
        flags = (row.get("history_flags") or "").strip().lower()
        if flags and flags not in {"none", ""}:
            return True

        def _int(key: str) -> int:
            try:
                return int(float(row.get(key) or 0))
            except (ValueError, TypeError):
                return 0

        last_90 = _int("last_90_days_claim_count")
        rejected = _int("rejected_claim")
        past = _int("past_claim_count")
        if last_90 >= 3:
            return True
        if past >= 3 and rejected / max(past, 1) >= 0.5:
            return True
        return False
