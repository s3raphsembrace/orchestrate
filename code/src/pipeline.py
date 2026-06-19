"""Two-stage orchestration for a single claim, plus a batch runner.

Stage 1: one cheap validity-gate call over the whole image set. Emits
quality/authenticity flags and the subset of usable image IDs. If the set is
unusable and short_circuit is on, we skip Stage 2 and return a
not_enough_information / invalid result — this is where most of the savings on
junk submissions come from.

Stage 2: one analysis call with the object-specific prompt, the evidence
rubric, the history summary, and the usable images. Produces the substantive
fields.

Merge: union the Stage 1 + Stage 2 risk flags, add the deterministic
user_history_risk flag, add manual_review_required when the result is low-trust,
and normalize everything to the closed vocabulary.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from .cache import ResponseCache
from .evidence import EvidenceRequirements
from .gemini_client import GeminiClient, Usage, load_and_prepare_image
from .history import UserHistory
from .prompts import build_stage1_prompt, build_stage2_prompt
from .schema import OUTPUT_COLUMNS, normalize_record, normalize_risk_flags


# Risk tokens that, if present, warrant routing the claim to a human reviewer.
MANUAL_REVIEW_TRIGGERS = (
    "possible_manipulation",
    "non_original_image",
    "claim_mismatch",
    "text_instruction_present",
    "user_history_risk",
)


def image_id_from_path(path: str) -> str:
    return Path(path.strip()).stem


def split_image_paths(image_paths: str) -> List[str]:
    return [p.strip() for p in str(image_paths).split(";") if p.strip()]


class Pipeline:
    def __init__(
        self,
        config,
        client: GeminiClient,
        history: UserHistory,
        evidence: EvidenceRequirements,
        cache: ResponseCache,
        dataset_root: str | Path,
        usage: Usage,
    ):
        self.config = config
        self.client = client
        self.history = history
        self.evidence = evidence
        self.cache = cache
        self.dataset_root = Path(dataset_root)
        self.usage = usage

    # -- image loading -----------------------------------------------------
    def _resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else (self.dataset_root / p)

    def _load_images(self, paths: List[str]):
        cap = self.config.runtime["max_images_per_claim"]
        max_edge = self.config.runtime["image_max_long_edge_px"]
        ids, blobs, missing = [], [], []
        for rel in paths[:cap]:
            fp = self._resolve(rel)
            iid = image_id_from_path(rel)
            if not fp.exists():
                missing.append(iid)
                continue
            try:
                blobs.append(load_and_prepare_image(fp, max_edge))
                ids.append(iid)
            except Exception:
                missing.append(iid)
        self.usage.add_images(len(blobs))
        return ids, blobs, missing

    # -- cached model call -------------------------------------------------
    def _call(self, model: str, prompt: str, blobs: List[bytes], max_out: int) -> dict:
        key = self.cache.make_key(model, prompt, blobs)
        cached = self.cache.get(key)
        if cached is not None:
            self.usage.record_cache_hit()
            return cached
        parsed, _, _ = self.client.generate_json(model, prompt, blobs, max_out)
        self.cache.put(key, parsed)
        return parsed

    # -- the two stages ----------------------------------------------------
    def run_claim(self, row: Dict[str, str]) -> Dict[str, str]:
        user_id = row.get("user_id", "")
        claim_object = str(row.get("claim_object", "")).strip().lower()
        user_claim = row.get("user_claim", "")
        image_paths = row.get("image_paths", "")
        all_ids = [image_id_from_path(p) for p in split_image_paths(image_paths)]
        gen = self.config.generation

        ids, blobs, missing = self._load_images(split_image_paths(image_paths))

        # No usable image files at all -> can't review.
        if not blobs:
            return self._finalize(
                row, all_ids,
                valid_image=False, evidence_met=False,
                evidence_reason="No usable image files were found for this claim.",
                issue_type="unknown", object_part="unknown",
                claim_status="not_enough_information",
                justification="No images available to review.",
                supporting_ids="none", severity="unknown",
                risk_flags=["cropped_or_obstructed"] if missing else [],
                user_id=user_id, claim_object=claim_object,
            )

        # ---- STAGE 1: validity gate (skipped in single_stage mode) ----
        single_stage = self.config.runtime.get("single_stage", False)
        if single_stage:
            # Halve model calls: trust all loaded images and let Stage 2 surface
            # any quality flags itself.
            s1_valid, s1_flags, usable_ids = True, [], ids
            return self._run_stage2(
                row, all_ids, ids, blobs, missing,
                user_id, claim_object, user_claim,
                s1_valid, s1_flags, usable_ids, gen,
            )

        s1_prompt = build_stage1_prompt(claim_object, ids)
        s1 = self._call(self.config.stage1_model, s1_prompt, blobs,
                        gen["stage1_max_output_tokens"])
        s1_valid = bool(s1.get("valid_image", True))
        s1_flags = [str(f) for f in (s1.get("risk_flags") or [])]
        usable_ids = [str(i) for i in (s1.get("usable_image_ids") or ids)] or ids

        short_circuit = self.config.runtime["short_circuit_on_invalid"]
        if short_circuit and not s1_valid:
            return self._finalize(
                row, all_ids,
                valid_image=False, evidence_met=False,
                evidence_reason="Stage-1 gate: " + str(s1.get("reason", "image set unusable")),
                issue_type="unknown", object_part="unknown",
                claim_status="not_enough_information",
                justification="Image set failed quality/validity gate; "
                              + str(s1.get("reason", "")),
                supporting_ids="none", severity="unknown",
                risk_flags=s1_flags + ["manual_review_required"],
                user_id=user_id, claim_object=claim_object,
            )

        return self._run_stage2(
            row, all_ids, ids, blobs, missing,
            user_id, claim_object, user_claim,
            s1_valid, s1_flags, usable_ids, gen,
        )

    # -- STAGE 2: full analysis (shared by both gated and single-stage paths) --
    def _run_stage2(self, row, all_ids, ids, blobs, missing, user_id,
                    claim_object, user_claim, s1_valid, s1_flags, usable_ids, gen):
        rubric = self.evidence.rubric_text(claim_object)
        hist = self.history.summary(user_id)
        s2_prompt = build_stage2_prompt(
            claim_object, user_claim, rubric, hist, usable_ids, s1_flags
        )
        # restrict images sent to Stage 2 to the usable subset (saves tokens)
        usable_blobs = [b for iid, b in zip(ids, blobs) if iid in set(usable_ids)] or blobs
        s2 = self._call(self.config.stage2_model, s2_prompt, usable_blobs,
                        gen["stage2_max_output_tokens"])

        merged_flags = list(dict.fromkeys(s1_flags + [str(f) for f in (s2.get("risk_flags") or [])]))
        if missing:
            merged_flags.append("cropped_or_obstructed")
        # Propagate the history file's own risk tokens verbatim (exact signal).
        for hf in self.history.risk_flags(user_id):
            if hf not in merged_flags:
                merged_flags.append(hf)
        # low-trust signals => ask for a human. Any of these implies a manual
        # review is warranted; user_history_risk is included because in the
        # labeled data it always co-occurs with manual_review_required.
        if any(f in merged_flags for f in MANUAL_REVIEW_TRIGGERS) \
                and "manual_review_required" not in merged_flags:
            merged_flags.append("manual_review_required")

        return self._finalize(
            row, all_ids,
            valid_image=s1_valid and bool(usable_ids),
            evidence_met=bool(s2.get("evidence_standard_met")),
            evidence_reason=s2.get("evidence_standard_met_reason", ""),
            issue_type=s2.get("issue_type", "unknown"),
            object_part=s2.get("object_part", "unknown"),
            claim_status=s2.get("claim_status", "not_enough_information"),
            justification=s2.get("claim_status_justification", ""),
            supporting_ids=s2.get("supporting_image_ids", "none"),
            severity=s2.get("severity", "unknown"),
            risk_flags=merged_flags,
            user_id=user_id, claim_object=claim_object,
        )

    # -- assemble + normalize a final row ----------------------------------
    def _finalize(self, row, all_ids, *, valid_image, evidence_met, evidence_reason,
                  issue_type, object_part, claim_status, justification,
                  supporting_ids, severity, risk_flags, user_id, claim_object) -> Dict[str, str]:
        rec = {
            "user_id": user_id,
            "image_paths": row.get("image_paths", ""),
            "user_claim": row.get("user_claim", ""),
            "claim_object": claim_object,
            "evidence_standard_met": evidence_met,
            "evidence_standard_met_reason": evidence_reason,
            "risk_flags": normalize_risk_flags(risk_flags, claim_object),
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": justification,
            "supporting_image_ids": supporting_ids,
            "valid_image": valid_image,
            "severity": severity,
        }
        normalized = normalize_record(rec, claim_object)
        return {col: normalized.get(col, "") for col in OUTPUT_COLUMNS}

    # -- batch runner ------------------------------------------------------
    def run_batch(self, rows: List[Dict[str, str]], progress=None) -> List[Dict[str, str]]:
        workers = self.config.runtime["max_workers"]
        results: List[Optional[Dict[str, str]]] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.run_claim, r): i for i, r in enumerate(rows)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # never let one row kill the batch
                    results[i] = self._error_row(rows[i], str(e))
                if progress is not None:
                    progress.update(1)
        return [r for r in results if r is not None]

    def _error_row(self, row, msg: str) -> Dict[str, str]:
        all_ids = [image_id_from_path(p) for p in split_image_paths(row.get("image_paths", ""))]
        return self._finalize(
            row, all_ids,
            valid_image=False, evidence_met=False,
            evidence_reason=f"processing_error: {msg[:200]}",
            issue_type="unknown", object_part="unknown",
            claim_status="not_enough_information",
            justification=f"processing_error: {msg[:200]}",
            supporting_ids="none", severity="unknown",
            risk_flags=["manual_review_required"],
            user_id=row.get("user_id", ""),
            claim_object=str(row.get("claim_object", "")).strip().lower(),
        )
