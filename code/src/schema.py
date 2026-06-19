"""Output schema, allowed values, and normalization.

The grader cares about a fixed column order and a closed vocabulary of values.
Everything the model returns is passed through `normalize_record` so a stray
or out-of-vocab value never reaches output.csv.
"""
from __future__ import annotations

from typing import Dict, List

# Exact output column order required by the spec.
OUTPUT_COLUMNS: List[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PARTS: Dict[str, set] = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}


def _coerce_bool(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def _clean_enum(value, allowed: set, fallback: str) -> str:
    v = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    return v if v in allowed else fallback


def normalize_risk_flags(value, object_type: str) -> str:
    """Accept list or semicolon string, drop unknowns, dedupe, default 'none'."""
    if value is None:
        return "none"
    items = value if isinstance(value, list) else str(value).split(";")
    seen: List[str] = []
    for item in items:
        flag = str(item).strip().lower().replace(" ", "_")
        if flag in RISK_FLAGS and flag != "none" and flag not in seen:
            seen.append(flag)
    return ";".join(seen) if seen else "none"


def normalize_supporting_ids(value) -> str:
    if value is None:
        return "none"
    items = value if isinstance(value, list) else str(value).split(";")
    cleaned = [str(i).strip() for i in items if str(i).strip() and str(i).strip().lower() != "none"]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def normalize_record(rec: Dict, object_type: str) -> Dict:
    """Force every model-produced field into the closed vocabulary."""
    parts = OBJECT_PARTS.get(object_type, {"unknown"})
    out = dict(rec)
    out["evidence_standard_met"] = _coerce_bool(rec.get("evidence_standard_met"))
    out["valid_image"] = _coerce_bool(rec.get("valid_image"))
    out["issue_type"] = _clean_enum(rec.get("issue_type"), ISSUE_TYPE, "unknown")
    out["object_part"] = _clean_enum(rec.get("object_part"), parts, "unknown")
    out["claim_status"] = _clean_enum(
        rec.get("claim_status"), CLAIM_STATUS, "not_enough_information"
    )
    out["severity"] = _clean_enum(rec.get("severity"), SEVERITY, "unknown")
    out["risk_flags"] = normalize_risk_flags(rec.get("risk_flags"), object_type)
    out["supporting_image_ids"] = normalize_supporting_ids(rec.get("supporting_image_ids"))
    out["evidence_standard_met_reason"] = str(
        rec.get("evidence_standard_met_reason", "")
    ).strip()[:500]
    out["claim_status_justification"] = str(
        rec.get("claim_status_justification", "")
    ).strip()[:500]
    return out
