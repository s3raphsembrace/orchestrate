#!/usr/bin/env python3
"""Score predictions against the labeled sample set.

Usage
-----
Most users should just run ``python evaluation/main.py`` (predict + score in one
step). This module is the scorer underneath, and can also be run standalone:

# 1) produce predictions on the sample (real or --dry-run), from code/:
python main.py --claims ../dataset/sample_claims.csv --out evaluation/sample_pred.csv
# 2) score them against the ground-truth columns in sample_claims.csv:
python -m evaluation.evaluate --pred evaluation/sample_pred.csv \
                              --truth ../dataset/sample_claims.csv

Reports per-field accuracy plus a confusion matrix and macro-F1 for the most
important field, `claim_status`. The truth file is `sample_claims.csv`, which
carries both the input columns and the expected-output columns.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.schema import normalize_risk_flags

# Fields we score. Free-text justifications are not exact-matched.
EXACT_FIELDS = [
    "evidence_standard_met", "issue_type", "object_part", "claim_status",
    "valid_image", "severity",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
KEY_FIELDS = ["user_id", "image_paths", "user_claim", "claim_object"]


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _key(row: Dict[str, str]) -> str:
    # Match prediction to truth on the stable identifying columns.
    return "|".join(str(row.get(k, "")).strip() for k in KEY_FIELDS)


def _norm(v: str) -> str:
    return str(v).strip().lower()


def _set(v: str) -> set:
    return {x.strip().lower() for x in str(v).split(";") if x.strip() and x.strip().lower() != "none"}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate(pred_path: Path, truth_path: Path) -> Dict:
    preds = {_key(r): r for r in _read(pred_path)}
    truth = _read(truth_path)

    exact_hits = defaultdict(int)
    set_scores = defaultdict(float)
    total = 0
    missing = 0
    cm = defaultdict(Counter)            # confusion matrix for claim_status
    status_labels = ["supported", "contradicted", "not_enough_information"]

    for t in truth:
        # only score rows that actually carry ground-truth output columns
        if not any(t.get(f) for f in EXACT_FIELDS):
            continue
        total += 1
        p = preds.get(_key(t))
        if p is None:
            missing += 1
            continue
        for f in EXACT_FIELDS:
            if _norm(p.get(f)) == _norm(t.get(f)):
                exact_hits[f] += 1
        for f in SET_FIELDS:
            set_scores[f] += _jaccard(_set(p.get(f)), _set(t.get(f)))
        cm[_norm(t.get("claim_status"))][_norm(p.get("claim_status"))] += 1

    scored = max(total - missing, 1)
    report = {
        "total_truth_rows": total,
        "missing_predictions": missing,
        "exact_field_accuracy": {f: exact_hits[f] / scored for f in EXACT_FIELDS},
        "set_field_jaccard": {f: set_scores[f] / scored for f in SET_FIELDS},
        "claim_status_confusion": {a: dict(cm[a]) for a in status_labels},
        "claim_status_macro_f1": _macro_f1(cm, status_labels),
    }
    return report


def _macro_f1(cm, labels) -> float:
    f1s = []
    for lab in labels:
        tp = cm[lab][lab]
        fp = sum(cm[o][lab] for o in labels if o != lab)
        fn = sum(cm[lab][o] for o in labels if o != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", default="dataset/sample_claims.csv")
    args = ap.parse_args(argv)

    rep = evaluate(Path(args.pred), Path(args.truth))
    print(f"Truth rows scored: {rep['total_truth_rows']} "
          f"(missing predictions: {rep['missing_predictions']})\n")
    print("Exact-match accuracy:")
    for f, v in rep["exact_field_accuracy"].items():
        print(f"  {f:24s} {v:6.1%}")
    print("\nSet-overlap (Jaccard):")
    for f, v in rep["set_field_jaccard"].items():
        print(f"  {f:24s} {v:6.3f}")
    print(f"\nclaim_status macro-F1: {rep['claim_status_macro_f1']:.3f}")
    print("claim_status confusion (truth -> pred):")
    for a, d in rep["claim_status_confusion"].items():
        print(f"  {a:24s} {d}")


if __name__ == "__main__":
    main()
