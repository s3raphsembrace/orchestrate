#!/usr/bin/env python3
"""Entry point: run the two-stage pipeline over a claims CSV -> output.csv.

Examples
--------
# Full test run (requires GEMINI_API_KEY):
python run.py --claims dataset/claims.csv --out output.csv

# Evaluate on the labeled sample first:
python run.py --claims dataset/sample_claims.csv --out sample_output.csv

# No-network smoke test of the plumbing (stubbed model):
python run.py --claims dataset/sample_claims.csv --out /tmp/dry.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.cache import ResponseCache
from src.config import load_config
from src.evidence import EvidenceRequirements
from src.gemini_client import GeminiClient, Usage
from src.history import UserHistory
from src.pipeline import Pipeline
from src.schema import OUTPUT_COLUMNS

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


def read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in OUTPUT_COLUMNS})


class _StubClient(GeminiClient):
    """Deterministic offline client for --dry-run (no API key, no network)."""

    def generate_json(self, model, prompt, image_bytes, max_output_tokens):
        self.usage.record(model, in_tok=1000, out_tok=120)
        if "validity gate" in prompt or "USABLE" in prompt:
            return ({"valid_image": True, "usable_image_ids": [], "risk_flags": [],
                     "reason": "stub"}, 1000, 120)
        return ({"extracted_claim": "stub", "evidence_standard_met": True,
                 "evidence_standard_met_reason": "stub", "issue_type": "unknown",
                 "object_part": "unknown", "claim_status": "not_enough_information",
                 "claim_status_justification": "stub (dry run)",
                 "supporting_image_ids": [], "severity": "unknown",
                 "risk_flags": []}, 1000, 120)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-Modal Evidence Review pipeline")
    ap.add_argument("--claims", required=True, help="input claims CSV")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("--dataset-root", default="dataset",
                    help="root that image_paths are relative to")
    ap.add_argument("--history", default="dataset/user_history.csv")
    ap.add_argument("--evidence", default="dataset/evidence_requirements.csv")
    ap.add_argument("--limit", type=int, default=None, help="process only first N rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a stubbed model: no API key or network needed")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    usage = Usage()
    cache = ResponseCache(config.cache["enabled"], config.cache["dir"])
    history = UserHistory.load(args.history)
    evidence = EvidenceRequirements.load(args.evidence)
    client = (_StubClient if args.dry_run else GeminiClient)(config, usage)

    pipeline = Pipeline(config, client, history, evidence, cache,
                        args.dataset_root, usage)

    rows = read_rows(Path(args.claims))
    if args.limit:
        rows = rows[: args.limit]

    progress = tqdm(total=len(rows), desc="claims") if tqdm else None
    results = pipeline.run_batch(rows, progress=progress)
    if progress:
        progress.close()

    write_rows(Path(args.out), results)

    # ---- console usage summary (handy for the operational report) ----
    print(f"\nWrote {len(results)} rows -> {args.out}", file=sys.stderr)
    print(f"Images processed: {usage.images_processed} | cache hits: {usage.cache_hits}",
          file=sys.stderr)
    for m in sorted(set(usage.calls_by_model)):
        print(f"  {m}: {usage.calls_by_model[m]} calls, "
              f"{usage.input_tokens_by_model.get(m,0)} in / "
              f"{usage.output_tokens_by_model.get(m,0)} out tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
