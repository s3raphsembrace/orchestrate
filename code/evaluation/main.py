#!/usr/bin/env python3
"""Evaluation entry point (the contract entry point named in AGENTS.md §6).

One command that closes the loop on the labeled sample set:

  1. run the two-stage pipeline over ``dataset/sample_claims.csv``,
  2. write the predictions to ``evaluation/sample_pred.csv``,
  3. score them against the expected-output columns carried in the same file,
  4. print per-field accuracy, set-overlap, and the ``claim_status`` macro-F1.

Run it from the ``code/`` directory:

    # plumbing only, no API key / network needed:
    python evaluation/main.py --dry-run

    # real evaluation (needs GEMINI_API_KEY):
    python evaluation/main.py

    # equivalently, as a module:
    python -m evaluation.main --dry-run

The scorer itself lives in ``evaluation/evaluate.py`` and is reused as-is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make this runnable both as ``python evaluation/main.py`` (script) and as
# ``python -m evaluation.main`` (module) by ensuring the ``code/`` root — which
# holds the ``src`` package — is importable.
CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT.parent / "dataset"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.cache import ResponseCache  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evidence import EvidenceRequirements  # noqa: E402
from src.gemini_client import GeminiClient, Usage  # noqa: E402
from src.history import UserHistory  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.schema import OUTPUT_COLUMNS  # noqa: E402

# Reuse the top-level entry point's IO helpers and the offline stub client so we
# don't duplicate them here.
from main import _StubClient, read_rows, write_rows  # noqa: E402
from evaluation.evaluate import evaluate  # noqa: E402


def predict(args) -> Path:
    """Run the pipeline over the eval claims file and write predictions."""
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
    results = pipeline.run_batch(rows)

    pred_path = Path(args.pred_out)
    write_rows(pred_path, results)

    print(f"Wrote {len(results)} predictions -> {pred_path}", file=sys.stderr)
    print(f"Images processed: {usage.images_processed} | "
          f"cache hits: {usage.cache_hits}", file=sys.stderr)
    for m in sorted(set(usage.calls_by_model)):
        print(f"  {m}: {usage.calls_by_model[m]} calls, "
              f"{usage.input_tokens_by_model.get(m, 0)} in / "
              f"{usage.output_tokens_by_model.get(m, 0)} out tokens",
              file=sys.stderr)
    return pred_path


def score(pred_path: Path, truth_path: Path) -> None:
    rep = evaluate(pred_path, truth_path)
    print(f"\nTruth rows scored: {rep['total_truth_rows']} "
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Predict on the labeled sample set and score the results.")
    ap.add_argument("--claims", default=str(DATA_DIR / "sample_claims.csv"),
                    help="labeled claims file to predict on")
    ap.add_argument("--truth", default=str(DATA_DIR / "sample_claims.csv"),
                    help="file carrying the expected-output columns")
    ap.add_argument("--pred-out", default=str(CODE_ROOT / "evaluation" / "sample_pred.csv"),
                    help="where to write predictions")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("--dataset-root", default=str(DATA_DIR),
                    help="root that image_paths are relative to")
    ap.add_argument("--history", default=str(DATA_DIR / "user_history.csv"))
    ap.add_argument("--evidence", default=str(DATA_DIR / "evidence_requirements.csv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N rows")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a stubbed model: no API key or network needed")
    ap.add_argument("--score-only", action="store_true",
                    help="skip prediction; score an existing --pred-out file")
    args = ap.parse_args(argv)

    pred_path = Path(args.pred_out) if args.score_only else predict(args)
    score(pred_path, Path(args.truth))


if __name__ == "__main__":
    main()
