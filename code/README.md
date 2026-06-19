# Multi-Modal Evidence Review — Solution

Verifies damage claims for **cars, laptops, and packages** by reading the claim
conversation, the submitted images, the user's claim history, and the minimum
evidence requirements, then deciding whether the images **support**,
**contradict**, or give **not_enough_information** for the claim.

For the full task spec and I/O schema see [`../problem_statement.md`](../problem_statement.md).

---

## Approach

A **two-stage, object-aware pipeline** over a Gemini vision model:

1. **Stage 1 — validity gate** (cheap model, every claim). One call over the
   whole image set decides whether the images are usable and emits quality /
   authenticity / wrong-object risk flags. If the set is unusable and
   `short_circuit_on_invalid` is on, Stage 2 is skipped — this is where most of
   the savings on junk submissions come from.
2. **Stage 2 — analysis** (stronger model, only on usable sets). One call with
   an **object-specific** prompt (car / laptop / package each get their own part
   and issue vocabulary), the evidence rubric, and the history summary. Produces
   the substantive fields.
3. **Merge + normalize.** Stage 1 + Stage 2 risk flags are unioned with
   deterministic history flags, `manual_review_required` is added for low-trust
   results, and every field is forced into the closed output vocabulary.

**Why deterministic history flags?** The dataset encodes the intended history
signal in the `history_flags` column, so the pipeline propagates
`user_history_risk` / `manual_review_required` from it directly instead of
re-deriving them from raw counts. On the labeled sample this matches the
history-driven flags exactly (20/20). The images remain the primary source of
truth; history only adds risk context and never flips a clear visual call.

---

## Layout

```text
code/
├── main.py                 # entry point: claims.csv -> output.csv
├── config.yaml             # models, pricing, concurrency, cache, retry
├── requirements.txt
├── .env.example            # copy to .env; set GEMINI_API_KEY
├── src/
│   ├── pipeline.py         # two-stage orchestration + batch runner
│   ├── prompts.py          # Stage-1 gate + object-specific Stage-2 prompts
│   ├── gemini_client.py    # Gemini wrapper: image prep, retry, usage metering
│   ├── evidence.py         # evidence_requirements.csv -> prompt rubric
│   ├── history.py          # user_history.csv -> summary + deterministic flags
│   ├── cache.py            # disk cache keyed on (model, prompt, image bytes)
│   ├── schema.py           # output columns + closed-vocabulary normalization
│   └── config.py
└── evaluation/
    ├── main.py             # evaluation entry point: predict on sample + score
    ├── evaluate.py         # scorer (accuracy, Jaccard, claim_status macro-F1)
    └── evaluation_report.md
```

---

## Setup

```bash
cd code
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set GEMINI_API_KEY
export GEMINI_API_KEY=...    # or rely on the .env / your shell
```

Secrets are read from the environment only (`GEMINI_API_KEY`, or
`GOOGLE_API_KEY`). Never hardcode keys.

---

## Run

All commands run from the `code/` directory. Paths default to the repo's
`dataset/` folder, so the common cases need no flags.

```bash
# Produce the submission file -> ../dataset/output.csv  (needs GEMINI_API_KEY)
python main.py

# No-network smoke test of the whole pipeline (stubbed model, no key):
python main.py --claims ../dataset/sample_claims.csv --out /tmp/dry.csv --dry-run
```

Useful flags: `--limit N` (first N rows), `--config path`, `--dataset-root`,
`--out`, `--claims`.

---

## Evaluate

```bash
# Predict on the labeled sample AND score it, in one command:
python evaluation/main.py            # add --dry-run for plumbing only
```

The scorer reports per-field exact-match accuracy, Jaccard overlap for the
multi-value fields, and a `claim_status` macro-F1 + confusion matrix. See
[`evaluation/evaluation_report.md`](evaluation/evaluation_report.md) for the
strategy comparison and the operational (cost / latency / rate-limit) analysis.

---

## Cost / latency (summary)

Default models are `gemini-2.5-flash-lite` (gate) and `gemini-2.5-flash`
(analysis). On a ~500-claim test set this is roughly **$0.50 total (~$0.001 /
claim)** and ~11 minutes wall-clock at `max_workers: 4`. Determinism
(`temperature: 0`) plus the disk cache make re-runs free. Full breakdown and
pricing assumptions are in the evaluation report.

---

## Notes

- Output conforms to the exact column order in `problem_statement.md`; all
  values are normalized to the allowed vocabulary in `src/schema.py`.
- No hardcoded test labels or file-specific answers.
- Tunable knobs (models, concurrency, image downscale, short-circuit, cache,
  retry/backoff) all live in `config.yaml`.
