# Multi-Modal Evidence Review

A two-stage, Gemini-powered pipeline that decides whether submitted images
**support**, **contradict**, or give **not_enough_information** for a damage
claim about a `car`, `laptop`, or `package`.

## Why two stages

1. **Stage 1 — validity gate** (`gemini-2.5-flash-lite`, cheap): one fast pass
   over the whole image set. Is it blurry, the wrong object, low-light,
   obstructed, a screenshot, or carrying injected text instructions? This makes
   `valid_image` / `risk_flags` reliable and **gates the expensive stage** — junk
   submissions never burn Stage-2 tokens.

2. **Stage 2 — full analysis** (`gemini-2.5-flash`, only if Stage 1 passes):
   given the claim, the matching `evidence_requirements` rubric, a user-history
   summary, and the usable images, it produces the substantive fields.
   **Object-specific prompts** matter here — "crack" means something different on
   a windshield, a laptop hinge, and a cardboard box, so each object has its own
   part vocabulary and guidance (`src/prompts.py`). A generic prompt yields mushy
   `object_part` / `issue_type`.

Then a **merge/normalize** step unions Stage-1 + Stage-2 risk flags, adds a
deterministic `user_history_risk` flag, adds `manual_review_required` for
low-trust results, and forces every field into the spec's closed vocabulary
(`src/schema.py`).

History is used as **risk context only** — it never, on its own, flips a
visually obvious decision (per the brief).

## Layout

```
evidence_review/
├── run.py                     # entry point: claims.csv -> output.csv
├── config.yaml                # models, pricing, concurrency, cache, retry
├── requirements.txt
├── .env.example               # GEMINI_API_KEY
├── src/
│   ├── config.py              # config loader
│   ├── schema.py              # output columns + closed-vocab normalization
│   ├── history.py             # user-history summary + risk derivation
│   ├── evidence.py            # evidence-requirements rubric injection
│   ├── prompts.py             # Stage-1 gate + object-specific Stage-2 prompts
│   ├── gemini_client.py       # Gemini wrapper: image prep, retry, usage metering
│   ├── cache.py               # deterministic disk cache
│   └── pipeline.py            # two-stage orchestration + batch runner
└── evaluation/
    ├── evaluate.py            # score vs. labeled sample (accuracy, F1, confusion)
    └── evaluation_report.md   # operational analysis (cost/latency/TPM-RPM)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your key, or:  export GEMINI_API_KEY=...
```

Place the provided data under `dataset/` (or point the flags elsewhere):

```
dataset/
├── claims.csv  sample_claims.csv  user_history.csv  evidence_requirements.csv
└── images/sample/...  images/test/...
```

## Run

```bash
# Smoke-test the plumbing with NO key and NO network (stubbed model):
python run.py --claims dataset/sample_claims.csv --out /tmp/dry.csv --dry-run

# Evaluate on the labeled sample:
python run.py --claims dataset/sample_claims.csv --out evaluation/sample_pred.csv
python -m evaluation.evaluate --pred evaluation/sample_pred.csv \
                              --truth dataset/sample_claims.csv

# Produce final predictions for the test set:
python run.py --claims dataset/claims.csv --out output.csv
```

`output.csv` is written with exactly the 14 required columns, in order.

## Cost / latency at a glance

~**$0.001 per claim** with the default Flash-Lite + Flash split (~$0.50 for a
500-claim set); ~11 min wall-clock at 4 workers. Full breakdown, pricing
assumptions, and TPM/RPM strategy in
[`evaluation/evaluation_report.md`](evaluation/evaluation_report.md). `run.py`
prints actual calls/tokens/images per run.

## Tuning knobs (all in `config.yaml`)

- `models.stage2` → swap to `gemini-2.5-pro` for harder cases.
- `runtime.short_circuit_on_invalid` → trade Stage-2 savings vs. recall.
- `runtime.max_workers` → throughput vs. rate limits.
- `runtime.image_max_long_edge_px` → image-token cost vs. detail.
- `cache.enabled` → free re-runs while iterating.

## Notes / assumptions

- Built against the documented schemas; column names are read from the CSV
  headers, so minor naming differences are tolerated for inputs.
- Gemini pricing in `config.yaml` reflects June 2026 published rates — update if
  Google changes them.
- The closed-vocabulary normalizer guarantees outputs stay within the allowed
  value lists even if the model returns a near-miss token.
```
