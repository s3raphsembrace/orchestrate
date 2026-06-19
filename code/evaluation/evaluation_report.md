# Evaluation & Operational Analysis

This document covers (1) how the system is evaluated against the labeled sample
set and (2) the operational cost / latency / rate-limit analysis required by the
brief. The pipeline prints real per-run usage to stderr, so the operational
numbers below are a *model* you can replace with measured values after a run.

---

## 1. How to evaluate

Run from the `code/` directory. One command predicts on the labeled sample and
scores the result:

```bash
# plumbing only (no API key / network):
python evaluation/main.py --dry-run

# real evaluation (needs GEMINI_API_KEY):
python evaluation/main.py
```

`evaluation/main.py` is the contract entry point (AGENTS.md §6). Under the hood
it runs the pipeline, writes `evaluation/sample_pred.csv`, and calls the scorer
in `evaluation/evaluate.py`. You can also score an existing prediction file
directly:

```bash
python -m evaluation.evaluate --pred evaluation/sample_pred.csv \
                              --truth ../dataset/sample_claims.csv
```

`evaluate.py` reports:

- **Exact-match accuracy** for the closed-vocabulary fields: `evidence_standard_met`,
  `issue_type`, `object_part`, `claim_status`, `valid_image`, `severity`.
- **Jaccard overlap** for the multi-value fields `risk_flags` and
  `supporting_image_ids` (order-independent set comparison).
- **`claim_status` macro-F1 + confusion matrix** — the headline metric, since
  supported / contradicted / not_enough_information is the core decision.

Free-text justifications are not exact-matched; spot-check them manually.

### Sample-set composition (20 labeled rows)

- Objects: car ×8, laptop ×6, package ×6.
- `claim_status`: supported ×12, contradicted ×5, not_enough_information ×3.
- `severity`: medium ×11, low ×3, none ×2, unknown ×3.
- 10/20 rows carry at least one risk flag; `manual_review_required` and
  `user_history_risk` are the most common.

This is small, so treat per-field accuracy as directional and always read the
confusion matrix rather than a single headline number.

### Strategy comparison (required: ≥2 configurations)

| # | Strategy | What changes | Trade-off |
|---|---|---|---|
| A | **Single-call, generic prompt** | One model call per claim; one prompt for all three objects; history risk left to the model. | Cheapest in calls, but `object_part`/`issue_type` blur across objects and quality/junk images still pay full price. Baseline. |
| B | **Two-stage + object-specific prompts (chosen)** | Cheap Stage-1 validity gate short-circuits junk; Stage-2 uses an object-specific part/issue vocabulary; history flags derived deterministically (see below). | Crisper part/issue values and big savings on unusable submissions, at one extra cheap call on valid claims. |
| — | **Stage-2 model swap** | B with `models.stage2 = gemini-2.5-pro` instead of `flash`. | Use only if the sample confusion matrix shows `flash` losing hard supported/contradicted calls; ~4× the Stage-2 cost (see §2.3). |

**Deterministic history flags (key accuracy fix).** The `user_history_risk` and
`manual_review_required` flags are *not* re-derived from raw claim counts — the
dataset already encodes the intended signal in the `history_flags` column. The
pipeline propagates those tokens verbatim and then adds `manual_review_required`
whenever a low-trust visual flag (`claim_mismatch`, `non_original_image`,
`possible_manipulation`, `text_instruction_present`) or `user_history_risk` is
present. On the 20-row sample this reproduces the labeled history-driven flags
**exactly (20/20)**, including the user whose `history_flags` is
`manual_review_required` rather than `user_history_risk` — a case a counts-based
heuristic gets wrong. The visual flags themselves still come from the model.

### Tuning loop
1. Run on the sample, read the confusion matrix.
2. If the gate is too aggressive (real claims landing in `not_enough_information`),
   loosen `short_circuit_on_invalid` or the Stage-1 validity bar.
3. If `object_part` / `issue_type` are noisy, edit the per-object guidance in
   `src/prompts.py` (the object-specific section), not a generic prompt.
4. If hard claims are misjudged, switch `models.stage2` to `gemini-2.5-pro` and
   re-measure cost vs. accuracy.

---

## 2. Operational analysis

### 2.1 Call structure (where the calls go)

| Stage | Model (default) | Calls | When |
|---|---|---|---|
| Stage 1 — validity gate | `gemini-2.5-flash-lite` | **1 per claim** (all images in one call) | every claim |
| Stage 2 — full analysis | `gemini-2.5-flash` | **1 per claim that passes the gate** | only valid image sets |

Design choices that bound call volume:
- **One Stage-1 call per claim** over the whole image set (not one per image) —
  cheaper and lets the model compare images and pick the usable subset.
- **Short-circuit on invalid** — blurry / wrong-object / junk submissions never
  reach the expensive Stage-2 model. On real claim traffic a meaningful share of
  submissions are unusable, so this is the main saver.
- **Deterministic + cached** (`temperature=0`, disk cache keyed on
  model+prompt+image bytes) — re-runs and retries cost nothing, and you can
  iterate on merge/normalize logic without re-billing vision calls.
- **Usable-subset images to Stage 2** — only images Stage 1 marked usable are
  re-sent, trimming image tokens.

### 2.2 Token model (per call, illustrative)

Image token cost assumed at **~258 tokens/image** (Gemini bills a small/standard
image around this; large images are tiled — we downscale to a 1024px long edge
to stay near one tile). Override `pricing.tokens_per_image` if you measure
otherwise.

| | input tokens | output tokens |
|---|---|---|
| Stage 1 (prompt ~450 + ~1.8 imgs) | ~950 | ~90 |
| Stage 2 (prompt ~900 + rubric/history ~250 + ~1.5 imgs) | ~1,540 | ~250 |

### 2.3 Worked cost example — 500-claim test set

Assumptions: **N = 500** claims, **1.8 images/claim avg**, **80% pass the gate**
(→ 400 Stage-2 calls). Pricing per 1M tokens (June 2026 defaults, in `config.yaml`):
Flash-Lite **$0.10 in / $0.40 out**, Flash **$0.30 in / $2.50 out**.

| Item | Count | Cost |
|---|---|---|
| Stage-1 calls (Flash-Lite) | 500 | 500 × (950·$0.10 + 90·$0.40)/1e6 ≈ **$0.066** |
| Stage-2 calls (Flash) | 400 | 400 × (1,540·$0.30 + 250·$2.50)/1e6 ≈ **$0.435** |
| **Total** | | **≈ $0.50** (~$0.001 / claim) |
| Images processed | ~900 | |

Sensitivity: routing Stage 2 to **`gemini-2.5-pro`** ($1.25 in / $10 out) raises
the Stage-2 cost to ≈ **$2.0** (total ≈ **$2.1**, ~$0.004/claim) — still cheap,
trade it in only if the sample-set accuracy needs it. Using the Gemini **Batch
API** (~50% off, 24h async) roughly halves either figure for the offline test run.

### 2.4 Latency / runtime

- Stage 1 ≈ 1–2 s; Stage 2 ≈ 2–4 s. Per claim ≈ **5–6 s sequential**.
- With `max_workers: 4`, 500 claims ≈ 500 × 5.5 / 4 ≈ **~11 minutes** wall clock.
- Raise `max_workers` to shorten this, subject to the rate limits below.

### 2.5 TPM / RPM and throttling strategy

- **RPM**: at 4 workers and ~5 s/claim the pipeline issues well under ~100
  req/min combined — comfortably inside Gemini paid-tier limits for Flash /
  Flash-Lite. If you hit a free-tier or low-tier RPM cap, lower `max_workers`.
- **TPM**: ~1.5–2k tokens/call × ~100 calls/min ≈ ~200k TPM — also within paid
  Flash limits. Large image sets are the main TPM driver; the 1024px downscale
  and the per-claim (not per-image) Stage-1 call keep this in check.
- **Retry/backoff**: `tenacity` exponential backoff (2 s → 60 s, 5 attempts)
  absorbs `429`/`503`; deterministic caching means a retried call that already
  succeeded is served from cache.
- **Batching/caching**: disk cache eliminates duplicate work across runs; for a
  one-shot offline test set the **Batch API** is the recommended throughput/cost
  lever.

> Limits change across tiers and over time — confirm current Gemini RPM/TPM for
> your project tier before a large run, and let the backoff handle the rest.

### 2.6 Reproducing the real numbers

Both entry points (`main.py` and `evaluation/main.py`) print actual calls,
tokens, images, and cache hits to stderr per run:

```
Images processed: 912 | cache hits: 0
  gemini-2.5-flash-lite: 500 calls, 478000 in / 45000 out tokens
  gemini-2.5-flash:      400 calls, 612000 in / 99000 out tokens
```

Plug those into the pricing table (or the `pricing` block in `config.yaml`) for
an exact post-run cost.
