# Evaluation & Operational Analysis

This document covers (1) how the system is evaluated against the labeled sample
set and (2) the operational cost / latency / rate-limit analysis required by the
brief. The pipeline prints real per-run usage to stderr (`run.py` summary), so
the numbers below are a *model* you can replace with measured values once the
dataset is wired in.

---

## 1. How to evaluate

```bash
# Predict on the labeled sample (real model, or add --dry-run for plumbing only)
python run.py --claims dataset/sample_claims.csv --out evaluation/sample_pred.csv

# Score predictions vs. the expected-output columns inside sample_claims.csv
python -m evaluation.evaluate --pred evaluation/sample_pred.csv \
                              --truth dataset/sample_claims.csv
```

`evaluate.py` reports:

- **Exact-match accuracy** for the closed-vocabulary fields: `evidence_standard_met`,
  `issue_type`, `object_part`, `claim_status`, `valid_image`, `severity`.
- **Jaccard overlap** for the multi-value fields `risk_flags` and
  `supporting_image_ids` (order-independent set comparison).
- **`claim_status` macro-F1 + confusion matrix** — the headline metric, since
  supported / contradicted / not_enough_information is the core decision.

Free-text justifications are not exact-matched; spot-check them manually.

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

`run.py` prints actual calls, tokens, images, and cache hits per run:

```
Images processed: 912 | cache hits: 0
  gemini-2.5-flash-lite: 500 calls, 478000 in / 45000 out tokens
  gemini-2.5-flash:      400 calls, 612000 in / 99000 out tokens
```

Plug those into the pricing table (or the `pricing` block in `config.yaml`) for
an exact post-run cost.
