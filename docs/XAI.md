# Explainability (XAI)

Two levels — **global** (what does the model rely on overall?) and **local** (why this event?) — plus calibrated-probability diagnostics. The Explainability Center view renders all of it; artifacts live under `ml/artifacts/`.

## Global: exact TreeSHAP

`shap.TreeExplainer` (exact TreeSHAP) on the LightGBM binary model over a **2,000-row training sample**; exported to `metrics/shap_global.json` (expected value 3.3045, in log-odds). Mean |SHAP| ranking:

| Rank | Feature | mean \|SHAP\| |
|---:|---|---:|
| 1 | `sttl` (source TTL) | **4.445** |
| 2 | `ct_dst_sport_ltm` | 1.473 |
| 3 | `ct_state_ttl` | 1.065 |
| 4 | `cat_proto` | 0.673 |
| 5 | `size_ratio` | 0.648 |

`sttl` is **3× the runner-up** — source time-to-live dominates the model's global behavior, which matches UNSW-NB15 literature (attack tooling often fixes TTLs). The multiclass model's global importance (LightGBM gain) is exported alongside (`metrics/multiclass_gain.json`) and rendered as a second chart.

## Local: two paths, both real math

**Path 1 — exact TreeSHAP, precomputed (4,403 events).** At training time, LightGBM `pred_contrib=True` computes exact TreeSHAP contributions for all 903 replay events **plus** a 3,500-row attack-heavy sample of the boot set → `explainability/shap_cache.json` (4,403 entries). When the engine explains an event whose test-row index is in the cache, it serves these exact values (baseline = expected value).

**Path 2 — Saabas path attributions, computed live in TypeScript.** For any *arbitrary* event (e.g., analyst-pasted flows via `POST /api/explain`), `lgbm.ts` computes Saabas attributions on the fly. Per tree, walking root→leaf, each split credits `child_internal_value − node_internal_value` to the split feature; per-tree deltas telescope, so:

```
baseline (Σ root internal_value = mean training prediction) + Σ contributions == raw model score
```

This identity holds by construction and was verified against Python — it is the same per-path math as treeinterpreter. The difference vs. TreeSHAP: Saabas is a path-dependent (not Shapley-consistent) approximation, exact at the score level but potentially different in how it splits correlated features' credit.

**The distinction is surfaced, not hidden:** every explanation payload carries a `method` badge — `exact-treeshap` or `saabas-path` — rendered as a chip in the UI (event dialog, Local Explanation Explorer, incident Explainability tab).

**Multiclass explanations:** for the 903 replay events, top-10 contributors for the **predicted class** were precomputed via the multiclass model's `pred_contrib` (cached as `mc` entries alongside the binary ones) and shown with all-10 class probabilities. Live arbitrary events get per-class Saabas via `saabasMulti`.

## Narratives, grounded

Human-readable narratives (e.g., "High attack confidence (96.4%) driven primarily by source TTL, destination ports in last 100 connections, protocol") are generated from **real feature labels only** — a fixed label dictionary over the 54 engineered features (`features.ts::FEATURE_LABELS`) with the event's actual values. No invented hosts, IPs, or threat names; categorical contributions render the actual `proto`/`service`/`state` value.

## What is intentionally *not* claimed

Attributions explain the **model's score**, not the attacker's intent. Correlated features share credit imprecisely under Saabas (use the cached exact TreeSHAP view for those rows); and with 54 features the top-6 positives / top-4 negatives shown in the UI are a ranked slice, not the full decomposition (the full vector is in the API payload).
