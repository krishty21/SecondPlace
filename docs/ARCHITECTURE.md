# Architecture

## System components

```
┌─────────────────────────── Python (offline) ───────────────────────────┐
│ notebooks/CipherMind_Model_Training_and_Evaluation.ipynb             │
│  1 data validation      7 IsolationForest (normal-only)                │
│  2 feature fit          8 KMeans k=8 + PCA clustering                  │
│  3 model comparison     9 TreeSHAP global importance                   │
│  4 feature ablation    10 official test scored ONCE                    │
│  5 5-fold OOF calib.   11 demo replay + boot sample + SHAP cache       │
│  6 final models        12 registry + operational metrics               │
└───────────────┬────────────────────────────────────────────────────────┘
                │  ml/artifacts/**  (pure JSON, ~134 MB)
                ▼
┌────────────────── soc-engine (Bun, port 3010) ─────────────────────────┐
│ artifacts.ts  loads registry, model dumps, config, caches              │
│ features.ts   FeaturePipeline implementing preprocessor/feature_config│
│ lgbm.ts       LightGBM tree walker (binary + multiclass, Saabas)       │
│ iforest.ts    IsolationForest paper-formula scorer + normalization     │
│ engine.ts     per-event orchestration: A→B→C→D + explanations          │
│ correlate.ts  streaming incident correlation + attack stories          │
│ state.ts      boot: 12,000 stratified test events scored live          │
│ replay.ts     deterministic replay, socket.io ticks + REST fallback    │
│ index.ts      REST router: health/dashboard/incidents/patterns/        │
│               explain/predict/batch/events/replay/metrics/model-info   │
└───────────────┬────────────────────────────────────────────────────────┘
                │  REST + socket.io  (frontend routes via ?XTransformPort=3010)
                ▼
┌────────────────── Next.js 16 (port 3000) ──────────────────────────────┐
│ SOC UI: 5 views + /api/ai/* LLM routes (z-ai-web-dev-sdk, server-side  │
│ only; strict evidence-grounding prompts + deterministic fallback)      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data flow

1. **Train (Python):** 45-column CSVs → `FeatureBuilder` (fit on train only) → 54 features: 39 raw numeric (median-imputed; `log1p` on the 24 columns with |skew| > 3), 12 derived ratios, 3 ordinal categoricals with `__unknown__` fallback. Everything reusable is exported as JSON (`ml/artifacts/metadata/model_registry.json` pins hyperparameters, calibration, threshold, risk config).
2. **Boot (soc-engine):** loads the JSON package, scores a 12,000-event stratified sample of the official test set **live through the real models** (never cached predictions), correlates it into incidents, builds the Command Center dashboard.
3. **Serve:** every API response (dashboard, incidents, patterns, explain, predict, replay ticks) is computed on demand by the same engine; the UI polls `/api/health` every 30 s and degrades gracefully while the engine is down.
4. **Replay:** 903 deterministic test rows with simulated timestamps are pre-scored, then streamed over socket.io in 300 ms ticks (virtual time × speed); a cursor-based REST snapshot endpoint backs up the websocket.

## The TypeScript inference engine

**Design.** `lgbm.ts` loads LightGBM's `dump_model()` JSON and walks each tree exactly like the native predictor (numeric `x[f] <= threshold` splits, categorical decision bit honored; binary probability = `sigmoid(a·logit + b)` with registry Platt coefficients; multiclass = temperature-scaled softmax). `iforest.ts` implements the original IsolationForest paper score

```
s(x) = 2^( −E(h(x)) / c(n) ),   E(h) = mean over trees of ( depth(x) + c(n_leaf − 1) )
```

with `c(n) = 2(ln(n−1) + γ) − 2(n−1)/n`, over serialized sklearn trees (`c_n = 13.0174` for the 1024-row subsample). The raw score is mapped to a deterministic 0–100 scale by piecewise-linear interpolation over **training-normal percentile anchors** stored in `models/isolation_forest.json` (p50=0.3978→15, p90=0.4846→40, p99=0.5908→70, p999=0.6541→90).

**Verification.** `tests/validate_ts_engine.py` sends 13 deterministic, diverse test rows (4 normals + one per ground-truth category) to `POST /api/predict/batch` and compares against Python LightGBM + the same Platt/temperature transform: **all rows match** — calibrated probability identical within 1e-4 (4 decimal places), verdict and predicted category identical on every row.

**Saabas path attributions** (live local explanations for arbitrary events): for each tree, walking root→leaf, each split attributes `child_value − node_value` to the split feature (`internal_value` = expected value at the node). The per-tree deltas telescope, so

```
baseline (Σ root internal_value) + Σ feature contributions == raw model score
```

— enforced by construction in `lgbm.ts` (`walkContribs`), verified against Python, same math as treeinterpreter-style path explanations. See [XAI.md](XAI.md).

## Why artifacts-as-JSON

Models are exported as plain JSON (binary 74.3 MB, multiclass 54.7 MB, IsolationForest 2.7 MB — `metrics/operational.json`), so inference runs **anywhere Bun runs**: no Python, no native LightGBM, no pickles at serving time. The JSON is simultaneously the training/serving interface and the auditable artifact of record.

## Memory & performance

| Measurement | Value | Source |
|---|---|---|
| Boot: 12,000 events through all models | **18.7 s** (~642 events/s) | live `/api/health` engineStats |
| Single-event benchmark (all models + explanation, 100-iter mean) | **1.7 ms** | live `/api/health` engineStats |
| End-to-end `POST /api/predict` (HTTP + JSON) | 3–5 ms | measured on the sandbox |
| Python batch inference (train-side reference) | 2,645.1 ms per 10k events | `metrics/operational.json` |
| Training wall time (final resumable pass) | 146.7 s | `metrics/operational.json` |

Per event the engine allocates one `Float64Array(54)`; trees are walked in place; behavioral vectors used by correlation are cached per event ID (flushed at 20k entries). Replay sessions pre-score their 903 events at start, so tick emission is pure streaming.
