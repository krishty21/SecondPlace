# CipherMind Sentinel — AI Security Operations Copilot

**UNSW-NB15 · CipherMind AI '26 · v1.0.0**

CipherMind Sentinel turns a flood of raw network-flow alerts into a small
number of **risk-scored, explained, correlated incidents**, powered by a
layered ML detection architecture (calibrated binary detector → multiclass
attack classifier → unsupervised anomaly detector → transparent risk score)
with a live SOC interface: real-time replay streaming, incident timelines with
epistemics labels, behavioral pattern exploration, an explainability center
(SHAP + calibration), and an LLM-backed AI analyst with a deterministic
fallback.
> **Everything in this repository runs from the shipped, pre-trained model
> artifacts. Nothing retrains at runtime.**
---
## Table of contents
1. [Final metrics (verified)](#final-metrics-verified)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Quick start — A. Run the trained application](#quick-start--a-run-the-trained-application)
6. [B. Retrain the models from the notebook](#b-retrain-the-models-from-the-notebook)
7. [Using the application (demo walkthrough)](#using-the-application-demo-walkthrough)
8. [API usage](#api-usage)
9. [Model artifacts & locations](#model-artifacts--locations)
10. [Dataset](#dataset)
11. [ML approach & methodology](#ml-approach--methodology)
12. [Docker](#docker)
13. [Troubleshooting](#troubleshooting)
14. [Limitations](#limitations)
15. [Security disclaimer](#security-disclaimer)
16. [Documentation index](#documentation-index)

---

## Final metrics (verified)

Official UNSW-NB15 test set (82,332 rows, scored **exactly once** after model
freeze; reproduced end-to-end by the master notebook in *verify* mode with
**max |Δ| = 0.0** against the shipped artifacts):

| Model | Metric | Value |
|---|---|---|
| **A — Binary attack detector** (LightGBM, Platt-calibrated, thr 0.46) | Accuracy / Precision / Recall | 0.8663 / 0.8124 / 0.9844 |
| | **F1** | **0.8902** |
| | **ROC-AUC / PR-AUC** | **0.9826 / 0.9873** |
| | Specificity / FPR | 0.0156 / 27.85% |
| **B — Multiclass classifier** (LightGBM, balanced, temperature-scaled) | Accuracy | 0.7711 |
| | **macro-F1 / weighted-F1** | **0.5197 / 0.7856** |
| | Balanced accuracy | 0.5585 |
| **C — Anomaly detector** (Isolation Forest, trained on normal traffic only) | **ROC-AUC** | **0.7963** |
| | **precision@1000** | **0.981** |

Model selection comparison (train-internal validation): **LightGBM 0.9736 F1 /
0.9947 AUC** > XGBoost 0.9722 > RandomForest 0.9715 > LogisticRegression
0.9547. Full metric sets, per-class tables, confusion matrices, calibration
curves and error analyses are produced live by the master notebook
(`notebooks/CipherMind_Model_Training_and_Evaluation.ipynb`) and documented in
`docs/EVALUATION.md` and `docs/MODEL_CARD.md`.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │  MASTER NOTEBOOK (training time, offline)           │
                    │  notebooks/CipherMind_Model_Training_and_Evaluation │
                    │  data → validation → features → comparison →        │
                    │  training → calibration → evaluation → SHAP         │
                    └───────────────────────┬─────────────────────────────┘
                                            │ exports (seed 42, deterministic)
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  MODEL ARTIFACTS  ml/artifacts/** (portable JSON)   │
                    │  models/ preprocessor/ explainability/ replay/      │
                    │  metrics/ reports/ metadata/                        │
                    └───────────────────────┬─────────────────────────────┘
                                            │ loaded at boot (never retrained)
                                            ▼
┌──────────────┐   REST + socket.io   ┌─────────────────────────────────────┐
│  SOC UI      │◄────────────────────►│  SOC ENGINE (TypeScript/Bun, :3010) │
│  Next.js 16  │  /api/*              │  • LightGBM tree walker (exact)    │
│  React 19    │                      │  • IsolationForest scorer (paper)  │
│  shadcn/ui   │                      │  • risk scoring + severity         │
│  5 views     │                      │  • incident correlation engine     │
│  :3000       │                      │  • deterministic replay streamer   │
│  + /api/ai/* │                      │  • Saabas local attributions       │
│  (LLM routes)│                      └─────────────────────────────────────┘
└──────────────┘
```

**Layered detection models** (all four contribute to every scored event):

| Layer | Question | Algorithm | Calibration |
|---|---|---|---|
| **A** | Is this flow an attack? | LightGBM binary (864 rounds) | Platt (OOF-fitted), F1-optimal threshold 0.46 |
| **B** | Which attack category? | LightGBM multiclass (102×10 trees, class-balanced) | Temperature scaling T=1.0035 |
| **C** | Is the behaviour unusual? | Isolation Forest (120 trees, **normal-only** training) | percentile-anchor normalization 0–100 |
| **D** | How urgent is it? | Transparent weighted risk formula (32/18/20/8/10/12) | none — auditable by design |

The TypeScript engine reproduces the Python models **exactly** — cross-checked
row-by-row by `tests/validate_ts_engine.py` (probabilities match ≤ 1e-4, all
verdicts/categories match; run it any time the engine is up).

The two AI endpoints (`/api/ai/incident-summary`, `/api/ai/analyst-chat`) are
implemented server-side (Next.js route handlers) with the `z-ai-web-dev-sdk`.
If no LLM endpoint is configured/reachable they degrade **gracefully to a
deterministic evidence-grounded fallback** — the UI badges the source
(`llm` vs `fallback`) either way.

---

## Project structure

```
CipherMind_Sentinel/
├── src/                        # Next.js 16 frontend + LLM API routes (app router)
│   ├── app/                    #   page.tsx (SOC UI), api/ai/* (LLM endpoints)
│   ├── components/soc/         #   5 views + shared SOC component library
│   └── lib/                    #   typed API client, formatters, tokens
├── mini-services/soc-engine/   # TypeScript ML inference engine (Bun, port 3010)
│   └── src/                    #   lgbm.ts, iforest.ts, features.ts, engine.ts,
│                               #   correlate.ts, replay.ts, state.ts, index.ts
├── notebooks/
│   └── CipherMind_Model_Training_and_Evaluation.ipynb   # MASTER ML NOTEBOOK
├── ml/artifacts/               # ★ shipped, pre-trained model artifacts
│   ├── models/                 #   binary/multiclass LightGBM (.txt + .json), isolation forest
│   ├── preprocessor/           #   feature_config.json (pipeline contract), clustering.json
│   ├── explainability/         #   shap_cache.json (exact local attributions)
│   ├── replay/                 #   demo_sequence.json (903 events), boot_sample.json
│   ├── metrics/                #   comparison, calibration, test_evaluation, SHAP, ...
│   ├── reports/                #   eval_summary.json, notebook_verification.json
│   └── metadata/               #   model_registry.json (versions, params, risk config)
├── dataset/                    # UNSW-NB15 CSVs (training + testing + feature spec)
├── docs/                       # 10-file documentation suite (see index below)
├── tests/
│   └── validate_ts_engine.py   # Python-vs-TypeScript inference parity check
├── scripts/                    # start-all / stop-all (sh + bat), sandbox helpers
├── requirements.txt            # pinned Python ML environment
├── .env.example                # environment template (NEXT_PUBLIC_ENGINE_URL)
└── README.md
```

---

## Prerequisites

| Requirement | Version used / tested |
|---|---|
| **Bun** (runs frontend + engine) | 1.3.x — install from <https://bun.sh> |
| **Python** (only for the notebook / parity test) | 3.12 (3.10+ should work) |
| OS | Linux, macOS, or Windows (start scripts for both included) |
| RAM | ≥ 2 GB free for the stack; ≥ 4 GB for notebook training mode |

No database is required — all model state lives in `ml/artifacts/**`.

---

## Quick start — A. Run the trained application

> **Path A uses the shipped, pre-trained artifacts. It does NOT retrain
> anything and does NOT require Python.** Path B (retraining) is a separate,
> optional flow below.

### Linux / macOS

```bash
# 1. unzip and enter the project
unzip CipherMind_Sentinel_Final.zip && cd CipherMind_Sentinel

# 2. start everything (installs deps on first run, creates .env)
chmod +x scripts/*.sh
./scripts/start-all.sh

# 3. open the SOC UI
open http://localhost:3000        # macOS   (xdg-open on Linux)
```

### Windows

```bat
:: 1. unzip and enter the project
cd CipherMind_Sentinel

:: 2. start everything (installs deps on first run, creates .env)
scripts\start-all.bat

:: 3. open the SOC UI
start http://localhost:3000
```

### Manual start (two terminals — any OS)

```bash
# terminal 1 — ML inference engine (port 3010)
cd mini-services/soc-engine
bun install                       # first run only
bun run dev

# terminal 2 — frontend (port 3000)
cd <project root>
cp .env.example .env              # first run only (Windows: copy .env.example .env)
bun install                       # first run only
bun run dev
```

Then open **http://localhost:3000**. The engine takes ~20–60 s to boot the
first time (it scores 12,000 real test events through the models to build the
Command Center state); the UI shows an "Engine connecting…" banner and
auto-recovers when it is ready.

**Stop:** `./scripts/stop-all.sh` (Windows: `scripts\stop-all.bat`), or `Ctrl+C`
in each terminal.

---

## B. Retrain the models from the notebook

> **Path B is optional.** The shipped artifacts were produced by exactly this
> notebook (`MODE = "train"`) and are verified by it (`MODE = "verify"`).

```bash
# 1. install the Python environment
pip install -r requirements.txt

# 2. make sure the dataset is in place (the ZIP ships it):
#    dataset/Training and Testing Sets/UNSW_NB15_training-set.csv   (175,341 rows)
#    dataset/Training and Testing Sets/UNSW_NB15_testing-set.csv    (82,332 rows)

# 3. open the master notebook
jupyter notebook notebooks/CipherMind_Model_Training_and_Evaluation.ipynb
```

The notebook has **two documented modes** (single config cell at the top):

| Mode | What it does | Runtime |
|---|---|---|
| `MODE = "verify"` *(shipped default)* | Loads the shipped `.txt` models, re-fits the preprocessing pipeline on the training CSV, re-runs the complete official-test evaluation, and **asserts every metric reproduces the shipped artifacts** (max \|Δ\| must be 0). Writes only `ml/artifacts/reports/notebook_verification.json`. | ~2–5 min |
| `MODE = "train"` | Full pipeline: dataset validation → EDA → features → 4-family model comparison → 5-fold OOF calibration → final training → one-shot test evaluation → SHAP → error/rare-class analysis → **exports the complete artifact set to `ml/artifacts/**`**. Deterministic (seed 42, `deterministic=True` LightGBM). | ~30–60 min (2 cores) |

After retraining, restart the engine to load the new artifacts:

```bash
./scripts/stop-all.sh && ./scripts/start-all.sh        # or restart the engine terminal
python3 tests/validate_ts_engine.py                   # optional: re-verify TS parity
```

The notebook is fully self-contained: no deleted scripts, no hidden variables,
no machine-specific paths (it locates the repo root automatically), and every
stage explains *what* it does and *why*.

---

## Using the application (demo walkthrough)

1. **Command Center** — boot state built from 12,000 real test events scored
   live by the engine: KPI cards, threat timeline, category breakdown,
   prioritized incident list, recent critical alerts (click any event for the
   full 45-field flow + attributions). Note the *"About this data"* card and
   the SIM badges — simulated metadata is always labelled.
2. **Live Replay** — press **Start Replay**: 903 real test events stream
   through the models via socket.io in escalating waves (background noise →
   reconnaissance → exploit → DoS campaigns). Watch incidents form in
   real time; use 2x/4x/8x speed; click an incident → *Open full
   investigation*. (If the socket can't connect, the UI automatically falls
   back to REST polling — the stream never silently dies.)
3. **Incident Investigation** — pick an incident (from replay or boot state):
   risk gauge, ground-truth mix (explicitly marked *not seen by the model*),
   attack story timeline with **Observed / Inferred / Prescriptive** epistemics
   labels, containment playbook, evidence table, explainability, related
   behavioral patterns, and the **AI Analyst** tab (LLM summary + copilot
   chat, badged `llm` or `fallback`).
4. **Pattern Explorer** — KMeans behavioral clusters (traffic-behaviour
   groups, *not* malware families) projected with PCA; click cards to
   highlight clusters; live alert counts from the boot state.
5. **Explainability Center** — global TreeSHAP top-20, multiclass gain,
   OOF reliability diagram + threshold curve, a **local explanation explorer**
   (pick any category/event → exact attributions + class probabilities), and
   the model-comparison tables incl. the selection rationale.

A 13-step guided demo script is in `docs/ARCHITECTURE.md` §Demo walkthrough.

---

## API usage

The engine (port 3010) exposes a JSON REST API + socket.io (full contract:
`docs/API_CONTRACT.md`). Quick examples (from the machine running the stack):

```bash
# health + engine stats
curl http://localhost:3010/api/health

# score a single raw UNSW-NB15 flow row (all 45 fields)
curl -X POST http://localhost:3010/api/predict \
     -H "Content-Type: application/json" \
     -d '{"event": {"dur":0.000011,"proto":"udp","service":"-","state":"INT", ...,"label":0,"attack_cat":"Normal"}}'

# batch score up to 5000 events
curl -X POST http://localhost:3010/api/predict/batch -H "Content-Type: application/json" \
     -d '{"events":[ {...}, {...} ]}'

# explain a single event (returns attributions + class probabilities + narrative)
curl -X POST http://localhost:3010/api/explain -H "Content-Type: application/json" -d '{"event": {...}}'

# start a replay session (then stream via socket.io or REST polling)
curl -X POST http://localhost:3010/api/replay/start -H "Content-Type: application/json" -d '{"speed":4}'
curl http://localhost:3010/api/replay/<id>/state?cursor=0
```

Other endpoints: `/api/dashboard`, `/api/incidents[/:id]`, `/api/patterns`,
`/api/explain/global`, `/api/model/info`, `/api/events?category=Exploits`,
`/api/metrics`. The LLM endpoints live same-origin on the frontend:
`POST /api/ai/incident-summary`, `POST /api/ai/analyst-chat`.

**Socket.io** (used by Live Replay): connect to the engine root and join a
replay — see `src/components/soc/views/live-replay.tsx` for the reference
client (`replay:join` / `replay:control` events, tick streaming, REST
fallback).

---

## Model artifacts & locations

| Purpose | Path |
|---|---|
| Binary detector (native + engine JSON) | `ml/artifacts/models/binary_lightgbm.txt` / `.json` |
| Multiclass classifier | `ml/artifacts/models/multiclass_lightgbm.txt` / `.json` |
| Isolation Forest (portable serialization) | `ml/artifacts/models/isolation_forest.json` |
| **Preprocessing contract** (medians, log1p cols, encoders, feature order) | `ml/artifacts/preprocessor/feature_config.json` |
| Behavioral clustering (KMeans+PCA) | `ml/artifacts/preprocessor/clustering.json` |
| Exact local SHAP cache (demo/boot events) | `ml/artifacts/explainability/shap_cache.json` |
| Deterministic demo replay (903 events) | `ml/artifacts/replay/demo_sequence.json` |
| Boot sample (12k stratified test rows) | `ml/artifacts/replay/boot_sample.json` |
| Calibration (Platt/T/threshold/curves) | `ml/artifacts/metrics/calibration.json` |
| One-shot official test evaluation | `ml/artifacts/metrics/test_evaluation.json` |
| Model comparison + ablation | `ml/artifacts/metrics/model_comparison.json`, `feature_ablation.json` |
| Global SHAP / multiclass gain | `ml/artifacts/metrics/shap_global.json`, `multiclass_gain.json` |
| **Model registry** (params, threshold, risk config, versions) | `ml/artifacts/metadata/model_registry.json` |
| Human/machine summary of the release | `ml/artifacts/reports/eval_summary.json` |
| Notebook verify-run report | `ml/artifacts/reports/notebook_verification.json` |

Every artifact is plain JSON (no pickle) and is loaded by the engine in a
fresh process at boot — verify any time with
`curl localhost:3010/api/health`.

---

## Dataset

**UNSW-NB15** (Moustafa & Slay, 2015) as supplied with the challenge:

| File | Rows | Role |
|---|---|---|
| `dataset/Training and Testing Sets/UNSW_NB15_training-set.csv` | 175,341 | model development |
| `dataset/Training and Testing Sets/UNSW_NB15_testing-set.csv` | 82,332 | one-shot held-out evaluation **and** runtime event source |
| `dataset/NUSW-NB15_features.csv` | 49 | feature spec (reference) |

Honest dataset facts (validated in notebook §04 and `docs/DATASET.md`):

* **No** source/destination IPs, ports, user ids, device ids, or timestamps
  exist in these CSVs. **None are fabricated** — replay timestamps and entity
  labels in the UI are clearly marked `SIM` / *simulated*.
* 45 columns; `id` is a per-file row number (82,332 collisions across files)
  and is excluded from features; label/category consistency verified (0
  mismatches).
* Attack prevalence shifts 68.1% (train) → 55.1% (test); 10 categories with a
  ~330:1 imbalance (`Worms` has 174 training rows).
* This dataset does **not** support true malware-family classification or
  phishing-email classification; no such claims are made.

---

## ML approach & methodology

The complete methodology — with rationale, plots and per-stage code — is the
**master notebook** (`notebooks/CipherMind_Model_Training_and_Evaluation.ipynb`,
26 sections). Summary of the approach:

1. **Leakage control**: preprocessors fitted on train only; model selection on
   a stratified train-internal validation split; calibration/threshold on 5-fold
   out-of-fold train predictions; official test scored exactly once.
2. **Features**: 39 raw numeric + 12 engineered behavioural ratios + 3 encoded
   categoricals (54 total); skew-aware `log1p`; `inf → train-median`
   imputation; unknown-category fallback bucket.
3. **Model selection**: LightGBM vs XGBoost vs RandomForest vs LogisticRegression
   on identical features — LightGBM wins (F1 0.9736 vs 0.9722/0.9715/0.9547).
4. **Calibration**: Platt scaling (binary) and temperature scaling
   (multiclass) fitted on OOF predictions; F1-optimal threshold 0.46.
5. **Anomaly layer**: Isolation Forest trained exclusively on normal traffic;
   paper-formula scoring serialized to portable JSON.
6. **Explainability**: exact TreeSHAP global importance; exact per-event
   attributions cached for demo events; Saabas live attributions with the
   verified identity baseline + Σcontribs = raw score.
7. **Risk scoring**: transparent weighted formula (attack confidence 32,
   anomaly 18, category severity 20, rarity 8, uncertainty 10, correlation 12)
   — every term auditable in `model_registry.json`.
8. **Portability**: all artifacts are JSON; the TypeScript engine re-implements
   inference exactly and is verified against Python (`tests/validate_ts_engine.py`).

---

## Docker

No Docker files ship with this release — the supported local path is the
two-process Bun stack above (the engine and frontend are both plain `bun run
dev` commands, so containerising them is straightforward if you prefer:
one container per service, engine on 3010, frontend with
`NEXT_PUBLIC_ENGINE_URL=http://engine:3010`). Docker support is a natural
next step, not a verified feature of this release.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| *"Engine connecting…" banner never clears | Engine boot takes 20–60 s (12k live-scored boot events). Check `mini-services/soc-engine/engine.log`; verify `curl localhost:3010/api/health`. |
| Frontend shows `ENGINE OFFLINE` on localhost:3000 | `.env` is missing `NEXT_PUBLIC_ENGINE_URL=http://localhost:3010` — create it from `.env.example` and restart `bun run dev` (Next inlines public env at start). |
| Port already in use | Stop the old stack (`scripts/stop-all.sh`) or free ports 3000/3010. |
| `bun: command not found` | Install Bun: <https://bun.sh> (Windows PowerShell: `powershell -c "irm bun.sh/install.ps1 | iex"`). |
| Replay "Could not start replay" | Engine still booting — wait for `/api/health` then press Start Replay again. |
| AI Analyst says `fallback` | No LLM endpoint reachable — expected outside the hosted environment; the deterministic fallback provides the full structured analysis. |
| Notebook `FileNotFoundError: dataset/...` | Run the notebook from the repo (e.g. `notebooks/`) so it can locate `dataset/`; both CSVs must exist for verify/train modes. |
| Notebook verify mode assertion failed | The shipped artifacts were modified — re-run with `MODE="train"` to regenerate, or re-extract the ZIP. |
| `python3 tests/validate_ts_engine.py` mismatches | The engine is running stale artifacts — restart the engine and re-run. |

---

## Limitations

Full detail in `docs/LIMITATIONS.md`. The headline facts:

* **Binary FPR 27.85%** at the F1-optimal threshold — a deliberate
  recall-heavy operating point for triage-with-correlation; do not deploy the
  threshold unchanged in a low-noise production SOC.
* **macro-F1 0.5197** — rare categories (`Worms`, `Shellcode`, `Analysis`,
  `Backdoor`) remain weak; DoS→Exploits is the dominant confusion (~2.5k
  events).
* Anomaly ROC-AUC 0.796 — useful for ranking/triage, not for standalone
  classification.
* The dataset contains **no real IPs/ports/users/timestamps**; replay metadata
  is simulated and labelled as such everywhere.
* Research prototype: model outputs must not drive automated containment in
  production without human review.

---

## Security disclaimer

This system is a **research/education prototype** for the CipherMind AI '26
challenge. It is not certified for production incident response. Model outputs
are probabilistic and miscalibration, drift and adversarial evasion remain
possible. The replay timestamps and entity identifiers are synthetic demo
metadata, not evidence of real network activity. Always corroborate model
output with independent telemetry before acting on it.

---

## Documentation index

| Document | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Component/dataflow deep-dive, TS inference engine internals, performance |
| `docs/API_CONTRACT.md` | Full REST + socket.io contract with response shapes |
| `docs/DATASET.md` | Dataset profile, quirks, honest gaps |
| `docs/MODEL_CARD.md` | Model cards for layers A–D (params, per-class metrics, limitations) |
| `docs/EVALUATION.md` | Comparison, ablation, calibration, one-shot test evaluation |
| `docs/XAI.md` | Global/local explainability methodology |
| `docs/THREAT_SCORING.md` | Risk formula, weights, worked example |
| `docs/INCIDENT_CORRELATION.md` | Correlation matching score, story stages, simulated-metadata disclosure |
| `docs/LIMITATIONS.md` | Dataset/model/product-level limitations |
| `docs/QA_REPORT.md` | End-to-end QA results (parity, endpoints, browser E2E) |
| `notebooks/CipherMind_Model_Training_and_Evaluation.ipynb` | **Master ML notebook** (training + verification, with executed outputs) |

---

*CipherMind Sentinel · CipherMind AI '26 · built on UNSW-NB15 (Moustafa & Slay, 2015)*
