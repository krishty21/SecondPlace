# CipherMind Sentinel — AI Security Operations Copilot

**Hackathon project (CipherMind AI '26) on the UNSW-NB15 network-intrusion dataset.**

CipherMind Sentinel transforms noisy network-security data into **prioritized threats**, **explainable detections**, **attack stories**, and **analyst-ready incident intelligence**: a Python pipeline trains a layered model architecture on flow records, a TypeScript inference service scores real test events live, and a SOC UI correlates alerts into incidents and grounds an LLM analyst summary strictly in model evidence. Every probability is calibrated, every risk score is transparent arithmetic, every limitation is documented ([docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

## Architecture

```
UNSW-NB15 CSVs (175,341 train / 82,332 test rows × 45 cols)
        │
        ▼
Python training pipeline — ml/scripts/train.py (12 stages, resumable)
  54-feature pipeline → model comparison → 5-fold OOF calibration
  → final models → KMeans/PCA clustering → TreeSHAP → JSON artifacts
        │
        ▼   ml/artifacts/  (models + preprocessor + metrics + replay, all JSON, ~134 MB)
soc-engine — Bun/TypeScript inference service (port 3010)
  LightGBM tree walker (binary + multiclass) · IsolationForest scorer
  · transparent risk engine · streaming correlation · deterministic replay
  REST /api/* + socket.io live ticks
        │
        ▼
Next.js 16 SOC UI (port 3000) — 5 views
  Command Center · Incident Investigation · Pattern Explorer
  · Explainability Center · Live Replay
  └─ z-ai LLM endpoints /api/ai/* (incident-summary, analyst-chat)
```

## Layered model architecture

| Model | Role | Algorithm |
|---|---|---|
| **A** | Binary attack detection | LightGBM (864 rounds), Platt-calibrated, F1-max threshold 0.46 |
| **B** | Attack-category classification (10 classes) | LightGBM (102 rounds, `class_weight: balanced`), temperature-scaled |
| **C** | Unsupervised anomaly scoring | IsolationForest trained on **normal traffic only** (120 trees) |
| **D** | Threat-risk prioritization | Transparent weighted formula, no learned labels — [docs/THREAT_SCORING.md](docs/THREAT_SCORING.md) |

Inputs to A–C: 54 engineered features (39 raw numeric + 12 derived ratios + 3 encoded categoricals; `log1p` on the 24 columns with training |skew| > 3). Details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Quick results (official test set, scored exactly once)

| Model | Key metrics |
|---|---|
| A — Binary | F1 **0.8902** · ROC-AUC **0.9826** · PR-AUC **0.9873** · recall **0.9844** · precision 0.8124 @ threshold 0.46 |
| B — Multiclass | accuracy **0.7711** · weighted F1 **0.7856** · macro F1 0.5197 |
| C — Anomaly | ROC-AUC **0.7963** · PR-AUC 0.7949 · precision@1000 **0.981** |
| Calibration | OOF Brier 0.02634 → 0.02607 (Platt); test Brier 0.08556 |

Model selection (validation): LightGBM 0.9736 F1 / 0.9947 AUC > XGBoost 0.9722 / 0.9942 > RandomForest 0.9715 / 0.9941 > LogReg 0.9547 / 0.9786. Full methodology: [docs/EVALUATION.md](docs/EVALUATION.md).

## How to run

```bash
bun install && bun run dev                  # UI + LLM endpoints, port 3000
cd mini-services/soc-engine && bun install && bun run dev   # inference engine, port 3010
python3 ml/scripts/train.py                 # optional retrain (seed 42, artifacts already committed)
```

The engine needs `ml/artifacts/` present (committed). Cold start scores a 12,000-event stratified boot sample live through the real models (~19 s on this 2-CPU sandbox; stats at `/api/health`). Recorded training time: 146.7 s for the final resumable pass (`metrics/operational.json`); a cold run of all 12 stages takes roughly 15 minutes on the 2-CPU/4 GB sandbox. Services are also managed idempotently by `scripts/ensure-services.sh` (starts whichever of :3010 / :3000 is down; safe for cron/QA loops).

## Key file map

| Path | What it is |
|---|---|
| `ml/scripts/train.py` | 12-stage training pipeline (leakage-safe, resumable) |
| `ml/training/features.py` | Shared feature engineering → `feature_config.json` |
| `ml/scripts/validate_ts_engine.py` | TS-vs-Python cross-validation |
| `ml/artifacts/` | Models, preprocessor, metrics, replay, registry (JSON) |
| `mini-services/soc-engine/src/` | `lgbm.ts` `iforest.ts` `engine.ts` `correlate.ts` `replay.ts` `state.ts` `index.ts` |
| `src/app/api/ai/` | LLM routes: `incident-summary`, `analyst-chat` |
| `src/components/soc/`, `src/lib/` | SOC UI components + typed API client |

## Demo walkthrough (the 13-step story)

1. Open `http://localhost:3000` → **Command Center**: boot state from 12,000 stratified test events scored live (not cached).
2. Read the KPI strip (events, alerts, active/critical incidents, detection rate, mean anomaly, FP indicator).
3. Scan the threat timeline and "What needs attention now?" — incidents ranked by risk.
4. Click a recent critical alert → dialog with the raw 45-field flow, calibrated probabilities, feature attributions.
5. Switch to **Live Replay** → Start; background traffic streams first (socket.io, REST fallback built in).
6. Watch **campaign1-recon** emerge — a Reconnaissance burst forms a correlated incident.
7. **campaign2-exploit** (Exploits + Shellcode) escalates incidents toward Critical; nudge speed 1×→8× to fast-forward.
8. Replay completes (903 events, 5 waves) → "Analyze final incidents".
9. **Incident Investigation**: open a replay incident (RINC-*) — risk gauge, simulated-entity chips, category mix.
10. Walk the 6 tabs: Overview, Timeline (attack story), Evidence, Explainability, AI Analyst, Related Patterns.
11. **AI Analyst**: generate the LLM incident summary (`source: llm`) and chat with the Analyst Copilot.
12. **Explainability Center**: global TreeSHAP (sttl dominates), reliability + threshold curves, local explorer with the `exact-treeshap` vs `saabas-path` method badge.
13. **Pattern Explorer**: 8 behavioral clusters (KMeans on training attacks) with live alert counts — labeled as behavior groups, not malware families.

## Honest limitations (summary)

- The 45-column CSVs contain **no IPs, timestamps, users, or devices**; all timestamps and `ENT-*` entities in the UI are clearly-labeled simulated replay metadata.
- Recall-heavy operating point: **FPR 27.85%** on test normals (deliberate SOC-triage posture, documented).
- Rare classes weak: Worms F1 0.541, Backdoor F1 0.049, Analysis F1 0.074; Exploits↔DoS confusion significant.
- Research prototype — single-tenant, no auth, no real-time feed ingestion.

Full list with numbers and production requirements: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Documentation index

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, TS inference engine, verification, performance |
| [docs/DATASET.md](docs/DATASET.md) | UNSW-NB15 as supplied, quirks, what is NOT in the data |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Per-model cards (A/B/C/D) |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Methodology, comparisons, calibration, leakage policy, reproduction |
| [docs/THREAT_SCORING.md](docs/THREAT_SCORING.md) | Risk formula, weights, bands, worked example |
| [docs/INCIDENT_CORRELATION.md](docs/INCIDENT_CORRELATION.md) | Alert→incident correlation, attack stories, playbooks |
| [docs/XAI.md](docs/XAI.md) | Global TreeSHAP + local exact/Saabas explanations |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Honest dataset/model/product limitations |
| [docs/QA_REPORT.md](docs/QA_REPORT.md) | QA actually performed (API, E2E, cross-validation, lint) |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | soc-engine REST + socket.io contract |
| [docs/dataset_analysis.md](docs/dataset_analysis.md) | Raw dataset profiling (generated) |
