# Evaluation

Governing rule: **the official test set (82,332 rows) is scored exactly once**, after model selection, calibration, and threshold tuning are frozen on training data only. Environment: Python 3.12.14, LightGBM 4.5.0, scikit-learn 1.5.2, seed 42.

## Model comparison (stratified 15% validation split of TRAIN)

Binary candidates at threshold 0.5 (`metrics/model_comparison.json`):

| Model | F1 | ROC-AUC | PR-AUC | FPR | Fit (s) |
|---|---:|---:|---:|---:|---:|
| **LightGBM** (selected) | **0.9736** | **0.9947** | **0.9975** | 0.0667 | 69.0 |
| XGBoost (250×8) | 0.9722 | 0.9942 | 0.9972 | 0.0751 | 12.3 |
| RandomForest (150) | 0.9715 | 0.9941 | 0.9969 | 0.0583 | 57.3 |
| LogisticRegression (scaled) | 0.9547 | 0.9786 | 0.9872 | 0.1717 | 3.2 |

Multiclass, same split: LightGBM (balanced) macro-F1 **0.6291** / balanced accuracy 0.6111 vs RandomForest (balanced) 0.6188 / 0.6456 — LightGBM selected on macro-F1 and JSON portability for the TS engine.

## Feature ablation (`metrics/feature_ablation.json`)

| Feature set | # Features | Validation F1 |
|---|---:|---:|
| Raw numeric only | 39 | 0.9725 |
| Full (+ derived + encoded categoricals) | 54 | **0.9736** |

Δ = **+0.0011** — derived features barely move binary F1; they are kept for interpretability (readable ratios in narratives) and robustness, not accuracy.

## Calibration & threshold (`metrics/calibration.json`)

- **Platt** (Model A), fitted on 5-fold OOF train predictions: `a=0.8152, b=−0.0441`. OOF Brier **0.026341 → 0.026067**.
- **Temperature** (Model B, Guo et al. 2017): T = **1.0035**; OOF NLL 0.408963 (T=1) → 0.408961 — already near-calibrated; scaling kept as cheap insurance.
- **Threshold:** F1-max over a 0.05–0.95 grid on OOF calibrated probabilities → **0.46**. Rationale: in SOC triage a missed attack costs more than an extra alert that correlation/risk will deprioritize; 0.46 sits left of 0.5, buying recall.
- **Reliability:** 10-bin OOF curve exported with the artifact and rendered in the Explainability Center (near-diagonal after Platt). Test Brier = **0.0856** (`test_evaluation.json`) — the gap vs. OOF reflects distribution shift, not re-tuning; the top test bin (0.9–1.0, 43,602 events) realizes 95.0% positives.

## Binary — official test (`test_evaluation.json.binary`, threshold 0.46)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8663 | 0.8124 | 0.9844 | 0.8902 | 0.9826 | 0.9873 | 0.0156 | **0.2785** |

Confusion matrix: TN 26,696 · FP 10,304 · FN 707 · TP 44,625 — 98.44% of attacks caught at the documented cost of alerting on 27.85% of normals.

## Multiclass — official test (`test_evaluation.json.multiclass`)

Accuracy 0.7711, macro F1 0.5197, weighted F1 0.7856, balanced accuracy 0.5585. Full 10-class table in [MODEL_CARD.md](MODEL_CARD.md). Confusion highlights: DoS→Exploits 2,543 and Backdoor→Exploits 370 dominate rare-class errors — attribution noise on already-flagged attacks more than detection failure.

## Anomaly ranking (`test_evaluation.json.anomaly`)

| Metric | Value |
|---|---:|
| ROC-AUC / PR-AUC vs label | 0.7963 / 0.7949 |
| Precision@100 / @500 / @1000 / @5000 | 0.960 / 0.978 / 0.981 / 0.8846 |
| Recall@100 / @500 / @1000 / @5000 | 0.0021 / 0.0108 / 0.0216 / 0.0976 |

Normalized scores separate normals (mean 24.7, p90 43.9) from attacks (mean 41.4, p90 68.4). As a **top-k triage ranking** the detector is strong (98% precision in the first 1,000); as a standalone classifier it is not (ROC-AUC 0.80).

## Operational metrics (`metrics/operational.json`)

Python batch inference **2,645.1 ms per 10,000 events** (train-side reference). TypeScript runtime (live `/api/health` engineStats): 12,000-event boot in 18.7 s (**642 events/s**), single-event benchmark **1.7 ms**, end-to-end `POST /api/predict` 3–5 ms. Artifacts: binary 74.3 MB, multiclass 54.7 MB, IsolationForest 2.7 MB, SHAP cache 2.3 MB (JSON). Training: 146.7 s final resumable pass (cold run ≈ 15 min on 2 CPUs).

## Leakage-prevention checklist (implemented in the master notebook `notebooks/CipherMind_Model_Training_and_Evaluation.ipynb`; originally `ml/scripts/train.py`, retired)

1. `id` excluded from features (identifier; collides across files — [DATASET.md](DATASET.md)).
2. Targets `label`/`attack_cat` excluded from inputs; schema asserted on load.
3. All preprocessors (medians, log-column selection, category maps) fitted on **train only**.
4. Model selection on a stratified validation split carved from **train only**.
5. Calibration (Platt/temperature) and the 0.46 threshold fitted on **5-fold OOF train predictions**.
6. Final models refit on full train; test predictions computed once, after freeze.
7. Test reliability/Brier computed report-only — never fed back into any fitting.

## How to reproduce

```bash
# full retraining (MODE='train') or verification (MODE='verify') — the canonical ML workflow
jupyter execute notebooks/CipherMind_Model_Training_and_Evaluation.ipynb   # or open in Jupyter and Run All
python3 tests/validate_ts_engine.py     # needs soc-engine on :3010 — all rows must match
```

Metrics land in `ml/artifacts/metrics/` (deterministic given seed 42 and the pinned environment). The cross-validation prints per-row `OK` lines and exits non-zero on any mismatch.
