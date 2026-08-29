# Model Card

Hyperparameters quoted from `ml/artifacts/metadata/model_registry.json`; metrics from `ml/artifacts/metrics/test_evaluation.json` (official test set, scored **exactly once** after model freeze).

**Shared inputs (Models A–C).** 54 features per flow: 39 raw numeric (train-median imputation; `log1p` on the 24 columns with training |skew| > 3), 12 derived ratios, 3 ordinal-encoded categoricals with `__unknown__` fallback (handles test's unseen `ACC`/`CLO` states). Targets: `label` (A), `attack_cat` (B). `id` excluded.

**Shared training process.** (1) Model comparison on a stratified 15% validation split of **train only**. (2) 5-fold out-of-fold train predictions fit Platt scaling (A), temperature (B), and the F1-max threshold. (3) Final models refit on full train with rounds = mean of fold best-iterations (binary folds [857, 920, 760, 1049, 735] → 864; multiclass [98, 109, 101, 100, 105] → 102). (4) Test scored once. Seed 42, deterministic LightGBM.

---

## Model A — Binary attack detector

- **Purpose:** is this flow an attack? Drives the alert/normal verdict.
- **Algorithm:** LightGBM, 864 rounds, `objective=binary`, `learning_rate=0.06`, `num_leaves=96`, `min_data_in_leaf=60`, `feature_fraction=0.85`, `bagging_fraction=0.9`, `bagging_freq=1`, `lambda_l2=1.0`. Platt calibration (`a=0.8152`, `b=−0.0441`), threshold 0.46 (OOF F1-max).
- **Test metrics:** accuracy 0.8663 · precision 0.8124 · recall 0.9844 · F1 0.8902 · ROC-AUC 0.9826 · PR-AUC 0.9873 · specificity 0.0156 · **FPR 0.2785**. Confusion matrix (rows true Normal/Attack): TN 26,696 · FP 10,304 · FN 707 · TP 44,625.
- **Limitations:** recall-heavy operating point — ~1 in 3.6 normal test flows becomes an alert. Deliberate SOC-triage posture (correlation + risk scoring filter the noise); see [EVALUATION.md](EVALUATION.md), [LIMITATIONS.md](LIMITATIONS.md).
- **Intended use:** research/demo of the full detection→triage→explanation loop; not production detection.

## Model B — Multiclass category classifier

- **Purpose:** which of 10 categories is this attack? Runs only on flows A flagged (category = B's argmax when verdict = Attack).
- **Algorithm:** LightGBM, 102 rounds, `objective=multiclass` (10 classes), `learning_rate=0.08`, `num_leaves=64`, `min_data_in_leaf=50`, `feature_fraction=0.85`, `bagging_fraction=0.9`, `bagging_freq=1`, `lambda_l2=1.0`, **`class_weight=balanced`**. Temperature scaling T=1.0035 (OOF-fitted).
- **Test metrics:** accuracy 0.7711 · macro F1 0.5197 · weighted F1 0.7856 · balanced accuracy 0.5585.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Analysis | 677 | 0.0551 | 0.1108 | 0.0736 |
| Backdoor | 583 | 0.0332 | 0.0909 | 0.0486 |
| DoS | 4,089 | 0.6239 | 0.1071 | 0.1828 |
| Exploits | 11,132 | 0.6095 | 0.8562 | 0.7121 |
| Fuzzers | 6,062 | 0.3194 | 0.6229 | 0.4223 |
| Generic | 18,871 | 0.9959 | 0.9720 | 0.9838 |
| Normal | 37,000 | 0.9721 | 0.7599 | 0.8530 |
| Reconnaissance | 3,496 | 0.9300 | 0.8095 | 0.8656 |
| Shellcode | 378 | 0.3792 | 0.8016 | 0.5149 |
| Worms | 44 | 0.6667 | 0.4545 | 0.5405 |

- **Limitations:** rare classes weak despite balanced weights (Backdoor F1 0.049, Analysis 0.074; Worms 0.541 on 44 rows). DoS recall collapses to 0.107 with heavy DoS→Exploits confusion (2,543 of 4,089 true DoS predicted Exploits) — the families share flow signatures in UNSW-NB15. Because A and B are independent, a flow can be verdict=Attack while B's argmax is Normal; the engine then labels the category "Normal" (layered-design edge case, visible in the UI).
- **Intended use:** attack-family context for triage and correlation, not definitive attribution.

## Model C — Anomaly detector (normal-only training)

- **Purpose:** unsupervised "how abnormal vs. learned normal traffic?" — ranks, corroborates, feeds risk scoring.
- **Algorithm:** IsolationForest trained **only on normal training flows** (30,000-row sample), 120 trees, `max_samples=1024`, contamination auto. Paper score `s(x)=2^(−E(h)/c(n))`, `c_n=13.0174`; normalized 0–100 via training-normal percentile anchors (p50 0.3978→15, p90 0.4846→40, p99 0.5908→70, p999 0.6541→90).
- **Test metrics (vs. attack label):** ROC-AUC 0.7963 · PR-AUC 0.7949 · precision@100 0.96, @500 0.978, @1000 0.981, @5000 0.8846 · normalized scores: normal mean 24.7 / attack mean 41.4 (p90 43.9 vs 68.4).
- **Limitations:** moderate AUC — useful for **triage ranking** (top-1000 is 98.1% attacks), not standalone detection.
- **Intended use:** prioritization signal inside Model D and the correlation engine.

## Model D — Transparent threat-risk scoring

- **Purpose:** one 0–100 analyst-facing risk number with Low/Medium/High/Critical bands.
- **Algorithm:** fixed weighted sum over attack confidence, anomaly, category severity, rarity, uncertainty, correlation — formula, weights, worked example in [THREAT_SCORING.md](THREAT_SCORING.md). No learned "fake labels"; every term is inspectable and lives in `model_registry.json` → `risk_config` (configurable without retraining).
- **Limitations:** hand-tuned weights (32/18/20/8/10/12) encode SOC judgment, not learned optima; bands are heuristic.
- **Intended use:** transparent prioritization for the demo; a real deployment should tune weights against analyst outcomes.
