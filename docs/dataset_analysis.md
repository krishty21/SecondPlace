# Dataset Analysis — UNSW-NB15 (as supplied)

Generated automatically by the master notebook (`notebooks/CipherMind_Model_Training_and_Evaluation.ipynb` §03–§04; originally `ml/scripts/analyze_dataset.py`, now retired). Source of truth: the uploaded ZIP only.

## Files & Shape

| File | Rows | Columns | Schema OK |
|---|---|---|---|
| dataset/Training and Testing Sets/UNSW_NB15_training-set.csv | 175,341 | 45 | True |
| dataset/Training and Testing Sets/UNSW_NB15_testing-set.csv | 82,332 | 45 | True |

- Train/test schema match: **True**
- ID overlap between train and test: **82332** — both official files number `id` from 1 independently, so `id` is NOT a globally unique key across files. This is another reason `id` is excluded from model features. Rows themselves are disjoint official splits.
- Duplicate rows: train 0, test 0
- Label vs attack-category consistency mismatches: train 0, test 0

## Class Distribution (label)

| Split | Normal (0) | Attack (1) | Attack ratio |
|---|---|---|---|
| Train | 56,000 | 119,341 | 68.1% |
| Test | 37,000 | 45,332 | 55.1% |

## Attack Category Distribution

| Category | Train | Test |
|---|---|---|
| Analysis | 2,000 | 677 |
| Backdoor | 1,746 | 583 |
| DoS | 12,264 | 4,089 |
| Exploits | 33,393 | 11,132 |
| Fuzzers | 18,184 | 6,062 |
| Generic | 40,000 | 18,871 |
| Normal | 56,000 | 37,000 |
| Reconnaissance | 10,491 | 3,496 |
| Shellcode | 1,133 | 378 |
| Worms | 130 | 44 |

Rare classes (train): Worms (130), Shellcode (1133), Analysis (2000), Backdoor (1746).
The multiclass problem is **highly imbalanced** (max/min ratio ≈ 431:1).

## Missing / Infinite Values

- Missing values in train: none
- Infinite values in train: none
- Missing values in test: none
- Infinite values in test: none

Infinite values are expected in ratio-like derived fields of the original corpus (e.g. `rate`, `dload`). The pipeline sanitizes them (inf -> NaN -> training median imputation) with statistics fitted **on the training split only**.

## Categorical Fields

| Field | Train cardinality | Unseen values in test vs train |
|---|---|---|
| proto | 133 | 0 |
| service | 13 | 0 |
| state | 9 | 2 |

Encoder design handles unseen categories at inference via a fallback bucket "unknown". Frequencies/maps fitted on training only.

## Known Dataset Quirks

- `is_ftp_login` is documented as boolean but contains values such as [0, 1, 2, 4]; treated as ordinal numeric.
- `attack_cat` uses the string "Normal" for benign rows. Consistency check passed with 0 mismatches.

## Leakage Risk Assessment

1. **Excluded from features:** `id` (identifier only), `label`, `attack_cat` (targets).
2. **Preprocessors (imputation medians, encoders, scalers, calibration) fitted on the training split only** — never on test.
3. **Model selection & threshold tuning** use a stratified validation split carved from the training set. The official test set is scored **exactly once** at the end.
4. The official train/test files each number `id` from 1 (all 82332 test ids numerically collide with train ids). `id` is therefore excluded from features; row-level content is the official disjoint split.
5. No timestamps exist in the supplied 45-column CSVs — the replay layer creates **clearly-labeled synthetic** event times for correlation demo purposes (see docs/INCIDENT_CORRELATION.md). No claims of real IP/user identity are made: `srcip/sport/dstip/dsport` are NOT present in these CSVs.

## Feature Engineering Implications

Highly skewed positive variables (|skew| > 3, see `ml/artifacts/metrics/dataset_profile.json` -> `skewness`) receive `log1p` transforms. Derived behavioral ratios are computed from present fields only:
`byte_ratio, packet_ratio, total_bytes, total_packets, total_loss, payload_per_packet, load_ratio, size_ratio, timing ratios` — full list and formulas in `docs/ARCHITECTURE.md`.
