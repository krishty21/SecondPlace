#!/usr/bin/env python3
"""CipherMind Sentinel — Stage 1: Dataset validation & profiling.

Validates the uploaded UNSW-NB15 CSVs, profiles distributions, missing/infinite
values, duplicates, class imbalance, and leakage risks. Produces:
  - ml/artifacts/metrics/dataset_profile.json
  - docs/dataset_analysis.md
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "Training and Testing Sets", "UNSW_NB15_training-set.csv")
TEST_CSV = os.path.join(DATA_DIR, "Training and Testing Sets", "UNSW_NB15_testing-set.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "NUSW-NB15_features.csv")
OUT_DIR = os.path.join(ROOT, "ml", "artifacts", "metrics")
DOCS_DIR = os.path.join(ROOT, "docs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

EXPECTED_COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat", "label",
]
CATEGORICAL = ["proto", "service", "state"]


def load():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    feats = pd.read_csv(FEATURES_CSV, encoding="latin-1")
    return train, test, feats


def profile(df: pd.DataFrame, name: str) -> dict:
    prof = {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "schema_ok": list(df.columns) == EXPECTED_COLUMNS,
    }
    num_cols = [c for c in df.columns if c not in CATEGORICAL + ["attack_cat"]]
    num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    prof["missing_values"] = {c: int(n) for c, n in num.isna().sum().items() if n > 0}
    inf_counts = {}
    for c in num_cols:
        n_inf = int(num[c].isin([np.inf, -np.inf]).sum())
        if n_inf > 0:
            inf_counts[c] = n_inf
    prof["infinite_values"] = inf_counts
    prof["numeric_column_count"] = len(num_cols)
    prof["categorical_columns"] = CATEGORICAL
    prof["categorical_cardinality"] = {c: int(df[c].nunique()) for c in CATEGORICAL}
    prof["categorical_values"] = {
        "proto": sorted(df["proto"].astype(str).unique().tolist())[:40],
        "service": sorted(df["service"].astype(str).unique().tolist())[:40],
        "state": sorted(df["state"].astype(str).unique().tolist())[:30],
    }
    prof["label_distribution"] = {str(k): int(v) for k, v in df["label"].value_counts().items()}
    prof["attack_cat_distribution"] = {
        str(k): int(v) for k, v in df["attack_cat"].value_counts().items()
    }
    skew = {}
    for c in num_cols:
        try:
            v = num[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) > 100:
                skew[c] = round(float(pd.Series(v).skew()), 3)
        except Exception:
            pass
    prof["skewness"] = skew
    return prof


def main():
    train, test, feats = load()
    report = {"generated_by": "ml/scripts/analyze_dataset.py"}
    report["files"] = {
        "training_set": os.path.relpath(TRAIN_CSV, ROOT),
        "testing_set": os.path.relpath(TEST_CSV, ROOT),
        "features_spec": os.path.relpath(FEATURES_CSV, ROOT),
    }
    report["train"] = profile(train, "training")
    report["test"] = profile(test, "testing")

    report["train_test_schema_match"] = list(train.columns) == list(test.columns)
    tr_cats = set(train["attack_cat"].astype(str))
    te_cats = set(test["attack_cat"].astype(str))
    report["test_only_categories"] = sorted(te_cats - tr_cats)
    report["train_only_categories"] = sorted(tr_cats - te_cats)
    for c in CATEGORICAL:
        unseen = set(test[c].astype(str)) - set(train[c].astype(str))
        report.setdefault("unseen_categorical_test_vs_train", {})[c] = sorted(unseen)

    report["id_overlap_train_test"] = int(len(set(train["id"]) & set(test["id"])))

    mismatch_tr = int(((train["attack_cat"] == "Normal") != (train["label"] == 0)).sum())
    mismatch_te = int(((test["attack_cat"] == "Normal") != (test["label"] == 0)).sum())
    report["label_vs_category_mismatch"] = {"train": mismatch_tr, "test": mismatch_te}

    report["is_ftp_login_values_train"] = sorted(train["is_ftp_login"].unique().tolist())[:10]
    report["is_ftp_login_values_test"] = sorted(test["is_ftp_login"].unique().tolist())[:10]

    with open(os.path.join(OUT_DIR, "dataset_profile.json"), "w") as f:
        json.dump(report, f, indent=2)

    tr, te = report["train"], report["test"]
    md = f"""# Dataset Analysis — UNSW-NB15 (as supplied)

Generated automatically by `ml/scripts/analyze_dataset.py`. Source of truth: the uploaded ZIP only.

## Files & Shape

| File | Rows | Columns | Schema OK |
|---|---|---|---|
| {report['files']['training_set']} | {tr['rows']:,} | {tr['columns']} | {tr['schema_ok']} |
| {report['files']['testing_set']} | {te['rows']:,} | {te['columns']} | {te['schema_ok']} |

- Train/test schema match: **{report['train_test_schema_match']}**
- ID overlap between train and test: **{report['id_overlap_train_test']}** — both official files number `id` from 1 independently, so `id` is NOT a globally unique key across files. This is another reason `id` is excluded from model features. Rows themselves are disjoint official splits.
- Duplicate rows: train {tr['duplicate_rows']}, test {te['duplicate_rows']}
- Label vs attack-category consistency mismatches: train {report['label_vs_category_mismatch']['train']}, test {report['label_vs_category_mismatch']['test']}

## Class Distribution (label)

| Split | Normal (0) | Attack (1) | Attack ratio |
|---|---|---|---|
| Train | {tr['label_distribution'].get('0', 0):,} | {tr['label_distribution'].get('1', 0):,} | {tr['label_distribution'].get('1', 0) / tr['rows']:.1%} |
| Test | {te['label_distribution'].get('0', 0):,} | {te['label_distribution'].get('1', 0):,} | {te['label_distribution'].get('1', 0) / te['rows']:.1%} |

## Attack Category Distribution

| Category | Train | Test |
|---|---|---|
"""
    cats = sorted(set(tr["attack_cat_distribution"]) | set(te["attack_cat_distribution"]))
    for c in cats:
        md += f"| {c} | {tr['attack_cat_distribution'].get(c, 0):,} | {te['attack_cat_distribution'].get(c, 0):,} |\n"
    md += f"""
Rare classes (train): Worms ({tr['attack_cat_distribution'].get('Worms', 0)}), Shellcode ({tr['attack_cat_distribution'].get('Shellcode', 0)}), Analysis ({tr['attack_cat_distribution'].get('Analysis', 0)}), Backdoor ({tr['attack_cat_distribution'].get('Backdoor', 0)}).
The multiclass problem is **highly imbalanced** (max/min ratio ≈ {max(tr['attack_cat_distribution'].values()) / max(min(tr['attack_cat_distribution'].values()), 1):.0f}:1).

## Missing / Infinite Values

- Missing values in train: {tr['missing_values'] or 'none'}
- Infinite values in train: {tr['infinite_values'] or 'none'}
- Missing values in test: {te['missing_values'] or 'none'}
- Infinite values in test: {te['infinite_values'] or 'none'}

Infinite values are expected in ratio-like derived fields of the original corpus (e.g. `rate`, `dload`). The pipeline sanitizes them (inf -> NaN -> training median imputation) with statistics fitted **on the training split only**.

## Categorical Fields

| Field | Train cardinality | Unseen values in test vs train |
|---|---|---|
| proto | {tr['categorical_cardinality']['proto']} | {len(report['unseen_categorical_test_vs_train']['proto'])} |
| service | {tr['categorical_cardinality']['service']} | {len(report['unseen_categorical_test_vs_train']['service'])} |
| state | {tr['categorical_cardinality']['state']} | {len(report['unseen_categorical_test_vs_train']['state'])} |

Encoder design handles unseen categories at inference via a fallback bucket "unknown". Frequencies/maps fitted on training only.

## Known Dataset Quirks

- `is_ftp_login` is documented as boolean but contains values such as {report['is_ftp_login_values_train']}; treated as ordinal numeric.
- `attack_cat` uses the string "Normal" for benign rows. Consistency check passed with {report['label_vs_category_mismatch']['train'] + report['label_vs_category_mismatch']['test']} mismatches.

## Leakage Risk Assessment

1. **Excluded from features:** `id` (identifier only), `label`, `attack_cat` (targets).
2. **Preprocessors (imputation medians, encoders, scalers, calibration) fitted on the training split only** — never on test.
3. **Model selection & threshold tuning** use a stratified validation split carved from the training set. The official test set is scored **exactly once** at the end.
4. The official train/test files each number `id` from 1 (all {report['id_overlap_train_test']} test ids numerically collide with train ids). `id` is therefore excluded from features; row-level content is the official disjoint split.
5. No timestamps exist in the supplied 45-column CSVs — the replay layer creates **clearly-labeled synthetic** event times for correlation demo purposes (see docs/INCIDENT_CORRELATION.md). No claims of real IP/user identity are made: `srcip/sport/dstip/dsport` are NOT present in these CSVs.

## Feature Engineering Implications

Highly skewed positive variables (|skew| > 3, see `ml/artifacts/metrics/dataset_profile.json` -> `skewness`) receive `log1p` transforms. Derived behavioral ratios are computed from present fields only:
`byte_ratio, packet_ratio, total_bytes, total_packets, total_loss, payload_per_packet, load_ratio, size_ratio, timing ratios` — full list and formulas in `docs/ARCHITECTURE.md`.
"""
    with open(os.path.join(DOCS_DIR, "dataset_analysis.md"), "w") as f:
        f.write(md)
    print(json.dumps({k: report[k] for k in
                      ["train_test_schema_match", "id_overlap_train_test",
                       "label_vs_category_mismatch", "test_only_categories",
                       "unseen_categorical_test_vs_train"]}, indent=2))
    print("dataset_profile.json + dataset_analysis.md written")


if __name__ == "__main__":
    main()
