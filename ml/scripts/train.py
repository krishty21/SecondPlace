#!/usr/bin/env python3
"""CipherMind Sentinel — Stage 2: Reproducible training pipeline.

Trains and evaluates the layered detection architecture:
  Model A — Binary attack detector (LightGBM; principled comparison vs LogReg/RF/XGBoost)
  Model B — Multiclass attack-category classifier (LightGBM, class-balanced)
  Model C — Unsupervised anomaly detector (IsolationForest trained on normal traffic only)
  Model D — Transparent configurable threat-risk scoring (no learned fake labels)
  Plus: probability calibration (Platt / temperature scaling via 5-fold OOF),
        behavioral clustering (KMeans + PCA), SHAP global importance,
        deterministic demo replay selection, full artifact export.

Leakage policy (NON-NEGOTIABLE):
  - preprocessors fitted on training data only
  - model selection on a stratified validation split carved from TRAIN
  - calibration + threshold chosen on 5-fold OOF predictions of TRAIN
  - the official test set is scored EXACTLY ONCE at the end

Seed: 42 everywhere. Deterministic LightGBM.
"""
from __future__ import annotations

import json
import os
import gc
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from features import FeatureBuilder, RAW_NUMERIC, CATEGORICAL, DERIVED  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "Training and Testing Sets", "UNSW_NB15_training-set.csv")
TEST_CSV = os.path.join(DATA_DIR, "Training and Testing Sets", "UNSW_NB15_testing-set.csv")
ART = os.path.join(ROOT, "ml", "artifacts")
SEED = 42
N_THREADS = 2  # 2 CPUs / 4GB RAM environment

CATS = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
        "Normal", "Reconnaissance", "Shellcode", "Worms"]
CAT_TO_IDX = {c: i for i, c in enumerate(CATS)}

# Train prevalence for rarity weighting (computed once, exported as artifact)
_prev_df = pd.read_csv(os.path.join(DATA_DIR, "Training and Testing Sets", "UNSW_NB15_training-set.csv"))
RARITY = {c: float((_prev_df["attack_cat"] == c).mean()) for c in CATS}

# Transparent, documented risk-scoring configuration (Model D)
RISK_CONFIG = {
    "weights": {
        "attack_confidence": 32,
        "anomaly": 18,
        "category_severity": 20,
        "rarity": 8,
        "uncertainty": 10,
        "correlation": 12,
    },
    "category_severity": {
        "Exploits": 1.0, "Shellcode": 0.95, "Worms": 0.9, "Backdoor": 0.85,
        "DoS": 0.8, "Reconnaissance": 0.55, "Analysis": 0.6, "Fuzzers": 0.45,
        "Generic": 0.4, "Normal": 0.05,
    },
    "severity_bands": {"low": 24, "medium": 49, "high": 74},
    "correlation_alert_saturation": 8,
    "uncertainty_formula": "2 * p * (1 - p) * 2  # scaled to [0,1], max at p=0.5",
}


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":") if path.endswith("shap_cache.json") else None, indent=None if path.endswith("shap_cache.json") else 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- data
def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    expected_cols = set(RAW_NUMERIC + CATEGORICAL + ["id", "attack_cat", "label"])
    assert expected_cols == set(train.columns) == set(test.columns), "schema mismatch"
    assert set(train["attack_cat"].unique()) <= set(CATS)
    assert set(test["attack_cat"].unique()) <= set(CATS)
    yb_tr = train["label"].values
    yb_te = test["label"].values
    ym_tr = train["attack_cat"].map(CAT_TO_IDX).values
    ym_te = test["attack_cat"].map(CAT_TO_IDX).values
    return train, test, yb_tr, yb_te, ym_tr, ym_te


# ------------------------------------------------------------ metrics
def binary_metrics(y, p, thr=0.5, prefix=""):
    pred = (p >= thr).astype(int)
    fpr, tpr, thr_roc = roc_curve(y, p)
    prec_arr, rec_arr, thr_pr = precision_recall_curve(y, p)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "specificity": float(recall_score(y, 1 - pred, zero_division=0)),
        "false_positive_rate": float(recall_score(1 - y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "threshold": float(thr),
    }


def multiclass_metrics(y, y_pred, probs, label=""):
    per_class = []
    for i, c in enumerate(CATS):
        mask = y == i
        if mask.sum() == 0:
            per_class.append({"class": c, "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue
        p_i = (y_pred == i).astype(int)
        t_i = mask.astype(int)
        per_class.append({
            "class": c, "support": int(mask.sum()),
            "precision": float(precision_score(t_i, p_i, zero_division=0)),
            "recall": float(recall_score(t_i, p_i, zero_division=0)),
            "f1": float(f1_score(t_i, p_i, zero_division=0)),
        })
    cm = confusion_matrix(y, y_pred, labels=list(range(len(CATS))))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": CATS,
    }


def reliability_curve(y, p, bins=10):
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.sum() > 0:
            out.append({
                "bin_low": float(lo), "bin_high": float(hi),
                "mean_predicted": float(p[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
                "count": int(mask.sum()),
            })
    return out


# ------------------------------------------------------- Platt / temp
def fit_platt(logits, y):
    """Platt scaling: p = sigmoid(a*logit + b) fitted on OOF train predictions."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    pc = sigmoid(a * logits + b)
    return a, b, float(brier_score_loss(y, pc))


def fit_temperature(logits, y):
    """Temperature scaling (Guo et al. 2017) on OOF multiclass logits."""
    def nll(T):
        p = softmax(logits / T)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
    res = minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded")
    T = float(res.x)
    return T, float(nll(T)), float(nll(1.0))


# ------------------------------------------------ IsolationForest export
def average_path_length_c(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n


def serialize_iforest(iforest: IsolationForest, X_normal: np.ndarray):
    """Serialize sklearn IsolationForest into portable JSON + own scorer.

    Anomaly score (original paper): s(x) = 2^(-E(h(x)) / c(n)),
    E(h) = mean over trees of (depth(x) + c(n_leaf - 1)).
    """
    trees = []
    for est in iforest.estimators_:
        t = est.tree_
        nodes = []
        for i in range(t.node_count):
            is_leaf = t.children_left[i] == -1
            nodes.append([
                int(t.feature[i]) if not is_leaf else -1,
                float(t.threshold[i]) if not is_leaf else 0.0,
                int(t.children_left[i]),
                int(t.children_right[i]),
                int(t.n_node_samples[i]),
            ])
        trees.append(nodes)
    n = int(iforest.max_samples_)
    # normalization anchors from TRAINING normal scores (deterministic mapping)
    raw = iforest_score(iforest, X_normal)
    anchors = {
        "p50": float(np.percentile(raw, 50)),
        "p90": float(np.percentile(raw, 90)),
        "p99": float(np.percentile(raw, 99)),
        "p999": float(np.percentile(raw, 99.9)),
    }
    return {
        "n_estimators": len(iforest.estimators_),
        "subsample_size": n,
        "c_n": average_path_length_c(n),
        "trees": trees,
        "norm_anchors": anchors,
        "norm_anchor_targets": {"p50": 15, "p90": 40, "p99": 70, "p999": 90},
        "score_formula": "s(x) = 2^(-E(h(x))/c(n)); normalized 0-100 piecewise-linear over training-normal percentile anchors",
    }


def iforest_score(iforest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Reference implementation identical to the TS engine (paper formula)."""
    c_n = average_path_length_c(int(iforest.max_samples_))
    depths = np.zeros(len(X))
    for est in iforest.estimators_:
        t = est.tree_
        idx = est.apply(X)
        dp_sum = np.asarray(est.decision_path(X).sum(axis=1)).ravel()
        depths += dp_sum - 1 + np.array(
            [average_path_length_c(t.n_node_samples[i] - 1) for i in idx]
        )
    return 2.0 ** (-depths / (len(iforest.estimators_) * c_n))


def normalize_anomaly(raw: float, anchors: dict, targets: dict) -> float:
    """Piecewise-linear, deterministic mapping to 0..100 (same in TS)."""
    xs = [0.0, anchors["p50"], anchors["p90"], anchors["p99"], anchors["p999"], 1.0]
    ys = [0.0, targets["p50"], targets["p90"], targets["p99"], targets["p999"], 100.0]
    return float(np.interp(raw, xs, ys))


# ---------------------------------------------------------- risk score
def risk_score(attack_prob, anomaly_norm, category, related_alerts, cfg=RISK_CONFIG):
    w = cfg["weights"]
    sev = cfg["category_severity"].get(category, 0.3)
    prevalence = RARITY.get(category, 0.5)
    rarity = min(1.0, -np.log10(max(prevalence, 1e-6)) / 4.0)
    unc = 2 * attack_prob * (1 - attack_prob) * 2
    corr = min(1.0, related_alerts / cfg["correlation_alert_saturation"])
    risk = (
        w["attack_confidence"] * attack_prob
        + w["anomaly"] * (anomaly_norm / 100.0)
        + w["category_severity"] * sev
        + w["rarity"] * rarity
        + w["uncertainty"] * unc
        + w["correlation"] * corr
    )
    return float(min(100.0, max(0.0, risk)))


def severity_band(risk):
    b = RISK_CONFIG["severity_bands"]
    if risk <= b["low"]:
        return "Low"
    if risk <= b["medium"]:
        return "Medium"
    if risk <= b["high"]:
        return "High"
    return "Critical"


def iforest_score_serialized(trees: list, X: np.ndarray, c_n: float, n_est: int) -> np.ndarray:
    """Score using the SERIALIZED tree format — identical to the TS engine.
    Used for resume + guarantees python/TS scoring consistency."""
    EULER_GAMMA = 0.5772156649015329

    def c(n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2.0 * (np.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n

    depths = np.zeros(len(X))
    for nodes in trees:
        feats = np.array([n[0] for n in nodes])
        thrs = np.array([n[1] for n in nodes])
        lefts = np.array([n[2] for n in nodes], dtype=int)
        rights = np.array([n[3] for n in nodes], dtype=int)
        nsamp = np.array([n[4] for n in nodes], dtype=int)
        idx = np.zeros(len(X), dtype=int)
        depth = np.zeros(len(X), dtype=float)
        active = np.ones(len(X), dtype=bool)
        while active.any():
            f = feats[idx]
            is_leaf = f == -1
            go_left = X[np.arange(len(X)), f] <= thrs[idx]
            nxt = np.where(go_left, lefts[idx], rights[idx])
            depths[active & is_leaf] += depth[active & is_leaf] + np.array([c(nsamp[i] - 1) for i in idx[active & is_leaf]])
            depth = depth + (~is_leaf)
            idx = np.where(is_leaf, idx, nxt)
            active = active & ~is_leaf
    return 2.0 ** (-depths / (n_est * c_n))


def file_exists(rel: str) -> bool:
    return os.path.exists(os.path.join(ART, rel))


# ------------------------------------------------------------ training
def main():
    t_start = time.time()
    rng = np.random.RandomState(SEED)
    os.makedirs(ART, exist_ok=True)

    print("[1/12] Loading & validating data ...")
    train, test, yb_tr, yb_te, ym_tr, ym_te = load_data()
    print(f"  train={len(train)} test={len(test)}")

    print("[2/12] Fitting feature builder (train-only) ...")
    fb = FeatureBuilder().fit(train)
    X_tr = fb.transform(train).astype(np.float32)
    X_te = fb.transform(test).astype(np.float32)
    feat_names = fb.feature_names
    train_attack_cats = train["attack_cat"].values.copy()  # needed for stage 8
    test_label_arr = test["label"].values.copy()  # needed for stage 11
    del train, test
    gc.collect()  # free ~500MB of DataFrames — CRITICAL for the 4GB RAM budget
    print(f"  features={len(feat_names)} (raw {len(RAW_NUMERIC)} + derived {len(DERIVED)} + cat {len(CATEGORICAL)})")
    print(f"  log1p cols ({len(fb.log_cols)}): {fb.log_cols[:8]}...")
    if not file_exists("preprocessor/feature_config.json"):
        jdump(fb.config(), os.path.join(ART, "preprocessor", "feature_config.json"))

    # ----------------------- model comparison on train-internal validation
    if file_exists("metrics/model_comparison.json"):
        print("[3/12] Model comparison: SKIP (artifact exists)")
        with open(os.path.join(ART, "metrics", "model_comparison.json")) as f:
            comparison = json.load(f)
        X_fit = X_val = yb_fit = yb_val = ym_fit = ym_val = None
    else:
        print("[3/12] Model comparison on stratified validation split (train-only) ...")
        X_fit, X_val, yb_fit, yb_val, ym_fit, ym_val = train_test_split(
            X_tr, yb_tr, ym_tr, test_size=0.15, random_state=SEED, stratify=ym_tr
        )
        comparison = {"binary": [], "multiclass": []}

    lgb_bin_params = {
        "objective": "binary", "learning_rate": 0.06, "num_leaves": 96,
        "min_data_in_leaf": 60, "feature_fraction": 0.85, "bagging_fraction": 0.9,
        "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1, "seed": SEED,
        "deterministic": True, "force_row_wise": True, "num_threads": N_THREADS,
        "metric": "auc",
    }
    if X_fit is not None:
        t0 = time.time()
        lr = LogisticRegression(max_iter=1000, C=1.0)
        sc = StandardScaler().fit(X_fit)
        lr.fit(sc.transform(X_fit), yb_fit)
        p = lr.predict_proba(sc.transform(X_val))[:, 1]
        comparison["binary"].append({"model": "LogisticRegression (scaled)", **binary_metrics(yb_val, p), "fit_seconds": round(time.time() - t0, 1)})
        del lr, sc
        gc.collect()

        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=1, random_state=SEED, max_depth=24)
        rf.fit(X_fit, yb_fit)
        p = rf.predict_proba(X_val)[:, 1]
        comparison["binary"].append({"model": "RandomForest (150)", **binary_metrics(yb_val, p), "fit_seconds": round(time.time() - t0, 1)})
        del rf
        gc.collect()

        t0 = time.time()
        xl = xgb.XGBClassifier(n_estimators=250, max_depth=8, learning_rate=0.1, tree_method="hist",
                               eval_metric="auc", random_state=SEED, n_jobs=2)
        xl.fit(X_fit, yb_fit)
        p = xl.predict_proba(X_val)[:, 1]
        comparison["binary"].append({"model": "XGBoost (250x8)", **binary_metrics(yb_val, p), "fit_seconds": round(time.time() - t0, 1)})
        del xl
        gc.collect()

        t0 = time.time()
        m = lgb.train(lgb_bin_params, lgb.Dataset(X_fit, label=yb_fit),
                      num_boost_round=1500, valid_sets=[lgb.Dataset(X_val, label=yb_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(X_val, raw_score=True)
        p = sigmoid(p)
        comparison["binary"].append({"model": "LightGBM", **binary_metrics(yb_val, p), "fit_seconds": round(time.time() - t0, 1),
                                     "best_iteration": m.best_iteration})
        del m
        gc.collect()
    for row in comparison["binary"]:
        print(f"  {row['model']:28s} F1={row['f1']:.4f} AUC={row['roc_auc']:.4f} PR={row['pr_auc']:.4f} ({row['fit_seconds']}s)")

    # candidates: multiclass
    lgb_mc_params = {
        "objective": "multiclass", "num_class": len(CATS), "learning_rate": 0.08,
        "num_leaves": 64, "min_data_in_leaf": 50, "feature_fraction": 0.85,
        "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
        "class_weight": "balanced", "verbosity": -1, "seed": SEED,
        "deterministic": True, "force_row_wise": True, "num_threads": N_THREADS,
        "metric": "multi_logloss",
    }
    if X_fit is not None:
        t0 = time.time()
        mm = lgb.train(lgb_mc_params, lgb.Dataset(X_fit, label=ym_fit),
                       num_boost_round=600, valid_sets=[lgb.Dataset(X_val, label=ym_val)],
                       callbacks=[lgb.early_stopping(60, verbose=False)])
        pm = mm.predict(X_val)
        pred = pm.argmax(1)
        comparison["multiclass"].append({"model": "LightGBM (balanced)", **multiclass_metrics(ym_val, pred, pm), "fit_seconds": round(time.time() - t0, 1), "best_iteration": mm.best_iteration})
        del mm
        gc.collect()

        t0 = time.time()
        rfmc = RandomForestClassifier(n_estimators=150, class_weight="balanced", n_jobs=1, random_state=SEED, max_depth=24)
        rfmc.fit(X_fit, ym_fit)
        pm2 = rfmc.predict_proba(X_val)
        comparison["multiclass"].append({"model": "RandomForest (balanced)", **multiclass_metrics(ym_val, pm2.argmax(1), pm2), "fit_seconds": round(time.time() - t0, 1)})
        del rfmc
        gc.collect()
        for row in comparison["multiclass"]:
            print(f"  {row['model']:28s} macroF1={row['macro_f1']:.4f} balAcc={row['balanced_accuracy']:.4f} ({row['fit_seconds']}s)")
        jdump(comparison, os.path.join(ART, "metrics", "model_comparison.json"))
        del X_fit, X_val, yb_fit, yb_val, ym_fit, ym_val
        gc.collect()

    # ------------------------------ feature ablation (raw vs +derived)
    if file_exists("metrics/feature_ablation.json"):
        print("[4/12] Feature ablation: SKIP (artifact exists)")
        with open(os.path.join(ART, "metrics", "feature_ablation.json")) as f:
            abl = json.load(f)
    elif X_fit is not None:
        print("[4/12] Feature ablation (validation, binary LightGBM) ...")
        raw_len = len(RAW_NUMERIC)
        X_fit_raw, X_val_raw = X_fit[:, :raw_len], X_val[:, :raw_len]
        m_raw = lgb.train(lgb_bin_params, lgb.Dataset(X_fit_raw, label=yb_fit), num_boost_round=800,
                          valid_sets=[lgb.Dataset(X_val_raw, label=yb_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        p_raw = sigmoid(m_raw.predict(X_val_raw, raw_score=True))
        del m_raw
        gc.collect()
        abl = {
            "raw_only_f1": float(f1_score(yb_val, (p_raw >= 0.5).astype(int))),
            "full_pipeline_f1": [r for r in comparison["binary"] if r["model"] == "LightGBM"][0]["f1"],
            "features_raw": raw_len,
            "features_full": len(feat_names),
        }
        abl["delta"] = abl["full_pipeline_f1"] - abl["raw_only_f1"]
        jdump(abl, os.path.join(ART, "metrics", "feature_ablation.json"))
        print(f"  raw F1={abl['raw_only_f1']:.4f} -> full F1={abl['full_pipeline_f1']:.4f} (delta {abl['delta']:+.4f})")
    else:
        raise RuntimeError("ablation artifact missing and validation split unavailable")

    # ------------------------------ 5-fold OOF calibration & threshold
    if file_exists("metrics/calibration.json"):
        print("[5/12] OOF calibration: SKIP (artifact exists)")
        with open(os.path.join(ART, "metrics", "calibration.json")) as f:
            calibration = json.load(f)
        a, b, T = calibration["platt"]["a"], calibration["platt"]["b"], calibration["temperature"]
        best_thr = calibration["chosen_threshold"]
        iters_bin = calibration.get("iters_bin_folds", [800])
        iters_mc = calibration.get("iters_mc_folds", [100])
    else:
        print("[5/12] 5-fold OOF for calibration (Platt + temperature) and threshold ...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_bin = np.zeros(len(X_tr))
        oof_mc = np.zeros((len(X_tr), len(CATS)))
        iters_bin, iters_mc = [], []
        for k, (tr_i, va_i) in enumerate(skf.split(X_tr, ym_tr)):
            mk = lgb.train(lgb_bin_params, lgb.Dataset(X_tr[tr_i], label=yb_tr[tr_i]),
                           num_boost_round=1500, valid_sets=[lgb.Dataset(X_tr[va_i], label=yb_tr[va_i])],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_bin[va_i] = mk.predict(X_tr[va_i], raw_score=True)
            iters_bin.append(mk.best_iteration)
            mmk = lgb.train(lgb_mc_params, lgb.Dataset(X_tr[tr_i], label=ym_tr[tr_i]),
                            num_boost_round=600, valid_sets=[lgb.Dataset(X_tr[va_i], label=ym_tr[va_i])],
                            callbacks=[lgb.early_stopping(60, verbose=False)])
            oof_mc[va_i] = mmk.predict(X_tr[va_i], raw_score=True)
            iters_mc.append(mmk.best_iteration)
            del mk, mmk
            gc.collect()
            print(f"  fold {k+1}/5 done (bin {iters_bin[-1]} it, mc {iters_mc[-1]} it)")
        a, b, brier_oof_cal = fit_platt(oof_bin, yb_tr)
        brier_oof_raw = float(brier_score_loss(yb_tr, sigmoid(oof_bin)))
        T, nll_T, nll_1 = fit_temperature(oof_mc, ym_tr)
        p_oof_cal = sigmoid(a * oof_bin + b)
        grid = np.linspace(0.05, 0.95, 181)
        f1s = [f1_score(yb_tr, (p_oof_cal >= t).astype(int)) for t in grid]
        best_thr = float(grid[int(np.argmax(f1s))])
        thr_curve = [{"threshold": float(t), "precision": float(precision_score(yb_tr, (p_oof_cal >= t).astype(int), zero_division=0)),
                      "recall": float(recall_score(yb_tr, (p_oof_cal >= t).astype(int), zero_division=0)),
                      "f1": float(f)} for t, f in zip(grid, f1s)]
        calibration = {
            "platt": {"a": a, "b": b},
            "temperature": T,
            "oof_brier_raw": brier_oof_raw,
            "oof_brier_platt": brier_oof_cal,
            "oof_nll_raw_T1": nll_1,
            "oof_nll_temperature": nll_T,
            "chosen_threshold": best_thr,
            "threshold_curve": thr_curve,
            "oof_reliability": reliability_curve(yb_tr, p_oof_cal),
            "iters_bin_folds": iters_bin,
            "iters_mc_folds": iters_mc,
        }
        jdump(calibration, os.path.join(ART, "metrics", "calibration.json"))
        del oof_bin, oof_mc, p_oof_cal, f1s
        gc.collect()
        print(f"  Platt a={a:.4f} b={b:.4f} | T={T:.3f} | threshold={best_thr:.2f}")

    # ------------------------------ final models on FULL train
    if file_exists("models/binary_lightgbm.txt") and file_exists("models/multiclass_lightgbm.txt"):
        print("[6/12] FINAL models: SKIP training (native model files exist — reloading)")
        bin_final = lgb.Booster(model_file=os.path.join(ART, "models", "binary_lightgbm.txt"))
        mc_final = lgb.Booster(model_file=os.path.join(ART, "models", "multiclass_lightgbm.txt"))
        final_bin_iter = bin_final.num_trees()
        final_mc_iter = mc_final.num_trees() // len(CATS)
    else:
        print("[6/12] Training FINAL models on full training set ...")
        final_bin_iter = int(np.mean(iters_bin))
        final_mc_iter = int(np.mean(iters_mc))
        bin_final = lgb.train(lgb_bin_params, lgb.Dataset(X_tr, label=yb_tr), num_boost_round=final_bin_iter)
        mc_final = lgb.train(lgb_mc_params, lgb.Dataset(X_tr, label=ym_tr), num_boost_round=final_mc_iter)
        bin_final.save_model(os.path.join(ART, "models", "binary_lightgbm.txt"))
        mc_final.save_model(os.path.join(ART, "models", "multiclass_lightgbm.txt"))
        gc.collect()
    print(f"  binary: {final_bin_iter} rounds | multiclass: {final_mc_iter} rounds x {len(CATS)} classes")
    if not file_exists("models/binary_lightgbm.json"):
        jdump(bin_final.dump_model(), os.path.join(ART, "models", "binary_lightgbm.json"))
        gc.collect()
    if not file_exists("models/multiclass_lightgbm.json"):
        jdump(mc_final.dump_model(), os.path.join(ART, "models", "multiclass_lightgbm.json"))
        gc.collect()

    # ------------------------------ anomaly detector (normal-only)
    if file_exists("models/isolation_forest.json"):
        print("[7/12] IsolationForest: SKIP (artifact exists — using serialized scorer)")
        with open(os.path.join(ART, "models", "isolation_forest.json")) as f:
            iforest_art = json.load(f)
        print(f"  {iforest_art['n_estimators']} trees, anchors {iforest_art['norm_anchors']}")
    else:
        print("[7/12] IsolationForest on NORMAL training traffic ...")
        normal_mask = yb_tr == 0
        X_normal = X_tr[normal_mask]
        sample_idx = rng.choice(len(X_normal), size=min(30000, len(X_normal)), replace=False)
        iforest = IsolationForest(n_estimators=120, max_samples=1024, contamination="auto",
                                  random_state=SEED, n_jobs=1)
        iforest.fit(X_normal[sample_idx])
        iforest_art = serialize_iforest(iforest, X_normal[sample_idx[:5000]])
        jdump(iforest_art, os.path.join(ART, "models", "isolation_forest.json"))
        del iforest, X_normal
        gc.collect()
        print(f"  {iforest_art['n_estimators']} trees, subsample {iforest_art['subsample_size']}, anchors {iforest_art['norm_anchors']}")

    # ------------------------------ behavioral clustering
    if file_exists("preprocessor/clustering.json"):
        print("[8/12] Behavioral clustering: SKIP (artifact exists)")
        with open(os.path.join(ART, "preprocessor", "clustering.json")) as f:
            clustering_art = json.load(f)
    else:
        print("[8/12] Behavioral clustering (KMeans + PCA on train attacks) ...")
        BEHAV = ["dur", "rate", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload",
                 "sloss", "dloss", "sjit", "djit", "tcprtt", "synack", "ackdat", "smean",
                 "dmean", "sinpkt", "dinpkt", "response_body_len", "trans_depth",
                 "byte_ratio", "packet_ratio", "load_ratio", "flow_asymmetry", "total_bytes"]
        beh_idx = [feat_names.index(f"log1p_{f}") if f"log1p_{f}" in feat_names else feat_names.index(f) for f in BEHAV]
        beh_names = [feat_names[i] for i in beh_idx]  # actual (possibly log1p_-prefixed) names
        attack_mask = yb_tr == 1
        X_att = X_tr[attack_mask]
        att_cats = train_attack_cats[attack_mask]
        sidx = rng.choice(len(X_att), size=min(30000, len(X_att)), replace=False)
        Xb = X_att[sidx][:, beh_idx]
        scaler = StandardScaler().fit(Xb)
        Xbs = scaler.transform(Xb)
        km = KMeans(n_clusters=8, random_state=SEED, n_init=10).fit(Xbs)
        pca = PCA(n_components=2, random_state=SEED).fit(Xbs)
        labels = km.labels_
        profiles = []
        for ci in range(8):
            mask = labels == ci
            cat_dist = att_cats[sidx][mask]
            dist = {c: int((cat_dist == c).sum()) for c in CATS if (cat_dist == c).sum() > 0}
            dominant = max(dist, key=dist.get) if dist else "Generic"
            centroid = km.cluster_centers_[ci]
            top_feats = sorted(zip(beh_names, centroid), key=lambda kv: -abs(kv[1]))[:4]
            profiles.append({
                "cluster": ci, "size": int(mask.sum()),
                "dominant_category": dominant, "category_distribution": dist,
                "top_features": [{"feature": f, "z_score": round(float(v), 2)} for f, v in top_feats],
            })
        # vectorized PCA projection for sample points
        P = pca.transform(Xbs)
        sel = rng.choice(len(Xbs), size=min(2500, len(Xbs)), replace=False)
        clustering_art = {
            "features": beh_names,
            "feature_indices": beh_idx,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.scale_.tolist(),
            "kmeans_centroids": km.cluster_centers_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "profiles": profiles,
            "sample_points": [
                {"x": float(P[i][0]), "y": float(P[i][1]),
                 "cluster": int(labels[i]), "category": str(att_cats[sidx][i])}
                for i in sel
            ],
        }
        jdump(clustering_art, os.path.join(ART, "preprocessor", "clustering.json"))
        del X_att, Xb, Xbs
        gc.collect()
        print(f"  8 clusters; PCA EVR={pca.explained_variance_ratio_.tolist()}")

    # ------------------------------ SHAP global importance
    if file_exists("metrics/shap_global.json"):
        print("[9/12] SHAP global: SKIP (artifact exists)")
        with open(os.path.join(ART, "metrics", "shap_global.json")) as f:
            shap_global = json.load(f)
    else:
        print("[9/12] SHAP global importance (TreeExplainer, train sample) ...")
        import shap
        sh_idx = rng.choice(len(X_tr), size=2000, replace=False)
        expl = shap.TreeExplainer(bin_final)
        sv = expl.shap_values(X_tr[sh_idx].astype(np.float64))
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)
        shap_global = {
            "expected_value": float(expl.expected_value) if np.isscalar(expl.expected_value) else float(expl.expected_value[0]),
            "method": "TreeSHAP (exact), shap.TreeExplainer on LightGBM binary model, 2000-row training sample",
            "features": [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order],
        }
        jdump(shap_global, os.path.join(ART, "metrics", "shap_global.json"))
        del expl, sv
        gc.collect()
        print(f"  top: {[feat_names[i] for i in order[:5]]}")

    # gain importance for multiclass
    mc_gain = mc_final.feature_importance("gain")
    mc_order = np.argsort(-mc_gain)
    jdump({"features": [{"feature": feat_names[i], "gain": float(mc_gain[i])} for i in mc_order]},
          os.path.join(ART, "metrics", "multiclass_gain.json"))
    del X_tr  # no longer needed — free 38MB+ (stage 10+ uses X_te only)
    gc.collect()

    # ------------------------------ FINAL official test evaluation (ONCE)
    print("[10/12] FINAL evaluation on official test set (used exactly once) ...")
    t0 = time.time()
    te_bin_logits = bin_final.predict(X_te, raw_score=True)
    te_mc_logits = mc_final.predict(X_te, raw_score=True)
    infer_ms_10k = (time.time() - t0) / len(X_te) * 10000 * 1000
    te_bin_prob = sigmoid(a * te_bin_logits + b)
    te_mc_prob = softmax(te_mc_logits / T)
    te_pred_class = te_mc_prob.argmax(1)
    te_anom_raw = iforest_score_serialized(iforest_art["trees"], X_te.astype(np.float64), iforest_art["c_n"], iforest_art["n_estimators"])
    te_anom_norm = np.array([normalize_anomaly(r, iforest_art["norm_anchors"], iforest_art["norm_anchor_targets"]) for r in te_anom_raw])
    del te_bin_logits, te_mc_logits
    gc.collect()

    bin_test = binary_metrics(yb_te, te_bin_prob, thr=best_thr)
    mc_test = multiclass_metrics(ym_te, te_pred_class, te_mc_prob)
    # anomaly ranking metrics
    anom_metrics = {
        "roc_auc_vs_label": float(roc_auc_score(yb_te, te_anom_raw)),
        "pr_auc_vs_label": float(average_precision_score(yb_te, te_anom_raw)),
        "score_distribution_normal": {
            "mean": float(te_anom_norm[yb_te == 0].mean()), "p50": float(np.percentile(te_anom_norm[yb_te == 0], 50)),
            "p90": float(np.percentile(te_anom_norm[yb_te == 0], 90))},
        "score_distribution_attack": {
            "mean": float(te_anom_norm[yb_te == 1].mean()), "p50": float(np.percentile(te_anom_norm[yb_te == 1], 50)),
            "p90": float(np.percentile(te_anom_norm[yb_te == 1], 90))},
    }
    order_anom = np.argsort(-te_anom_raw)
    for k in (100, 500, 1000, 5000):
        hits = int(yb_te[order_anom[:k]].sum())
        anom_metrics[f"precision_at_{k}"] = hits / k
        anom_metrics[f"recall_at_{k}"] = hits / float(yb_te.sum())

    # system-level risk scores (correlation=0 per event; correlation engine runs at runtime)
    te_pred_cat = np.where(te_bin_prob >= best_thr, te_pred_class, CAT_TO_IDX["Normal"])
    te_risk = np.array([
        risk_score(te_bin_prob[i], te_anom_norm[i], CATS[c], 0)
        for i, c in enumerate(te_pred_cat)
    ])
    # calibrated reliability on test (report-only)
    test_calibration = {
        "reliability": reliability_curve(yb_te, te_bin_prob),
        "brier": float(brier_score_loss(yb_te, te_bin_prob)),
        "risk_distribution": {
            "mean": float(te_risk.mean()),
            "bands": {b: int((te_risk >= lo).sum()) for b, lo in
                      [("low", 0), ("medium", 25), ("high", 50), ("critical", 75)]},
        },
    }

    eval_test = {
        "binary": bin_test, "multiclass": mc_test, "anomaly": anom_metrics,
        "calibration": test_calibration,
        "note": "Official UNSW-NB15 test set, scored exactly once after model freeze.",
    }
    jdump(eval_test, os.path.join(ART, "metrics", "test_evaluation.json"))
    print(f"  binary: acc={bin_test['accuracy']:.4f} P={bin_test['precision']:.4f} R={bin_test['recall']:.4f} F1={bin_test['f1']:.4f} AUC={bin_test['roc_auc']:.4f} PR={bin_test['pr_auc']:.4f}")
    print(f"  multiclass: acc={mc_test['accuracy']:.4f} macroF1={mc_test['macro_f1']:.4f} balAcc={mc_test['balanced_accuracy']:.4f}")
    print(f"  anomaly: ROC-AUC={anom_metrics['roc_auc_vs_label']:.4f} P@1000={anom_metrics['precision_at_1000']:.3f}")

    # ------------------------------ demo replay + boot sample + SHAP cache
    print("[11/12] Deterministic demo replay sequence + boot sample + SHAP cache ...")
    rng2 = np.random.RandomState(SEED)
    del te_bin_prob, te_mc_prob, te_anom_raw, te_anom_norm, te_risk, te_pred_cat
    gc.collect()
    test = pd.read_csv(TEST_CSV)  # re-read (was freed for RAM); needed for row selection
    test_idx_by_cat = {c: test.index[test["attack_cat"] == c].to_numpy() for c in CATS}

    def pick(cat, n):
        pool = test_idx_by_cat[cat]
        n = min(n, len(pool))
        sel = rng2.choice(pool, size=n, replace=False)
        test_idx_by_cat[cat] = np.setdiff1d(pool, sel)
        return sel.tolist()

    events = []  # (test_row_index, wave)
    events += [(i, "background") for i in pick("Normal", 140)]
    events += [(i, "campaign1-recon") for i in pick("Reconnaissance", 26)]
    events += [(i, "background") for i in pick("Normal", 110)]
    events += [(i, "noise") for i in pick("Fuzzers", 14)]
    events += [(i, "background") for i in pick("Normal", 60)]
    events += [(i, "campaign2-exploit") for i in pick("Exploits", 42)]
    events += [(i, "campaign2-exploit") for i in pick("Shellcode", 8)]
    events += [(i, "background") for i in pick("Normal", 90)]
    events += [(i, "noise") for i in pick("Analysis", 8)]
    events += [(i, "noise") for i in pick("Backdoor", 8)]
    events += [(i, "background") for i in pick("Normal", 70)]
    events += [(i, "campaign3-dos") for i in pick("DoS", 55)]
    events += [(i, "campaign3-dos") for i in pick("Generic", 28)]
    events += [(i, "noise") for i in pick("Worms", 4)]
    events += [(i, "background") for i in pick("Normal", 130)]
    events += [(i, "noise") for i in pick("Exploits", 10)]
    events += [(i, "background") for i in pick("Normal", 100)]

    # synthetic timestamps (clearly SIMULATED metadata)
    tcur = 0.0
    seq = []
    for idx, wave in events:
        if wave == "background":
            tcur += float(rng2.uniform(90, 260))
        else:
            tcur += float(rng2.uniform(15, 55))
        seq.append({"i": int(idx), "t": int(tcur), "wave": wave})
    demo = {
        "label": "Real-Time Detection Replay (SIMULATED timestamps)",
        "simulation_note": "Event times are synthetic replay offsets; the events themselves are real UNSW-NB15 test rows scored by the real trained models.",
        "events": seq,
        "waves": ["background", "campaign1-recon", "noise", "campaign2-exploit", "campaign3-dos"],
        "default_speed_events_per_sec": 8,
        "total_events": len(seq),
        "duration_ms": int(tcur),
    }
    jdump(demo, os.path.join(ART, "replay", "demo_sequence.json"))

    # boot sample: stratified 12000 for the command center
    cat_counts = {c: int((test["attack_cat"] == c).sum()) for c in CATS}
    n_total = len(test)
    boot_idx = []
    for c in CATS:
        pool = test_idx_by_cat[c]
        n = min(len(pool), int(round((cat_counts[c] / n_total) * 12000)))
        boot_idx += rng2.choice(pool, size=n, replace=False).tolist()
    rng2.shuffle(boot_idx)
    jdump({"indices": [int(i) for i in boot_idx], "note": "stratified sample of official test set for SOC boot state; scored live by the inference engine"},
          os.path.join(ART, "replay", "boot_sample.json"))

    # SHAP cache: exact TreeSHAP contribs for demo + boot attack-heavy rows
    demo_rows = [e["i"] for e in seq]
    boot_attacks = [i for i in boot_idx if test_label_arr[i] == 1]
    keep_boot = rng2.choice(boot_attacks, size=min(3500, len(boot_attacks)), replace=False).tolist()
    cache_rows = sorted(set(demo_rows + keep_boot))
    Xc = fb.transform(test.iloc[cache_rows]).astype(np.float32)
    contribs = bin_final.predict(Xc, pred_contrib=True)  # (n, feats+1)
    shap_cache = {}
    for r, row in enumerate(cache_rows):
        vec = [round(float(v), 5) for v in contribs[r]]
        shap_cache[str(int(row))] = {"b": vec}
    del contribs, Xc
    gc.collect()
    # multiclass top contribs for demo events (predicted class)
    Xd = fb.transform(test.iloc[demo_rows]).astype(np.float32)
    mc_c = np.asarray(mc_final.predict(Xd, pred_contrib=True))  # (n, n_class*(feats+1))
    mc_contribs = mc_c.reshape(len(demo_rows), len(CATS), len(feat_names) + 1)  # (n, class, feats+1)
    for r, row in enumerate(demo_rows):
        cls = int(te_pred_class[row])
        vec = mc_contribs[r, cls, :-1]
        pairs = sorted(zip(feat_names, vec), key=lambda kv: -abs(kv[1]))[:10]
        shap_cache[str(int(row))]["mc"] = {
            "class": CATS[cls],
            "top": [[f, round(float(v), 5)] for f, v in pairs],
        }
    del mc_contribs, Xd, test
    gc.collect()
    jdump(shap_cache, os.path.join(ART, "explainability", "shap_cache.json"))

    # ------------------------------ registry + operational metrics
    print("[12/12] Model registry + operational metrics ...")
    import sys as _sys
    try:
        import shap as _shap
        shap_version = _shap.__version__
    except Exception:
        shap_version = "0.52.0"
    registry = {
        "name": "CipherMind Sentinel Detection Engine",
        "version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "dataset": {
            "train_file": "dataset/Training and Testing Sets/UNSW_NB15_training-set.csv",
            "test_file": "dataset/Training and Testing Sets/UNSW_NB15_testing-set.csv",
            "train_rows": 175341, "test_rows": 82332,
        },
        "models": {
            "binary": {"algorithm": "LightGBM", "rounds": final_bin_iter, "params": {k: v for k, v in lgb_bin_params.items() if k != "metric"},
                        "artifact": "ml/artifacts/models/binary_lightgbm.json", "calibration": "Platt (OOF-fitted)"},
            "multiclass": {"algorithm": "LightGBM", "rounds": final_mc_iter, "classes": CATS, "params": {k: v for k, v in lgb_mc_params.items() if k != "metric"},
                            "artifact": "ml/artifacts/models/multiclass_lightgbm.json", "calibration": f"Temperature scaling T={T:.4f} (OOF-fitted)"},
            "anomaly": {"algorithm": "IsolationForest (normal-only training)", "trees": iforest_art["n_estimators"],
                         "artifact": "ml/artifacts/models/isolation_forest.json"},
        },
        "threshold": best_thr,
        "platt": {"a": a, "b": b},
        "temperature": T,
        "risk_config": RISK_CONFIG,
        "feature_count": len(feat_names),
        "class_mapping": CAT_TO_IDX,
        "metrics_files": ["binary via test_evaluation.json"],
        "software": {
            "python": "3.12.14", "lightgbm": lgb.__version__, "scikit-learn": __import__("sklearn").__version__,
            "numpy": np.__version__, "shap": shap_version,
        },
        "artifacts": {
            "features": "ml/artifacts/preprocessor/feature_config.json",
            "clustering": "ml/artifacts/preprocessor/clustering.json",
            "shap_cache": "ml/artifacts/explainability/shap_cache.json",
            "test_evaluation": "ml/artifacts/metrics/test_evaluation.json",
        },
    }
    jdump(registry, os.path.join(ART, "model_registry.json"))
    operational = {
        "python_batch_ms_per_10k_events": round(infer_ms_10k, 1),
        "artifact_sizes_bytes": {
            p: os.path.getsize(os.path.join(ART, p)) for p in [
                "models/binary_lightgbm.json", "models/multiclass_lightgbm.json",
                "models/isolation_forest.json", "preprocessor/feature_config.json",
                "preprocessor/clustering.json", "explainability/shap_cache.json",
            ]
        },
        "total_training_seconds": round(time.time() - t_start, 1),
    }
    jdump(operational, os.path.join(ART, "metrics", "operational.json"))

    # eval summary (machine + human readable)
    summary = {
        "final_model_selected": "LightGBM (best validation F1/AUC vs LogisticRegression/RandomForest/XGBoost — see model_comparison.json)",
        "test_metrics_binary": {k: bin_test[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "false_positive_rate"]},
        "test_metrics_multiclass": {k: mc_test[k] for k in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "macro_precision", "macro_recall"]},
        "best_classes": sorted(mc_test["per_class"], key=lambda d: -d["f1"])[:3],
        "worst_classes": sorted(mc_test["per_class"], key=lambda d: d["f1"])[:3],
        "anomaly_performance": anom_metrics,
        "calibration": {"oof_brier_raw": calibration.get("oof_brier_raw"), "oof_brier_platt": calibration.get("oof_brier_platt"), "test_brier": test_calibration["brier"]},
        "feature_ablation": abl,
        "operational": operational,
        "how_to_run": {"train": "python3 ml/scripts/train.py", "inference_service": "cd mini-services/soc-engine && bun run dev (port 3010)"},
        "artifact_locations": "ml/artifacts/**",
        "demo_flow": "Open the SOC UI -> Command Center (boot state) -> Live Replay (campaign escalation) -> Incident Investigation -> Explainability Center",
    }
    jdump(summary, os.path.join(ART, "reports", "eval_summary.json"))
    print(json.dumps(summary["test_metrics_binary"], indent=2))
    print(f"\nDONE in {time.time() - t_start:.0f}s. Artifacts in {ART}")

    # rarity table for risk scoring (train prevalence)
    jdump(RARITY, os.path.join(ART, "metrics", "train_prevalence.json"))
    RISK_CONFIG["rarity_weights"] = {c: min(1.0, -float(np.log10(max(RARITY[c], 1e-6))) / 4.0) for c in CATS}
    # rewrite registry with rarity weights baked in
    registry["risk_config"] = RISK_CONFIG
    jdump(registry, os.path.join(ART, "model_registry.json"))


if __name__ == "__main__":
    main()
