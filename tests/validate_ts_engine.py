#!/usr/bin/env python3
"""Cross-validation: Python model predictions vs soc-engine (TypeScript) predictions.

Verifies that the TypeScript inference engine reproduces the trained LightGBM
models bit-for-bit (within float tolerance) on real test rows. This test uses
ONLY the shipped artifacts — the exported `feature_config.json` pipeline and
the native `.txt` model files — exactly like the production engine does
(no Python-side training code is involved).

Usage (from the repository root, with the soc-engine running on :3010):

    python3 tests/validate_ts_engine.py             # default engine http://localhost:3010
    ENGINE_URL=http://host:3010 python3 tests/validate_ts_engine.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "dataset" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"
ART = ROOT / "ml" / "artifacts"
ENGINE_URL = os.environ.get("ENGINE_URL", "http://localhost:3010").rstrip("/")

CATS = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
        "Normal", "Reconnaissance", "Shellcode", "Worms"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_feature_config(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Apply the exported feature pipeline (identical semantics to the TS engine).

    Uses ONLY the statistics stored in feature_config.json — the same contract
    production runs on. Order: raw numerics (with log1p on configured cols)
    -> derived ratios -> categorical ordinals.
    """
    eps = cfg["eps"]
    raw = cfg["raw_numeric"]
    medians = cfg["medians"]
    log_cols = set(cfg["log_cols"])
    cat_maps = cfg["cat_maps"]
    unknown = cfg["unknown_code"]
    derived_names = [d["name"] for d in cfg["derived"]]

    num = df[raw].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    num = num.fillna(pd.Series({c: medians[c] for c in raw}))
    for c in log_cols:
        num[c] = np.log1p(num[c].clip(lower=0))

    der = pd.DataFrame(index=df.index)
    sbytes = df["sbytes"].astype(float); dbytes = df["dbytes"].astype(float)
    spkts = df["spkts"].astype(float);  dpkts = df["dpkts"].astype(float)
    sload = df["sload"].astype(float);  dload = df["dload"].astype(float)
    if "byte_ratio" in derived_names:
        der["byte_ratio"] = sbytes / (dbytes + eps)
    if "packet_ratio" in derived_names:
        der["packet_ratio"] = spkts / (dpkts + eps)
    if "total_bytes" in derived_names:
        der["total_bytes"] = sbytes + dbytes
    if "total_packets" in derived_names:
        der["total_packets"] = spkts + dpkts
    if "total_loss" in derived_names:
        der["total_loss"] = df["sloss"].astype(float) + df["dloss"].astype(float)
    if "payload_per_packet" in derived_names:
        der["payload_per_packet"] = (sbytes + dbytes) / (spkts + dpkts + eps)
    if "load_ratio" in derived_names:
        der["load_ratio"] = sload / (dload + eps)
    if "size_ratio" in derived_names:
        der["size_ratio"] = df["smean"].astype(float) / (df["dmean"].astype(float) + eps)
    if "rtt_ratio" in derived_names:
        der["rtt_ratio"] = df["synack"].astype(float) / (df["ackdat"].astype(float) + eps)
    if "jitter_ratio" in derived_names:
        der["jitter_ratio"] = df["sjit"].astype(float) / (df["djit"].astype(float) + eps)
    if "interpkt_ratio" in derived_names:
        der["interpkt_ratio"] = df["sinpkt"].astype(float) / (df["dinpkt"].astype(float) + eps)
    if "flow_asymmetry" in derived_names:
        der["flow_asymmetry"] = (sbytes - dbytes) / (sbytes + dbytes + eps)
    der = der.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cat_cols = []
    for c in cfg["categorical"]:
        cat_cols.append(df[c].astype(str).map(cat_maps[c]).fillna(unknown[c]).astype(int).values)

    mat = np.column_stack(
        [num[c].values for c in raw]
        + [der[n].values for n in derived_names]
        + cat_cols
    )
    expected = cfg["feature_names"]
    rebuilt = ([f"log1p_{c}" if c in log_cols else c for c in raw]
               + derived_names + [f"cat_{c}" for c in cfg["categorical"]])
    assert rebuilt == expected, f"feature order mismatch vs feature_config: {set(rebuilt) ^ set(expected)}"
    return mat.astype(np.float64)


def main() -> int:
    test = pd.read_csv(TEST_CSV)
    cfg = json.load(open(ART / "preprocessor" / "feature_config.json"))
    reg = json.load(open(ART / "metadata" / "model_registry.json"))
    a, b, T, thr = reg["platt"]["a"], reg["platt"]["b"], reg["temperature"], reg["threshold"]

    bin_m = lgb.Booster(model_file=str(ART / "models" / "binary_lightgbm.txt"))
    mc_m = lgb.Booster(model_file=str(ART / "models" / "multiclass_lightgbm.txt"))

    # deterministic diverse rows: 4 normals + 1 of each attack category
    rows = []
    normal_idx = test.index[test["attack_cat"] == "Normal"]
    for i in normal_idx[3:7]:
        rows.append(int(i))
    for cat in CATS:
        if cat == "Normal":
            continue
        idx = test.index[test["attack_cat"] == cat]
        rows.append(int(idx[7]))
    print(f"engine: {ENGINE_URL} | validating rows: {rows}")

    X = apply_feature_config(test.iloc[rows], cfg).astype(np.float32)
    bin_logit = bin_m.predict(X, raw_score=True)
    mc_logit = mc_m.predict(X, raw_score=True)

    events = []
    for r in rows:
        ev = {}
        for k, v in test.iloc[r].to_dict().items():
            if isinstance(v, str):
                ev[k] = v
            elif float(v).is_integer():
                ev[k] = int(v)
            else:
                ev[k] = float(v)
        events.append(ev)

    req = json.dumps({"events": events}).encode()
    http_req = urllib.request.Request(
        f"{ENGINE_URL}/api/predict/batch", data=req,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(http_req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    if "events" not in payload:
        print("TS response error:", str(payload)[:400])
        return 1

    ok = True
    for i, (r, ev) in enumerate(zip(rows, payload["events"])):
        py_prob = float(sigmoid(a * bin_logit[i] + b))
        py_mc = softmax(mc_logit[i], T)
        py_cat = CATS[int(np.argmax(py_mc))]
        py_verdict = "Attack" if py_prob >= thr else "Normal"
        ts_prob = ev["attackProbability"]
        ts_cat = ev["category"]
        ts_verdict = ev["binaryVerdict"]
        match = (abs(py_prob - ts_prob) < 1e-4
                 and py_cat == (ts_cat if ts_verdict == "Attack" else "Normal")
                 and py_verdict == ts_verdict)
        gt = test.iloc[r]["attack_cat"]
        print(f"row {r:6d} gt={gt:14s} py={py_prob:.4f}/{py_verdict:6s}/{py_cat:14s} "
              f"ts={ts_prob:.4f}/{ts_verdict:6s}/{ts_cat:14s} anom={ev['anomalyScore']:5.1f} "
              f"risk={ev['riskScore']:5.1f} {'OK' if match else 'MISMATCH'}")
        if not match:
            ok = False

    print("\nRESULT:", "ALL MATCH — the TypeScript engine reproduces the Python models"
          if ok else "MISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
