#!/usr/bin/env python3
"""Cross-validation: Python model predictions vs soc-engine (TypeScript) predictions.

Verifies the TS tree-walking engine reproduces the trained LightGBM models
bit-for-bit (within float tolerance) on real test rows.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "/home/z/my-project"
sys.path.insert(0, f"{ROOT}/ml/training")
from features import FeatureBuilder  # noqa: E402

TEST_CSV = f"{ROOT}/dataset/Training and Testing Sets/UNSW_NB15_testing-set.csv"
ART = f"{ROOT}/ml/artifacts"

test = pd.read_csv(TEST_CSV)
fb = FeatureBuilder().fit(pd.read_csv(f"{ROOT}/dataset/Training and Testing Sets/UNSW_NB15_training-set.csv"))

reg = json.load(open(f"{ART}/model_registry.json"))
a, b, T = reg["platt"]["a"], reg["platt"]["b"], reg["temperature"]
thr = reg["threshold"]

bin_m = lgb.Booster(model_file=f"{ART}/models/binary_lightgbm.txt")
mc_m = lgb.Booster(model_file=f"{ART}/models/multiclass_lightgbm.txt")
CATS = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
        "Normal", "Reconnaissance", "Shellcode", "Worms"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z, t=1.0):
    z = z / t
    e = np.exp(z - z.max())
    return e / e.sum()

# pick deterministic diverse rows: 4 normals + 1 of each attack category
rows = []
for cat in CATS:
    idx = test.index[test["attack_cat"] == cat]
    rows.append(int(idx[7]))
print(f"validating rows: {rows}")

X = fb.transform(test.iloc[rows]).astype(np.float32)
bin_logit = bin_m.predict(X, raw_score=True)
mc_logit = mc_m.predict(X, raw_score=True)

events = []
for r in rows:
    events.append({k: (v if isinstance(v, str) else (int(v) if float(v).is_integer() else float(v)))
                   for k, v in test.iloc[r].to_dict().items()})

req = json.dumps({"events": events})
out = subprocess.run(
    ["curl", "-s", "-m", "60", "-X", "POST", "-H", "Content-Type: application/json",
     "-d", req, "http://localhost:3010/api/predict/batch"],
    capture_output=True, text=True,
)
resp = json.loads(out.stdout)
if "events" not in resp:
    print("TS response error:", out.stdout[:400]); sys.exit(1)

ok = True
for i, (r, ev) in enumerate(zip(rows, resp["events"])):
    py_prob = float(sigmoid(a * bin_logit[i] + b))
    py_mc = softmax(mc_logit[i], T)
    py_cat = CATS[int(np.argmax(py_mc))]
    py_verdict = "Attack" if py_prob >= thr else "Normal"
    ts_prob = ev["attackProbability"]
    ts_cat = ev["category"]
    ts_verdict = ev["binaryVerdict"]
    match = abs(py_prob - ts_prob) < 1e-4 and py_cat == (ts_cat if ts_verdict == "Attack" else "Normal") and py_verdict == ts_verdict
    gt = test.iloc[r]["attack_cat"]
    anom = ev["anomalyScore"]
    print(f"row {r:6d} gt={gt:14s} py={py_prob:.4f}/{py_verdict:6s}/{py_cat:14s} ts={ts_prob:.4f}/{ts_verdict:6s}/{ts_cat:14s} anom={anom:5.1f} risk={ev['riskScore']:5.1f} {'OK' if match else 'MISMATCH'}")
    if not match:
        ok = False
print("\nRESULT:", "ALL MATCH — TS engine reproduces Python models" if ok else "MISMATCHES FOUND")
sys.exit(0 if ok else 1)
