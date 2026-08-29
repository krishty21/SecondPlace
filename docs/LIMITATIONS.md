# Limitations (honest)

Every number below comes from `ml/artifacts/metrics/test_evaluation.json` or sibling artifacts. We consider stating these clearly a feature, not an apology.

## Dataset-level

- **No network identity**: the supplied 45-column CSVs contain no `srcip/sport/dstip/dsport`, no timestamps, no users, no devices. All timestamps and `ENT-*` entities in the product are **clearly-labeled simulations** (replay offsets; deterministic feature hashes); no real host, user, or geography claims are made anywhere.
- **UNSW-NB15 is synthetic lab traffic from 2015** (IXIA PerfectStorm generator + real normal captures). It is **not** phishing, malware-file, ransomware, or enterprise telemetry; nothing here demonstrates capability on those.
- `id` numbers from 1 independently per file (82,332 colliding ids) — excluded from features; two unseen test `state` values (ACC, CLO) are absorbed by the unknown-bucket encoder rather than modeled.
- Extreme class imbalance (Worms: 130 train rows) and a train→test shift (attack ratio 68.1% → 55.1%).

## Model-level

- **Rare-class weakness** (multiclass, official test): Backdoor F1 **0.049**, Analysis F1 **0.074**, DoS F1 **0.183**; Worms 0.541 on 44 support rows; Shellcode 0.515. These classes are effectively unreliable for attribution.
- **Exploits↔DoS confusion**: 2,543 of 4,089 true DoS rows predicted Exploits (DoS recall 0.107) — the families share flow signatures; categories are "behavior family", not ground truth.
- **Recall-heavy operating point**: threshold 0.46 yields recall 0.9844 but **FPR 27.85%** on test normals (specificity 0.0156). Deliberate for SOC triage — correlation + risk scoring deprioritize — but the alert stream is noisy by design.
- **Calibration good, not perfect**: OOF Brier 0.0263 → 0.0261 (Platt), but **test Brier 0.0856** — a visible train→test calibration gap from distribution shift. Multiclass temperature (T = 1.0035) barely moved NLL; class probabilities remain sharpest where data is plentiful.
- **Anomaly detector is moderate** (ROC-AUC **0.7963**): excellent top-k triage ranking (precision@1000 = 0.981), not a standalone detector.
- **Layered-model edge case**: A (binary) and B (multiclass) are independent; a flow can be flagged Attack while B's argmax is Normal, and the UI then shows category "Normal" on an alert.

## Product-level

- **Research prototype**: single-tenant, no authentication, no persistence; soc-engine state is in-memory and rebuilt on restart.
- **Replay, not a feed**: Live Replay streams a deterministic 903-event demo sequence over socket.io — no packet capture, syslog, or NetFlow connector exists.
- **LLM dependency with a safety net**: summaries/chat use the z-ai SDK server-side with strictly evidence-grounded prompts (no invented IPs/CVEs/users; simulated metadata disclosed); on any LLM failure the routes fall back to a deterministic template summary (badged `source: fallback`).
- **TS engine verification depth**: cross-validated against Python on **10 diverse rows** (all matched to 4 decimal places). Full-test equivalence is strongly expected (identical tree JSON, identical transforms, verified identities) but not exhaustively asserted row-by-row.

## What production would need

1. Real telemetry with true identity + time (or honest enrichment), retention, audit logging.
2. An FPR-conscious operating point (per-deployment threshold tuning, possibly cost-sensitive learning) plus feedback loops from analyst verdicts.
3. Rare-class remediation: more representative data (or anomaly-only handling of ultra-rare families), per-class thresholds, human-in-the-loop labels.
4. AuthN/AuthZ, multi-tenancy, persistence, horizontal scaling, streaming ingestion, model versioning/rollback, drift monitoring on calibration and class mix.
5. Correlation validated against real incident ground truth — current parameters (45 s window, 0.45 threshold, families, 80-alert cap) are reasoned, not learned or tuned on labeled incidents.
