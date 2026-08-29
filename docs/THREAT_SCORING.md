# Threat Scoring (Model D)

Source of truth: `ml/artifacts/model_registry.json` → `risk_config` (identical implementation in `ml/scripts/train.py::risk_score` and `mini-services/soc-engine/src/engine.ts::riskScore`).

## Formula

```
risk = 32·attack_confidence            # p = Platt-calibrated Model A probability
     + 18·(anomaly / 100)              # Model C normalized 0–100
     + 20·category_severity            # lookup below (predicted category)
     +  8·rarity                        # min(1, −log10(train_prevalence) / 4)
     + 10·uncertainty                   # 2·p·(1−p)·2   (max 1.0 at p = 0.5)
     + 12·correlation                   # min(1, related_alerts / 8)

risk = clamp(risk, 0, 100)
```

**Category severity** (`category_severity`): Exploits 1.0 · Shellcode 0.95 · Worms 0.9 · Backdoor 0.85 · DoS 0.8 · Analysis 0.6 · Reconnaissance 0.55 · Fuzzers 0.45 · Generic 0.4 · Normal 0.05.

**Rarity** is precomputed per category from **train** prevalence and baked into the registry (`rarity_weights`): Worms 0.7825 (prevalence 0.00074) · Shellcode 0.5474 · Backdoor 0.5005 · Analysis 0.4857 · Reconnaissance 0.3058 · DoS 0.2888 · Fuzzers 0.2460 · Exploits 0.1801 · Generic 0.1605 · Normal 0.1239. Rarer predicted attacks contribute more risk.

## Severity bands

| Band | Range |
|---|---|
| Low | 0–24 |
| Medium | 25–49 |
| High | 50–74 |
| Critical | 75–100 |

## Worked example

Event with calibrated attack probability **p = 0.95**, anomaly score **70**, predicted category **Exploits**, and **5 related alerts** in its incident:

| Term | Arithmetic | Points |
|---|---|---:|
| attack_confidence | 32 × 0.95 | 30.40 |
| anomaly | 18 × (70/100) | 12.60 |
| category_severity | 20 × 1.00 (Exploits) | 20.00 |
| rarity | 8 × 0.1801 (−log10(0.19045)/4) | 1.44 |
| uncertainty | 10 × (2 × 0.95 × 0.05 × 2) = 10 × 0.19 | 1.90 |
| correlation | 12 × (5/8) | 7.50 |
| **Total** | 30.40 + 12.60 + 20.00 + 1.44 + 1.90 + 7.50 | **73.84** |

**Risk = 73.8 → High** (a 6th correlated alert would add 12 × (6/8 − 5/8) = +1.5 → 75.3 → Critical).

Note what the formula does: high confidence and a dangerous category dominate; uncertainty *adds* a little risk (an uncertain 0.5-probability event gets the full 10 points, reflecting "could go either way — look at it"), and growing correlation raises risk as an incident escalates.

## Incident-level risk

The correlation engine rescored each alert with its current `related_alerts` count; incident risk = **max event risk** over its lifetime (with the correlation boost applied per event as the incident grows). Event-level risk in the boot state uses `related_alerts = 0`; the correlation term only activates once events group into incidents. As a system-level reference (train.py, per-event with correlation = 0), the risk distribution over all 82,332 test events has mean 38.4 with cumulative counts — ≥25: 57,552 · ≥50: 34,034 · ≥75: 0 — reported for transparency in `test_evaluation.json.calibration.risk_distribution` (no event reaches the Critical band without the correlation term).

## Configurability & rationale

Every weight, severity value, band edge, and the saturation constant (8) lives in `model_registry.json.risk_config` — editable without retraining, and consumed identically by the Python evaluator and the TS engine. Design rationale: a **transparent, auditable** score beats a learned one here because (a) there are no ground-truth "risk" labels to learn from — any learned score would be trained on invented targets; (b) analysts can interrogate each point ("why 73.8?") down to the exact term; and (c) SOC teams inevitably re-tune priorities, which is a JSON edit, not a retraining run.
