# Incident Correlation

How a deliberately noisy alert stream (binary FPR 27.85% on test — see [EVALUATION.md](EVALUATION.md)) becomes a small, ranked incident list. Implementation: `mini-services/soc-engine/src/correlate.ts` (streaming, in-memory, per-session).

## From alerts to incidents

Only events with binary verdict **Attack** enter correlation. Each incoming alert is matched against **open incidents** (last activity within a 45-second sliding window of simulated time; incidents older than the window are closed — status Active→Closed, Escalating→Contained). For each candidate incident a **matching score** is computed:

```
s = (same_family ? 1.0 : 0.15)          # predicted-category family
  + entity_overlap × 0.5                 # shared simulated pseudo-entity
  − time_penalty × 0.35                  # time_penalty = Δt / 45s window
  − behavioral_dist / 42                 # Euclidean distance on standardized
                                         # 26-dim behavioral subspace ÷ threshold
```

The best-scoring incident wins if `s > 0.45`; otherwise a new incident is born (`INC-0001…` for the boot state, `RINC-0001…` for replay sessions — prefixes avoid id collisions). Behavioral distance uses the same standardized feature subspace as the KMeans clustering (clustering artifact's `feature_indices`, scaler stats from training attacks); distance ≥ 42 (the threshold) cancels the family bonus entirely, so a "same-category" event with alien flow behavior still starts its own incident.

**Category families** (allow multi-stage campaigns to merge into one incident): Exploits/Shellcode/Backdoor · Reconnaissance/Analysis · DoS/Generic · Fuzzers/Analysis. A family transition *plus* higher risk flips the incident to status **Escalating** and may shift its dominant category — that's how the replay's recon→exploit progression shows up as one escalating story.

**Caps:** max 400 retained events per incident (ring buffer) and **80 alerts** per incident (`MAX_ALERTS_PER_INCIDENT`) so continuous bursts chain into sequential incidents instead of one mega-incident.

## Incident risk

Every accepted alert is re-scored by Model D with `related_alerts` = the incident's current alert count (the correlation term, see [THREAT_SCORING.md](THREAT_SCORING.md)); incident risk = the **max** such risk over its lifetime, and the trajectory is kept (last 200 points) and rendered as a sparkline/line chart.

## Attack stories (with epistemics)

When an incident is finalized (replay end, or on demand), a 5-stage story is built, each stage tagged with its epistemic status:

| Stage | Trigger | Epistemics |
|---|---|---|
| 1. Initial suspicious behavior | first alert | **Observed** |
| 2. Repeated anomalous behavior | ≥3 events with anomaly ≥ 40 | **Observed** |
| 3. Attack pattern transition | category change within incident | **Inferred** |
| 4. Incident peak | highest-risk event | **Observed** |
| 5. Recommended response | always | **Prescriptive** |

Definitions surfaced in the UI: **Observed** = directly computed from model outputs on real flow records; **Inferred** = interpretation of observed patterns (hedged language); **Prescriptive** = recommended actions, not facts. Every stage links to its evidence event IDs (stage 5 has none — it's advice). Per-category **containment playbooks** are attached (Exploits: isolate + patch review + persistence hunt; Worms: immediate containment + patch propagation vector; Recon: rate-limit + exposure verification + escalation-to-exploitation watch; etc. — full map in `correlate.ts::playbookFor`).

## Simulated metadata — full disclosure

The correlation signals that look "network-ish" are synthetic, and the product says so (violet SIM tags, prompts, fallback copy):

- **Timestamps** come from replay offsets generated at training time (`demo_sequence.json`) or boot-time burst/gap heuristics — the dataset contains **no timestamps**.
- **`ENT-*` entities** are deterministic FNV-1a hashes of flow attributes (`proto|service|sttl≫2|ct_dst_src_ltm|ct_srv_src|dttl≫2`) mapped into ENT-1000…ENT-9999 — the dataset contains **no IPs/users/devices**, so "entity overlap" measures feature-level similarity, not a shared host.

Correlation therefore runs on **model outputs + flow features only** — predicted category, temporal proximity (in simulated time), behavioral distance, and pseudo-entity overlap — and never on invented ground truth.
