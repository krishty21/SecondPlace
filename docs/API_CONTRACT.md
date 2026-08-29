# SOC Engine API Contract (port 3010)

Base URL (from the Next.js frontend): `/api/...?XTransformPort=3010`
Socket.io: `io("/?XTransformPort=3010")` — path MUST be `/`.

All responses are JSON. All model outputs are computed LIVE by the TypeScript inference engine from trained artifacts (LightGBM binary/multiclass, IsolationForest, risk engine, correlation engine).

---

## GET /api/health
```json
{ "status": "ok", "service": "soc-engine", "modelsLoaded": true, "version": "1.0.0",
  "trainedAt": "...", "uptimeSec": 12, "incidentsTracked": 42,
  "engine": { "bootScoringMs": 8123, "eventsPerSec": 1478, "singleEventLatencyMs": 0.42 } }
```

## GET /api/dashboard
Command Center boot state (12,000 stratified test events scored live at boot).
```json
{
  "kpis": { "totalEvents": 12000, "totalAlerts": 5412, "activeIncidents": 38, "criticalIncidents": 9,
            "detectionRate": 0.451, "highRiskTrend": 3.2, "falsePositiveIndicator": 0.031,
            "meanAnomaly": 47.2, "medianResponseRisk": 61.0 },
  "categoryBreakdown": [{ "category": "Exploits", "count": 1204, "meanRisk": 71.3 }, ...],
  "severityBreakdown": [{ "severity": "Critical", "count": 412 }, ...],
  "timeline": [{ "t": 0, "events": 210, "alerts": 95, "incidents": 3, "meanRisk": 44.1 }, ...60 buckets],
  "topIncidents": [Incident...12],          // see Incident shape below (slimmed: last 6 events)
  "recentCritical": [ScoredEvent...8],       // high/critical alerts, newest first
  "generatedAt": "ISO", "sampleDescription": "...",
  "engineStats": { "bootScoringMs": 8123, "eventsPerSec": 1478, "singleEventLatencyMs": 0.42 }
}
```

## GET /api/incidents?limit=50&severity=Critical
`{ "total": 41, "incidents": [Incident...] }`

## GET /api/incidents/:id
`{ "incident": Incident (full), "groundTruthMix": {"Exploits": 31, ...} }` (ground truth shown for demo transparency)

### Incident shape
```json
{
  "incidentId": "INC-0007", "title": "Exploits activity pattern", "status": "Escalating|Active|Contained|Closed",
  "category": "Exploits", "categoryMix": {"Exploits": 28, "Shellcode": 2},
  "firstSeen": 120340, "lastSeen": 258999,           // simulated replay ms
  "eventCount": 30, "alertCount": 30, "entities": ["ENT-1042", "ENT-3391"],  // simulated pseudo-entities
  "riskScore": 86.4, "riskTrajectory": [{"t":..., "risk":..., "count":...}],
  "severity": "Critical", "meanConfidence": 0.93, "meanAnomaly": 61.0, "peakAnomaly": 92.0,
  "topContributors": [{"feature":"log1p_sbytes","contribution":1.92}, ...],
  "events": [ScoredEvent...],
  "story": [ { "index": 1, "title": "Initial suspicious behavior", "detail": "...",
               "timestamp": 120340, "evidenceEventIds": ["R-40122"], "epistemics": "Observed|Inferred|Simulated|Prescriptive" }, ...5 stages ],
  "containmentPlaybook": ["Isolate affected hosts...", ...]
}
```

### ScoredEvent shape
```json
{
  "eventId": "R-40122", "rowIndex": 40122, "t": 121000, "wave": "campaign2-exploit",
  "raw": { ...45 UNSW-NB15 raw fields incl. attack_cat & label... },
  "attackProbability": 0.9712, "binaryVerdict": "Attack", "attackConfidence": 0.9712,
  "category": "Exploits", "categoryProbs": {"Analysis":0.001, ...all 10 classes},
  "anomalyRaw": 0.71, "anomalyScore": 74.0, "riskScore": 88.2, "severity": "Critical",
  "cluster": 3, "entity": "ENT-1042",
  "explanation": {
    "method": "exact-treeshap|saabas-path", "baseline": -1.1042,
    "topPositive": [{ "feature": "log1p_sbytes", "value": 12.31, "contribution": 1.92, "direction": "increases_risk" }, ...6],
    "topNegative": [...4],
    "narrative": "High attack confidence (97.1%) driven primarily by unusual source load, ..."
  },
  "groundTruth": "Exploits"   // only for replay & /api/events
}
```

## GET /api/patterns
```json
{ "clusters": [{ "cluster": 0, "size": 4810, "dominant_category": "DoS",
                 "category_distribution": {...}, "top_features": [{"feature":"rate","z_score":2.8},...],
                 "liveAlertCount": 412, "centroidPca": {"x":1.2,"y":-0.4} }, ...8],
  "scatter": [{ "x": ..., "y": ..., "cluster": 0, "category": "DoS" }, ...2500],
  "pcaExplainedVariance": [0.31, 0.18], "featuresUsed": ["dur","rate",...],
  "notes": "Behavior clusters from KMeans (k=8) ... NOT malware families." }
```

## GET /api/explain/global
```json
{ "shapGlobal": { "expected_value": -1.10, "method": "...", "features": [{"feature":"log1p_sbytes","mean_abs_shap":0.92}, ...54] },
  "multiclassGain": { "features": [{"feature":"sttl","gain":8123.1}, ...] },
  "calibration": { "platt": {"a":0.98,"b":0.01}, "temperature": 1.4, "chosen_threshold": 0.42,
                   "threshold_curve": [{"threshold":0.05,"precision":..,"recall":..,"f1":..}, ...],
                   "oof_reliability": [{"bin_low":0.0,"bin_high":0.1,"mean_predicted":0.02,"fraction_positive":0.01,"count":...}, ...],
                   "oof_brier_raw": 0.081, "oof_brier_platt": 0.052 },
  "testEvaluation": { "binary": {...metrics...}, "multiclass": {...}, "anomaly": {...}, "calibration": {...} },
  "datasetProfile": {...}, "methodology": {...} }
```

## POST /api/predict  — body `{ "event": { ...45 fields... } }`
Returns `{ ...ScoredEvent, "latencyMs": 0.38 }`

## POST /api/predict/batch — body `{ "events": [ ... ] }` (≤5000)
`{ "count": n, "latencyMs": 12.3, "events": [ScoredEvent...] }`

## GET /api/events?category=DoS&n=10&seed=7
Deterministic real test-set events of a category, scored live (with groundTruth).

## GET /api/model/info — registry + model comparison + ablation
## GET /api/metrics — test evaluation + operational + runtime stats

---

## Replay (socket.io primary, REST fallback)

1. `POST /api/replay/start {speed:1}` → `{ "replayId": "RP-...", "total": 1215, "durationMs": 195000 }`
2. socket.io: `io("/?XTransformPort=3010")`
   - client emits `replay:join {replayId}` → joins room
   - client emits `replay:control {replayId, action: "play"|"pause"|"speed"|"seek"|"stop", value?}`
   - server emits `replay:tick`:
```json
{ "replayId": "RP-...", "virtualTime": 61200, "done": false,
  "events": [ScoredEvent... new since last tick (explanations trimmed to top-3)],
  "incidents": [Incident... changed since last tick (slimmed)],
  "incidentsChanged": true,
  "stats": { "processed": 431, "total": 1215, "alerts": 96, "normals": 335, "incidents": 4,
             "criticalIncidents": 1, "riskMax": 91.2, "byCategory": {"Reconnaissance": 22, ...}, "throughput": 8.1 } }
```
3. REST fallback: `GET /api/replay/:id/state?cursor=n` → tick snapshot + cursor; `POST /api/replay/:id/control {action,value}`

**Simulated metadata notice**: event timestamps and pseudo-entity IDs in replay are SYNTHETIC (the CSV has no timestamps/IPs). Waves: `background → campaign1-recon → noise → campaign2-exploit → campaign3-dos` escalation story.

---

## Next.js LLM endpoints (port 3000, relative URLs)

### POST /api/ai/incident-summary
Body: `{ "evidence": { incident fields + top events + contributors + story } }`
Returns:
```json
{ "source": "llm"|"fallback",
  "sections": { "executiveSummary": "...", "technicalAnalysis": "...", "whyThisMatters": "...",
                "recommendedInvestigation": ["..."], "suggestedContainment": ["..."], "confidenceCaveats": "..." } }
```

### POST /api/ai/analyst-chat
Body: `{ "messages": [{"role":"user","content":"..."}], "context": { incident evidence } }`
Returns `{ "reply": "...", "source": "llm"|"fallback" }`
