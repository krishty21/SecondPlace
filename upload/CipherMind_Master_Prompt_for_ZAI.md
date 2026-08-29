# MASTER BUILD PROMPT — CipherMind Sentinel
## CipherMind AI '26 | UNSW-NB15 | Full-Stack AI SOC Copilot

You are the lead AI/ML engineer, cybersecurity engineer, backend engineer, frontend engineer, MLOps engineer, QA engineer, and technical writer for this hackathon project.

I am uploading a ZIP containing ONLY the currently selected UNSW-NB15 files. Treat the uploaded ZIP as the authoritative source for the dataset. Do not invent columns, rows, metadata, or capabilities that are not present in the uploaded files.

The objective is to build a competition-grade, locally runnable, fully tested prototype called:

**CipherMind Sentinel — AI Security Operations Copilot**

Core positioning:

> Transform noisy network-security data into prioritized threats, explainable detections, attack stories, and analyst-ready incident intelligence.

This is NOT supposed to be a simple notebook or a single classifier. It should be an end-to-end AI security product whose detection engine is scientifically evaluated and whose outputs are presented through a polished SOC interface.

---

# 1. HACKATHON REQUIREMENTS

The hackathon challenge asks for an intelligent cyber-defense platform that can:

- Identify anomalies
- Uncover hidden attack patterns
- Prioritize critical threats
- Assist analysts with AI-generated incident insights in real time
- Analyze UNSW-NB15 traffic
- Distinguish real threats from background noise
- Identify suspicious network behavior
- Discover relationships between attack patterns
- Provide attack timelines / incident context
- Generate clear, actionable explanations
- Explain why a prediction was made

Mandatory deliverables:

1. Working AI prototype
2. GitHub repository with source code and README
3. 3–5 minute demo video
4. Project documentation
5. AI model details, training process, and evaluation metrics

Bonus-oriented capabilities:

- Interactive dashboard
- Explainable AI
- Live deployment
- Real-time attack visualization
- LLM-powered incident summaries
- Innovative features beyond the problem statement

Evaluation:

- AI Model Performance & Accuracy — 30%
- Innovation & Technical Implementation — 25%
- Problem Solving & Practical Impact — 20%
- User Experience & Explainability — 15%
- Documentation, Demo & Presentation — 10%

Design every engineering decision to maximize these criteria.

---

# 2. IMPORTANT DATASET REALITY

The ZIP supplied to you contains exactly these files:

- `NUSW-NB15_features.csv`
- `Training and Testing Sets/UNSW_NB15_training-set.csv`
- `Training and Testing Sets/UNSW_NB15_testing-set.csv`

Observed dataset sizes:

- Training: 175,341 rows × 45 columns
- Testing: 82,332 rows × 45 columns

The training set contains:

- label = 0: 56,000 normal records
- label = 1: 119,341 attack records

Attack-category distribution in training:

- Normal: 56,000
- Generic: 40,000
- Exploits: 33,393
- Fuzzers: 18,184
- DoS: 12,264
- Reconnaissance: 10,491
- Analysis: 2,000
- Backdoor: 1,746
- Shellcode: 1,133
- Worms: 130

The test set is the official supplied testing set. Use it as the final held-out evaluation set. Do NOT merge train and test and randomly re-split them for the final score.

CRITICAL LIMITATION:

The provided 45-column training/testing CSVs do NOT contain these raw fields:

- srcip
- sport
- dstip
- dsport

The separate feature-description file documents those fields as part of the broader UNSW-NB15 feature specification, but they are not present in the provided model-training CSVs.

Therefore:

- Do NOT pretend the model sees real source IPs or destination IPs.
- Do NOT claim true user identity, device identity, or network-topology reconstruction from this dataset.
- Do NOT build explanations that quote non-existent IP/user/device fields.
- Do NOT fabricate "real" lateral movement paths from fields we do not have.

The product may still have:
- attack timelines,
- incident correlation,
- behavioral clusters,
- synthetic/replay scenarios,
- pseudo-entities for demo visualization,

but those must be clearly marked as simulated/reconstructed from available flow-level information rather than falsely presented as raw ground truth.

Likewise, UNSW-NB15 is a network intrusion dataset, not an email-phishing dataset and not a malware-binary dataset. Do not falsely claim that this model directly predicts phishing emails or malware family identity from this data.

The architecture may be extensible to those future data sources, but the implemented hackathon model must remain truthful about what the supplied data supports.

---

# 3. DATASET SCHEMA TO USE

The actual model-training CSV columns are:

1. id — record identifier
2. dur — record duration
3. proto — protocol
4. service — network service
5. state — connection state
6. spkts — source packet count
7. dpkts — destination packet count
8. sbytes — source-to-destination bytes
9. dbytes — destination-to-source bytes
10. rate — traffic rate
11. sttl — source TTL
12. dttl — destination TTL
13. sload — source load
14. dload — destination load
15. sloss — source packet loss/retransmission
16. dloss — destination packet loss/retransmission
17. sinpkt — source inter-packet timing
18. dinpkt — destination inter-packet timing
19. sjit — source jitter
20. djit — destination jitter
21. swin — source TCP window
22. stcpb — source TCP base sequence
23. dtcpb — destination TCP base sequence
24. dwin — destination TCP window
25. tcprtt — TCP round-trip time
26. synack — SYN/ACK timing
27. ackdat — ACK timing
28. smean — source packet mean size
29. dmean — destination packet mean size
30. trans_depth — transaction depth
31. response_body_len — response body length
32. ct_srv_src — connections involving service/source
33. ct_state_ttl — state/TTL relationship count
34. ct_dst_ltm — destination activity count
35. ct_src_dport_ltm — source/destination-port activity count
36. ct_dst_sport_ltm — destination/source-port activity count
37. ct_dst_src_ltm — destination/source relationship count
38. is_ftp_login — FTP login indicator
39. ct_ftp_cmd — FTP command count
40. ct_flw_http_mthd — HTTP method/flow count
41. ct_src_ltm — source activity count
42. ct_srv_dst — service/destination relationship count
43. is_sm_ips_ports — same-IP/same-port indicator
44. attack_cat — attack category target
45. label — binary target

For supervised training:

- Binary target = `label`
- Multiclass target = `attack_cat`

Do not use the target columns as input features.

Be extremely careful with `id`: treat it as an identifier, not a learned semantic security feature. Prefer excluding it from model features unless you can demonstrate a non-leaky legitimate use.

---

# 4. PRIMARY PRODUCT VISION

Build a system with this flow:

DATA
↓
Validation / preprocessing
↓
Feature engineering
↓
AI detection
↓
Anomaly scoring
↓
Threat scoring
↓
Alert normalization
↓
Incident correlation
↓
Attack-story reconstruction
↓
Explainable AI
↓
Analyst copilot / incident summary
↓
SOC dashboard

The main user is a SOC analyst who needs to answer:

1. What is happening?
2. How serious is it?
3. Why does the model believe this?
4. Which alerts belong together?
5. What happened first?
6. What is the likely attack category?
7. What should I investigate next?

---

# 5. ML ARCHITECTURE

Do NOT rely on one model.

Build a layered detection architecture.

## Model A — Binary Attack Detector

Goal:

Normal vs malicious.

Recommended starting point:

- XGBoost or LightGBM
- Use the implementation that provides the best validated balance of accuracy, recall, macro-F1, inference speed, and interpretability.

Evaluate alternatives if computationally practical:

- Logistic Regression
- Random Forest
- XGBoost / LightGBM

Do a principled comparison and choose the final model based on validation metrics, not preference.

Output:

- probability_normal
- probability_attack
- attack_confidence

---

## Model B — Multiclass Attack Classifier

Goal:

Predict `attack_cat`.

Classes:

- Normal
- Analysis
- Backdoor
- DoS
- Exploits
- Fuzzers
- Generic
- Reconnaissance
- Shellcode
- Worms

Because the class distribution is highly imbalanced, explicitly handle imbalance.

Investigate:

- class weights
- sample weights
- stratified validation
- appropriate probability calibration
- possibly controlled oversampling only on the training portion if justified

Do NOT use naive oversampling before splitting.

Do NOT allow data leakage.

Optimize for:

- macro-F1
- per-class recall
- balanced accuracy
- confusion matrix
- weighted F1
- overall accuracy

Do not report only accuracy.

Especially investigate rare classes such as Worms and Shellcode.

---

## Model C — Unsupervised Anomaly Detector

Purpose:

Identify unusual behavior even when the supervised classifier is uncertain.

Start with:

- Isolation Forest

Train the anomaly detector primarily on NORMAL training behavior.

Output:

- raw anomaly score
- normalized anomaly score 0–100

Make the transformation deterministic and documented.

Optionally compare with an Autoencoder if compute/resources permit, but do not add complexity merely for appearance. Keep Isolation Forest as the reliable fallback.

---

## Model D — Threat Risk Scoring Engine

Do NOT train a fake "risk model" from invented labels.

Create a transparent, configurable risk-scoring layer from validated signals.

Possible components:

- supervised attack confidence
- anomaly score
- attack category severity
- uncertainty
- rarity
- repeated related alerts
- incident escalation/progression
- contextual weighting

Normalize to 0–100.

Example output:

0–24 = Low
25–49 = Medium
50–74 = High
75–100 = Critical

Document the exact formula and weights in the project documentation.

Keep the formula configurable.

---

# 6. FEATURE ENGINEERING

The final pipeline must be reproducible.

Expected transformations:

### Numerical features

Handle:

- missing values
- infinite values
- extreme skew/outliers where appropriate
- scaling only when required by the specific model

Consider log transforms for highly skewed positive variables where scientifically justified.

### Categorical features

At minimum:

- proto
- service
- state

Use a production-safe encoding strategy.

Prefer an encoder that safely handles unseen categories at inference.

### Derived behavioral features

You may create additional features from fields actually present.

Examples:

- byte_ratio = sbytes / (dbytes + epsilon)
- packet_ratio = spkts / (dpkts + epsilon)
- payload_per_packet
- total_bytes
- total_packets
- total_loss
- directional asymmetry
- RTT/timing-derived features
- load ratios
- packet-size asymmetry
- state/traffic interaction features
- burst/rate proxies

Do not create derived variables that imply fields we do not have.

Perform feature ablation to prove whether engineering helps.

---

# 7. DATA LEAKAGE PREVENTION

This is NON-NEGOTIABLE.

Implement:

- training-only fitting of preprocessors
- training-only class balancing
- no target leakage
- no test-derived feature statistics
- no test-based threshold tuning
- no test-based model selection

Use the official train/test files correctly.

Within the training set, create a stratified validation split or stratified cross-validation for model selection and threshold calibration.

Use the supplied official test set exactly once for final benchmark evaluation after the final pipeline has been frozen.

Record random seeds.

---

# 8. EVALUATION REQUIREMENTS

Build a complete evaluation report.

Required:

### Binary

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix
- specificity / false-positive rate
- threshold analysis

### Multiclass

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Balanced Accuracy
- Per-class precision / recall / F1
- Confusion matrix
- one-vs-rest ROC-AUC if computationally useful

### Anomaly detector

Because labels exist for evaluation but are NOT used as the primary training signal:

- anomaly-score distributions
- attack vs normal separation
- ROC-AUC / PR-AUC for anomaly ranking
- precision@k or recall@k for useful analyst triage scenarios
- threshold analysis

### Operational metrics

Also measure:

- inference latency
- batch throughput
- model artifact size
- API response latency where feasible

The goal is not just high accuracy.

The goal is useful security triage.

---

# 9. MODEL CALIBRATION

Evaluate whether predicted probabilities are trustworthy.

Where justified, use:

- Platt scaling
- isotonic calibration
- calibration curves
- Brier score

Do not claim that a model is "97% certain" unless its probability behavior supports that interpretation.

---

# 10. EXPLAINABLE AI

Use SHAP or another robust explainability method.

For each prediction, the interface must be able to answer:

> Why did the model classify this event this way?

Show:

- top positive contributors
- top negative contributors
- feature value
- contribution direction
- model confidence
- predicted class
- baseline/expected behavior if available

Make explanations human readable.

Example:

"High attack confidence was driven primarily by unusually high source traffic load, packet rate, and destination/source activity."

Do not fabricate reasons.

The explanation must be generated directly from actual model features.

---

# 11. INCIDENT CORRELATION ENGINE

UNSW-NB15 rows are individual network-flow records and do not contain a complete SOC event stream.

Build an incident-correlation layer that operates on model outputs.

Possible logic:

- sliding time windows when timestamps are available/created in the replay layer
- similarity of predicted attack categories
- similarity of behavioral feature profiles
- repeated patterns
- temporal proximity
- source/session grouping when actual identifiers are available in the runtime data

Because the supplied CSV lacks source/destination IP fields, do not claim true IP-based correlation.

For demonstrations, create a replay/event layer that assigns synthetic event IDs and clearly labels any synthetic entity information.

Goal:

Turn many related alerts into fewer incidents.

Example:

50 related alerts
→ 1 correlated incident

---

# 12. ATTACK STORY / TIMELINE

Create an analyst-friendly timeline.

Example structure:

1. Initial suspicious behavior
2. Escalation / repeated anomalies
3. Attack-category transition
4. High-confidence malicious event
5. Incident peak
6. Recommended response

The "story" must be based on actual observed model outputs and derived event relationships.

Do not invent attack steps unsupported by the data.

The UI can label inferred transitions as:

- Observed
- Inferred
- Simulated

This distinction is important for trust.

---

# 13. LLM INCIDENT COPILOT

Use an LLM ONLY for explanation, summarization, prioritization assistance, and analyst communication.

Do NOT use an LLM as the primary network-flow classifier.

Input the LLM with structured evidence such as:

- model prediction
- confidence
- anomaly score
- risk score
- attack category
- top SHAP contributors
- incident timeline
- correlated event statistics
- explicitly marked inferred/simulated context

The LLM must NOT invent facts.

Require the prompt to:

- cite supplied evidence in prose
- distinguish observed vs inferred
- avoid inventing IPs/users/malware families
- produce uncertainty language
- recommend safe analyst investigation steps

Generate:

### Executive Summary

1–3 sentences.

### Technical Analysis

Explain what the model saw.

### Why This Matters

Business/security impact.

### Recommended Investigation

Concrete analyst checks.

### Suggested Containment

Only when appropriate and framed as recommended action, not confirmed truth.

### Confidence / Caveats

Explicitly mention limitations.

If no external LLM API key is available, the application must still work using a deterministic local fallback summary.

---

# 14. THREAT / BEHAVIOR CLUSTERING

The challenge asks for hidden attack patterns and relationships.

Build a clustering module using model/feature representations.

Possible approach:

- standardized behavioral feature vector
- dimensionality reduction for visualization (e.g. UMAP/PCA)
- clustering (HDBSCAN / DBSCAN / K-Means)

Use clustering to identify:

- recurring behavioral groups
- attack-category similarities
- unusual outlier clusters

Do NOT call these "malware families" unless supported by actual malware-sample data.

Use precise terminology:

- behavior clusters
- attack-pattern clusters
- traffic-behavior groups

This is scientifically honest and still addresses hidden-pattern discovery.

---

# 15. FRONTEND — FULL-STACK SOC EXPERIENCE

Build a polished React application.

Preferred stack:

- React
- TypeScript
- Tailwind CSS
- a mature charting library
- interactive graph/timeline library where useful
- FastAPI backend
- Python ML layer

Do not create a generic admin template.

The interface should look like a serious modern SOC product.

Suggested visual language:

- dark security operations theme
- restrained use of alert/severity colors
- high information density
- strong typography
- clear hierarchy
- fast scanning
- accessible status indicators
- responsive layout

---

# 16. UI PAGES

## Page 1 — SOC Command Center

Primary screen.

Show:

- Total Alerts
- Active Incidents
- Critical Threats
- Detection Rate
- False Positive Indicator
- High-Risk Trend
- Attack-category breakdown
- Threat timeline
- Recent critical incidents

Main panel:

### "What needs attention now?"

Prioritized incident list.

Each item:

- risk score
- severity
- predicted attack
- confidence
- anomaly score
- number of correlated alerts
- status
- explanation preview

---

## Page 2 — Incident Investigation

When analyst opens an incident:

Show:

- incident severity
- overall risk score
- attack category
- confidence
- anomaly score
- incident summary
- evidence
- event timeline
- attack progression
- top contributing features
- recommended investigation steps

Use tabs or sections:

Overview
Timeline
Evidence
Explainability
AI Analyst
Related Patterns

---

## Page 3 — Threat Intelligence / Pattern Explorer

Show:

- behavior clusters
- category distributions
- unusual patterns
- cluster sizes
- representative feature profiles

Interactive visualization preferred.

---

## Page 4 — Explainability Center

Show:

- global feature importance
- local SHAP explanation
- selected event details
- confidence calibration
- feature contribution charts

---

## Page 5 — Live Detection / Replay

This is important for the demo.

Allow the user to:

- start a replay
- pause
- resume
- change replay speed
- see alerts appear
- watch incidents form
- watch risk scores change
- open an incident in real time

The replay uses the provided test data or a selected test subset.

Do not claim this is a live production network feed.

Label it clearly as:

"Real-Time Detection Replay"

---

# 17. SOC DEMO EXPERIENCE

Build the application so the demo is visually compelling.

Ideal flow:

1. Open command center
2. Show alert stream
3. Start replay
4. Several low/medium alerts arrive
5. Model identifies increasingly suspicious behavior
6. Related events are correlated into an incident
7. Incident becomes High/Critical
8. Analyst opens incident
9. Timeline appears
10. SHAP explains why
11. LLM summarizes incident
12. Analyst sees recommended next steps
13. Pattern explorer shows related behavior cluster

The story should demonstrate:

>  many noisy events → one prioritized security story

---

# 18. BACKEND API

Create clean APIs.

At minimum:

- health/status
- model metadata
- single prediction
- batch prediction
- anomaly scoring
- risk scoring
- explainability
- incidents
- incident detail
- replay start/control
- clustering/pattern summary
- AI incident summary

Use typed request/response schemas.

Handle errors cleanly.

Add API documentation.

---

# 19. MODEL ARTIFACTS — REQUIRED

This is CRITICAL.

After training, save and provide the COMPLETE runnable model package.

Must include:

- binary classifier weights/model
- multiclass classifier weights/model
- anomaly detector
- preprocessing/encoding objects
- feature-engineering configuration
- class-label mappings
- threshold configuration
- risk-scoring configuration
- model version metadata
- training configuration
- feature list
- metrics JSON
- evaluation report
- reproducibility information

A user should be able to download the repository and use the saved artifacts without retraining.

Do NOT merely save the final predictions.

Do NOT merely give notebooks.

I need the actual model artifacts/weights and preprocessing pipeline.

Use an organized directory such as:

`artifacts/models/`
`artifacts/preprocessor/`
`artifacts/metrics/`
`artifacts/reports/`

---

# 20. MODEL REGISTRY METADATA

Create a machine-readable model metadata file containing:

- model name
- model version
- training date
- dataset file names
- train/test row counts
- feature count
- feature list
- algorithm
- hyperparameters
- random seed
- class mapping
- thresholds
- evaluation metrics
- artifact paths
- software versions

---

# 21. TRAINING PIPELINE

Build a reproducible training entry point.

The user should be able to run something equivalent to:

`train`

and get:

1. dataset validation
2. preprocessing
3. feature engineering
4. validation
5. model selection
6. training
7. calibration if used
8. test evaluation
9. artifact serialization
10. metrics report
11. plots
12. model metadata

Do not force the user to manually edit Python notebooks.

Prefer clean CLI commands/scripts.

---

# 22. INFERENCE PIPELINE

Inference should load artifacts and run without retraining.

Provide:

- Python inference service
- REST API
- batch inference option
- UI integration

Support a single event and batch inputs.

Return structured JSON.

---

# 23. TESTING REQUIREMENTS

You MUST actually test the implementation.

At minimum:

### Data tests

- expected columns exist
- target columns exist
- no unexpected data corruption
- dtypes are reasonable
- missing/infinite values handled
- train/test schema matches

### ML tests

- model loads from saved artifacts
- preprocessing loads
- single prediction works
- batch prediction works
- anomaly scoring works
- explanations work
- labels map correctly

### API tests

- health endpoint
- prediction endpoint
- batch prediction
- incidents endpoint
- explanation endpoint
- replay endpoint

### Frontend tests

- app builds
- routes load
- API integration works
- loading states work
- empty/error states work
- incident detail works

### End-to-end

Actually run:

dataset
→ model
→ API
→ UI

and confirm the user can go from input event to visible prediction.

Do not merely say that tests "should pass".

Run them.

---

# 24. PERFORMANCE / RESOURCE AWARENESS

The app must be practical on a normal developer machine.

Avoid unnecessarily enormous models.

For inference:

- load models once
- do not retrain per request
- batch where appropriate
- cache expensive explainability computations when sensible
- avoid blocking the main API process

Provide a lightweight demo mode.

---

# 25. SECURITY / RELIABILITY

Implement good engineering practices:

- environment variables for secrets
- no hardcoded API keys
- safe API error handling
- input validation
- CORS configured appropriately
- no arbitrary code execution
- no leaking stack traces to users
- explicit LLM fallback if unavailable

---

# 26. REPOSITORY STRUCTURE

Use a professional structure similar to:

/
├── frontend/
├── backend/
├── ml/
│   ├── training/
│   ├── inference/
│   ├── preprocessing/
│   ├── explainability/
│   ├── clustering/
│   └── evaluation/
├── artifacts/
│   ├── models/
│   ├── preprocessor/
│   ├── metrics/
│   └── reports/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── scripts/
├── docs/
├── .env.example
├── README.md
└── docker-compose.yml

Adapt this structure when necessary, but preserve clean separation between:

- frontend
- API
- ML
- artifacts
- tests
- documentation

---

# 27. PARALLEL SUB-AGENTS

USE PARALLEL AGENTS / SUB-AGENTS aggressively where your environment supports them.

Do not make one agent sequentially do every task.

Create a coordinated work breakdown.

## Agent 1 — Dataset Scientist

Responsibilities:

- inspect uploaded files
- validate schema
- analyze distributions
- inspect missing/infinite values
- check duplicates
- identify leakage risks
- analyze class imbalance
- recommend useful feature engineering
- generate dataset report

Deliverables:

`docs/dataset_analysis.md`
`artifacts/metrics/dataset_profile.json`

---

## Agent 2 — ML Research Engineer

Responsibilities:

- baseline models
- compare candidate algorithms
- validation strategy
- class imbalance handling
- binary classifier
- multiclass classifier
- threshold analysis
- probability calibration
- model selection

Deliverables:

trained model artifacts
evaluation reports
comparison tables

---

## Agent 3 — Anomaly/XAI Engineer

Responsibilities:

- Isolation Forest
- anomaly normalization
- threshold strategy
- SHAP integration
- global/local explanations
- explanation API contract

Deliverables:

anomaly model
XAI module
explainability documentation

---

## Agent 4 — Threat Intelligence Engineer

Responsibilities:

- risk scoring architecture
- alert normalization
- incident correlation
- behavioral clustering
- attack-story logic
- observed vs inferred vs simulated semantics

Deliverables:

incident engine
risk-scoring documentation
pattern explorer data contracts

---

## Agent 5 — Backend Engineer

Responsibilities:

- FastAPI
- inference service
- APIs
- model loading
- caching
- replay service
- LLM orchestration
- API tests

Deliverables:

complete backend

---

## Agent 6 — Frontend Engineer

Responsibilities:

- React/TypeScript application
- SOC command center
- incident investigation
- explainability UI
- threat pattern explorer
- real-time replay
- responsive design

Deliverables:

complete frontend

---

## Agent 7 — MLOps / Packaging Engineer

Responsibilities:

- reproducible training
- serialization
- model artifacts
- version metadata
- requirements
- Docker
- local setup
- environment configuration

Deliverables:

artifact package
Docker setup
deployment instructions

---

## Agent 8 — QA / Red-Team Reviewer

Responsibilities:

- inspect the entire implementation
- run tests
- find data leakage
- find broken APIs
- find UI bugs
- verify model loading
- verify reproducibility
- challenge unsupported claims
- test edge cases
- produce final punch list

Deliverables:

`docs/qa_report.md`

---

## Agent 9 — Hackathon Judge / Product Reviewer

Act like a skeptical judge.

Review:

- innovation
- technical depth
- practical impact
- UX
- explainability
- demo wow factor
- documentation

Suggest only changes that can realistically improve judging score.

Do NOT sacrifice correctness for flashy features.

---

# 28. AGENT COORDINATION

Agents may work in parallel, but integration must happen carefully.

Before finalizing:

1. Merge modules
2. Resolve interface mismatches
3. Run full tests
4. Run end-to-end flow
5. Rebuild frontend
6. Start backend
7. Load saved models
8. Execute sample prediction
9. Run replay
10. Verify incident generation
11. Verify XAI
12. Verify LLM fallback
13. Verify README

No module is considered finished until it works with the complete system.

---

# 29. IMPORTANT PRODUCT DECISIONS

Prefer:

- XGBoost/LightGBM for tabular supervised detection
- Isolation Forest for anomaly detection
- SHAP for explainability
- transparent threat scoring
- behavioral clustering
- FastAPI
- React + TypeScript
- polished SOC UI
- reproducible artifacts

Do NOT add technology merely because it sounds impressive.

Every feature must improve one or more of:

- detection
- prioritization
- explanation
- investigation
- demo quality

---

# 30. WHAT NOT TO DO

DO NOT:

- build only a Jupyter notebook
- report only accuracy
- use the test set to tune the final model
- leak target values
- use `id` as a meaningful security feature without justification
- fabricate IPs/users/devices
- fabricate malware families
- fabricate phishing detection from this network dataset
- claim true real-time production traffic
- use an LLM as the primary classifier
- hardcode secrets
- leave model weights out of the repository/release package
- create a README that does not actually work
- say "tested" without actually running tests
- produce a dashboard that is disconnected from the real model

---

# 31. REQUIRED DOCUMENTATION

Produce these documents:

### README.md

Must include:

- project overview
- architecture
- prerequisites
- Python version
- Node version
- installation
- environment variables
- dataset placement
- training command
- evaluation command
- inference command
- backend startup
- frontend startup
- Docker setup
- model artifact location
- sample API usage
- demo instructions
- troubleshooting

A person cloning the project onto a clean machine should be able to follow this README.

### docs/ARCHITECTURE.md

Detailed architecture.

### docs/DATASET.md

Exact dataset understanding and limitations.

### docs/MODEL_CARD.md

For every model:

- purpose
- inputs
- outputs
- training process
- metrics
- limitations
- intended use

### docs/EVALUATION.md

Detailed metrics and methodology.

### docs/THREAT_SCORING.md

Risk formula and reasoning.

### docs/INCIDENT_CORRELATION.md

How alerts become incidents.

### docs/XAI.md

How explanations are generated.

### docs/LIMITATIONS.md

Be honest.

This increases credibility.

---

# 32. DEMO SEED DATA

Create a deterministic demo configuration.

The demo should always produce an interesting sequence.

Requirements:

- same random seed / deterministic selection
- selected attack-heavy events
- selected normal background
- progressive alert escalation
- visible incident creation
- explainability output
- LLM summary/fallback

Do not fake model results.

Choose real events from the supplied dataset and replay their model outputs.

If synthetic metadata is added for visualization, mark it as simulation metadata.

---

# 33. ACCEPTANCE CRITERIA

Do not finish until ALL of these are true:

[ ] Dataset loads successfully

[ ] Dataset schema validated

[ ] Train/test handling is correct

[ ] Binary classifier trained

[ ] Multiclass classifier trained

[ ] Anomaly detector trained

[ ] Final models evaluated on official test set

[ ] Metrics generated

[ ] Confusion matrices generated

[ ] Rare classes analyzed

[ ] Model probabilities calibrated or calibration explicitly assessed

[ ] SHAP explanations work

[ ] Risk scoring works

[ ] Alert correlation works

[ ] Incident timeline works

[ ] Behavioral clustering works

[ ] Backend API works

[ ] React frontend works

[ ] Replay mode works

[ ] LLM summary works or safe fallback works

[ ] Saved model artifacts load correctly

[ ] A fresh Python process can load artifacts

[ ] Frontend builds successfully

[ ] Automated tests pass

[ ] End-to-end test passes

[ ] README instructions verified

[ ] No hardcoded secrets

[ ] Unsupported dataset capabilities are not claimed

---

# 34. FINAL DELIVERABLE PACKAGE

At completion, return:

1. Full source code
2. Trained model artifacts / weights
3. Preprocessing artifacts
4. Anomaly detector
5. Metrics
6. Evaluation reports
7. Model metadata
8. React frontend
9. FastAPI backend
10. Tests
11. Docker configuration
12. README
13. Architecture documentation
14. Model card
15. Dataset documentation
16. Threat-scoring documentation
17. XAI documentation
18. Incident-correlation documentation
19. Limitations documentation

Also provide a concise final report containing:

### Final model selected
Why it was selected.

### Final test metrics
All key metrics.

### Best/worst classes
Especially rare classes.

### Anomaly performance
Ranking metrics.

### Explainability
How it works.

### Operational performance
Inference latency/throughput.

### How to run locally
Exact commands.

### Model artifact locations
Exact paths.

### Demo flow
Exact steps.

---

# 35. QUALITY BAR

Act as if this will be reviewed by:

- an ML engineer
- a cybersecurity engineer
- a SOC analyst
- a startup CTO
- and a very skeptical hackathon judge

The final product must be:

- technically defensible
- scientifically evaluated
- visually polished
- reproducible
- locally runnable
- explainable
- honest about limitations
- impressive in a 3–5 minute demo

Do not optimize for buzzwords.

Optimize for:

**credible AI + useful SOC workflow + excellent UX + excellent proof.**

---

# 36. START NOW

First inspect the uploaded ZIP and validate every file.

Then run the project as a coordinated multi-agent build.

Do not ask me to manually perform obvious implementation steps that you can handle yourself.

Use parallel agents for independent modules.

Before making major architecture choices, inspect the actual dataset.

Before calling the project complete, run the full test/evaluation/integration cycle.

Most importantly:

**I need an actual working implementation, not a plan, not pseudocode, and not a partial prototype.**

The finished system should allow me to:

1. launch the backend
2. launch the frontend
3. load the trained models
4. run a prediction
5. view the prediction in the SOC UI
6. inspect why it was predicted
7. see its risk score
8. see related events/incidents
9. run the replay
10. obtain an analyst-ready AI summary

And I need the complete saved models/weights and all required preprocessing artifacts so I can run the system locally without retraining.
