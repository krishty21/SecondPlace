# QA Report

QA performed on the running system (soc-engine :3010, Next.js :3000, artifacts as committed). Sources: worklog, QA session logs, and re-verification during documentation.

## 1. Dataset validation

dataset validation (master notebook `notebooks/CipherMind_Model_Training_and_Evaluation.ipynb` §04; originally `ml/scripts/analyze_dataset.py`, retired) + train-time assertions: schema OK on both files (45 columns), train/test schema match, 0 duplicate rows, 0 missing/infinite values, **0 label↔attack-category mismatches** in both splits, unseen-category scan (state: ACC, CLO) → unknown-bucket handling confirmed. Results committed in `ml/artifacts/metrics/dataset_profile.json`.

## 2. TypeScript-vs-Python cross-validation

`python3 tests/validate_ts_engine.py` against the live engine: 13 deterministic rows (4 normals + one per ground-truth category) POSTed to `/api/predict/batch` and compared with Python LightGBM + Platt/temperature transforms. **All rows match** — calibrated probability identical within 1e-4 (4 decimal places), verdict + predicted category identical on every row. Saabas identity (baseline + Σcontribs == raw score) verified separately during development.

## 3. API endpoint verification

All soc-engine endpoints curl-verified returning **HTTP 200** with contract-conformant JSON: `/api/health`, `/api/dashboard`, `/api/incidents` (+ `/api/incidents/:id` incl. groundTruthMix), `/api/patterns`, `/api/explain` (POST), `/api/predict` (POST), `/api/predict/batch` (POST), `/api/events`, `/api/replay/start` + `/state` + `/control` + `/debug`, `/api/model/info`, `/api/metrics`. Live `/api/health` during QA: engineStats boot 18.7 s / 642 ev/s / 1.7 ms single-event. Frontend LLM routes integration-tested with real incident payloads: `/api/ai/incident-summary` and `/api/ai/analyst-chat` both returned `source: "llm"` completions; deterministic fallback exercised by design (badged `source: fallback`).

## 4. socket.io replay streaming test

Full replay session over the websocket: **29 emitted ticks, 903 events streamed, 10 incidents created, 185 incident updates** (changed-incident payloads on ticks), final `done: true` tick with finalized stories. REST fallback (`/api/replay/:id/state?cursor=`) verified as the recovery path on forced disconnect.

## 5. Browser E2E (headless)

- All **5 views rendered** and tab-switched: Command Center, Incident Investigation, Pattern Explorer, Explainability Center, Live Replay.
- **LLM summary generated** end-to-end in Incident Investigation → AI Analyst (`source: llm`, all 6 sections rendered).
- **Replay speed control 1×→8× verified**: accelerating mid-replay advanced processing from 181 to **817 of 903** events within the observation window; play/pause/seek controls responsive.
- **Mobile 390 px responsive** layout checked (nav collapses, grids stack, dialogs scroll).
- **0 console errors** across the session (warnings: none beyond React devtools noise).
- Engine-down UX: killing soc-engine shows the "Engine connecting…" banner with retry states; auto-recovery and query invalidation verified when it returns.

## 6. Lint / typecheck status

- `bun run lint` (ESLint, Next app): **passes**, no errors.
- `bunx tsc --noEmit`: app code (`src/`) **clean**; root invocation also scans `mini-services/soc-engine` (intentional Bun-style `.ts` import extensions — the engine runs under Bun's transpiler and is verified by cross-validation rather than tsc) and `skills/` (other agents' code).
- soc-engine runtime stability: survived repeated replay sessions + boot re-scores without leaks (behavioral-vector cache bounded at 20k entries).

## Known minor issues

| Issue | Severity | Status |
|---|---|---|
| KPI card sublabel truncation on narrow screens (ellipsis cuts long labels) | Cosmetic | Accepted for the demo; fix is a one-line CSS clamp |
| Root `tsc` picks up mini-services/skills (include `**/*.ts`) | Tooling nit | Documented; no app impact |

No functional defects open at time of writing.
