# Task 5 — frontend-engineer — Frontend SOC UI

## Status: COMPLETE (backend live-verification pending — engine not up during dev window)

## What was built
Single-page Next.js 16 App (src/app/page.tsx → `SocApp`, the only route), dark SOC theme, 5 client-side views:

| View | File | Highlights |
|---|---|---|
| Command Center | `src/components/soc/views/command-center.tsx` | 6 count-up KPIs, threat timeline (dual-axis ComposedChart), category bars, prioritized incident list, recent critical alerts, about-data card |
| Incident Investigation | `views/incident-investigation.tsx` | search/filter list (mobile Select), RiskGauge header, 6 tabs: Overview / Timeline / Evidence / Explainability / AI Analyst (summary + chat) / Related Patterns |
| Pattern Explorer | `views/pattern-explorer.tsx` | 2500-pt scatter in 8 cluster series, centroid X-marks, click-highlight, cluster cards w/ z-score bars, category distribution sidebar |
| Explainability Center | `views/explainability-center.tsx` | global SHAP top-20, multiclass gain, reliability + Brier, threshold curve, local explanation explorer (POST /api/explain), model comparison tables + ablation |
| Live Replay | `views/live-replay.tsx` | socket.io `/ ?XTransformPort=3010`, play/pause/speed/progress/virtual clock, 200-row flash feed, live incidents w/ sparklines + inline story, stats strip, REST polling fallback |

## Key files
- `src/lib/soc-api.ts` — typed API client (contract-mirrored interfaces; XTransformPort=3010 for engine, same-origin for /api/ai/*)
- `src/lib/soc-ui.ts` — palettes (severity rose/orange/amber/emerald, accent cyan #06b6d4, violet SIM), formatters (mm:ss, 1dp risk, %)
- `src/components/soc/*` — primitives, risk-gauge, kpi-card, incident-row, event-detail, contribution-bar, cluster-card, engine-status, providers
- `src/app/globals.css` — dark SOC palette + scrollbars + `.soc-bg`; `layout.tsx` — `class="dark"` + metadata

## Integration notes for later agents
- Engine fetches: ALWAYS `/api/...?XTransformPort=3010` (relative). Socket: `io("/?XTransformPort=3010", {transports:["websocket","polling"]})`.
- Replay incident IDs collide with boot incident IDs (both start INC-0001) → Live Replay passes the live incident object as fallback data to View 2 ("replay session" badge); it does NOT refetch `/api/incidents/:id` for those.
- Engine-down UX: amber "Engine connecting…" banner + skeletons/error states with retry; when /api/health recovers, all queries auto-invalidate (soc-app.tsx).
- Fixed pre-existing TS error in `src/app/api/ai/analyst-chat/route.ts` (role union type) — tsc now clean for the Next app.
- Verified: tsc --noEmit clean, `bun run lint` clean, dev.log clean, all 5 views render in headless browser, both AI endpoints integration-tested (source:"llm").

## Pending
- Final live contract check: `curl http://localhost:3010/api/health`, `/api/dashboard`, `/api/patterns` once ML training finishes and the engine starts — types were coded strictly to docs/API_CONTRACT.md + engine source; no drift expected.
