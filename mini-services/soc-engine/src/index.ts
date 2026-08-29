/**
 * CipherMind Sentinel — SOC Engine (mini-service, port 3010)
 * REST API + socket.io live replay streamer. Loads the trained model artifacts
 * and scores REAL UNSW-NB15 test events on demand.
 */
import { createServer, type IncomingMessage, type ServerResponse } from "http";
import { Server } from "socket.io";
import { loadArtifacts, type RawEvent } from "./artifacts.ts";
import { SocState, csvRowToEvent, slimIncident } from "./state.ts";
import { ReplayManager } from "./replay.ts";
import { CATEGORIES } from "./engine.ts";

const PORT = 3010;

console.log("[soc-engine] booting ...");
const state = new SocState();
const artifacts = loadArtifacts();

const httpServer = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  const p = url.pathname;
  const cors = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
  Object.entries(cors).forEach(([k, v]) => res.setHeader(k, v));
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  const json = (data: unknown, code = 200) => {
    res.writeHead(code, { "Content-Type": "application/json", ...cors });
    res.end(JSON.stringify(data));
  };
  const body = async (): Promise<Record<string, unknown>> => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    try { return JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}"); } catch { return {}; }
  };
  const err = (message: string, code = 400) => json({ error: message }, code);

  try {
    // ---------------------------------------------------------------- health
    if (p === "/api/health") {
      return json({
        status: "ok",
        service: "soc-engine",
        modelsLoaded: true,
        version: artifacts.registry.version,
        trainedAt: artifacts.registry.trained_at,
        uptimeSec: Math.round(process.uptime()),
        incidentsTracked: state.incidents.length,
        engine: state.dashboard.engineStats,
      });
    }

    // ------------------------------------------------------------ model info
    if (p === "/api/model/info") {
      return json({
        registry: artifacts.registry,
        comparison: artifacts.modelComparison,
        featureAblation: artifacts.evalSummary.feature_ablation,
      });
    }

    // ------------------------------------------------------------- dashboard
    if (p === "/api/dashboard") {
      return json(state.dashboard);
    }

    // ------------------------------------------------------------- incidents
    if (p === "/api/incidents") {
      const limit = Math.min(200, Number(url.searchParams.get("limit") ?? 50));
      const sev = url.searchParams.get("severity");
      let list = state.incidents;
      if (sev) list = list.filter((i) => i.severity === sev);
      return json({ total: list.length, incidents: list.slice(0, limit).map(slimIncident) });
    }
    const incMatch = p.match(/^\/api\/incidents\/([\w-]+)$/);
    if (incMatch) {
      const inc = state.incidentDetail(incMatch[1]);
      if (!inc) return err("incident not found", 404);
      // enrich with ground truth mix (demo transparency)
      const gt = new Map<string, number>();
      for (const e of inc.events) { const g = (e.raw as RawEvent).attack_cat; gt.set(g, (gt.get(g) ?? 0) + 1); }
      return json({ incident: inc, groundTruthMix: Object.fromEntries(gt) });
    }

    // ------------------------------------------------------------- patterns
    if (p === "/api/patterns") {
      const c = artifacts.clustering;
      // live cluster distribution over current boot alerts
      const dist = new Map<number, number>();
      for (const e of state.events) {
        if (e.binaryVerdict === "Attack" && e.cluster !== null) dist.set(e.cluster, (dist.get(e.cluster) ?? 0) + 1);
      }
      return json({
        clusters: c.profiles.map((p2) => ({
          ...p2,
          liveAlertCount: dist.get(p2.cluster) ?? 0,
          centroidPca: projectCentroid(p2.cluster),
        })),
        scatter: c.sample_points,
        pcaExplainedVariance: c.pca_explained_variance,
        featuresUsed: c.features,
        notes: "Behavior clusters from KMeans (k=8) on standardized behavioral features of TRAINING attacks; live counts from boot-state alerts scored by the real models. These are traffic-behavior groups, NOT malware families.",
      });
    }

    // -------------------------------------------------------- explainability
    if (p === "/api/explain/global") {
      return json({
        shapGlobal: artifacts.shapGlobal,
        multiclassGain: artifacts.mcGain,
        calibration: {
          platt: artifacts.calibration.platt,
          temperature: artifacts.calibration.temperature,
          chosen_threshold: artifacts.calibration.chosen_threshold,
          threshold_curve: artifacts.calibration.threshold_curve,
          oof_reliability: artifacts.calibration.oof_reliability,
          oof_brier_raw: artifacts.calibration.oof_brier_raw,
          oof_brier_platt: artifacts.calibration.oof_brier_platt,
        },
        testEvaluation: artifacts.testEval,
        datasetProfile: artifacts.datasetProfile,
        methodology: {
          global: "Exact TreeSHAP (shap.TreeExplainer) on the LightGBM binary model over a 2000-row training sample",
          local: "Exact TreeSHAP when precomputed for the event (replay/boot rows); otherwise Saabas path attributions computed live by the TypeScript engine (verified: baseline + sum(contribs) == raw model score)",
        },
      });
    }

    if (p === "/api/explain" && req.method === "POST") {
      const b = await body();
      const ev = b.event as RawEvent | undefined;
      if (!ev) return err("body must include `event`");
      const scored = state.engine.score(ev, { includeGroundTruth: false });
      return json(scored);
    }

    // ------------------------------------------------------------ prediction
    if (p === "/api/predict" && req.method === "POST") {
      const b = await body();
      const ev = b.event as RawEvent | undefined;
      if (!ev) return err("body must include `event`");
      const t0 = performance.now();
      const scored = state.engine.score(ev, { includeGroundTruth: false });
      return json({ ...scored, latencyMs: Number((performance.now() - t0).toFixed(3)) });
    }

    if (p === "/api/predict/batch" && req.method === "POST") {
      const b = await body();
      const evs = (b.events ?? []) as RawEvent[];
      if (!Array.isArray(evs) || evs.length === 0) return err("body must include `events` array");
      if (evs.length > 5000) return err("batch limited to 5000 events");
      const t0 = performance.now();
      const scored = evs.map((ev) => state.engine.score(ev));
      return json({
        count: scored.length,
        latencyMs: Number((performance.now() - t0).toFixed(2)),
        events: scored,
      });
    }

    // ------------------------------------------------------------ test events
    if (p === "/api/events") {
      const cat = url.searchParams.get("category") ?? "Exploits";
      const n = Math.min(50, Number(url.searchParams.get("n") ?? 10));
      const seed = Number(url.searchParams.get("seed") ?? 7);
      // deterministic sample of test rows in the category
      const matches: number[] = [];
      const catIdx = state.header.indexOf("attack_cat");
      for (let i = seed % 100; i < state.testRows.length && matches.length < n * 3; i += 97) {
        if (state.testRows[i][catIdx] === cat) matches.push(i);
      }
      const events = matches.slice(0, n).map((i) => {
        const ev = csvRowToEvent(state.header, state.testRows[i]);
        const scored = state.engine.score(ev, { rowIndex: i, eventId: `T-${i}`, includeGroundTruth: true });
        return scored;
      });
      return json({ category: cat, count: events.length, events });
    }

    // ---------------------------------------------------------------- replay
    if (p === "/api/replay/start" && req.method === "POST") {
      const b = await body();
      const res2 = replay.start({ speed: Number(b.speed ?? 1), engine: state.engine });
      return json(res2);
    }
    const snapMatch = p.match(/^\/api\/replay\/([\w-]+)\/state$/);
    if (snapMatch) {
      const since = Number(url.searchParams.get("cursor") ?? 0);
      const snap = replay.snapshot(snapMatch[1], since);
      if (!snap) return err("replay not found", 404);
      return json({ ...snap, cursor: replay.getSession(snapMatch[1])?.cursor ?? 0 });
    }
    const dbgMatch = p.match(/^\/api\/replay\/([\w-]+)\/debug$/);
    if (dbgMatch) {
      const s = replay.getSession(dbgMatch[1]);
      if (!s) return err("replay not found", 404);
      return json({
        cursor: s.cursor, virtualTime: s.virtualTime, playing: s.playing,
        speed: s.speed, done: s.done, total: s.scored.length,
        firstEventT: s.scored[0]?.t, lastEventT: s.scored[s.scored.length - 1]?.t,
        stats: s.stats,
      });
    }
    if (p.match(/^\/api\/replay\/([\w-]+)\/control$/) && req.method === "POST") {
      const id = p.match(/^\/api\/replay\/([\w-]+)\/control$/)![1];
      const b = await body();
      const ok = replay.control(id, b.action as "play", Number(b.value));
      return ok ? json({ ok: true }) : err("replay not found", 404);
    }

    // ---------------------------------------------------------------- metrics
    if (p === "/api/metrics") {
      return json({
        testEvaluation: artifacts.testEval,
        operational: artifacts.operational,
        evalSummary: artifacts.evalSummary,
        engineRuntime: state.dashboard.engineStats,
      });
    }

    return err(`no route: ${p}`, 404);
  } catch (e) {
    console.error("[soc-engine] error:", e);
    return err("internal error", 500);
  }
});

function projectCentroid(cluster: number): { x: number; y: number } {
  const c = artifacts.clustering;
  const cen = c.kmeans_centroids[cluster];
  const p1 = c.pca_components[0], p2 = c.pca_components[1];
  let x = 0, y = 0;
  for (let i = 0; i < cen.length; i++) {
    x += (cen[i] - c.pca_mean[i]) * p1[i];
    y += (cen[i] - c.pca_mean[i]) * p2[i];
  }
  return { x: Number(x.toFixed(2)), y: Number(y.toFixed(2)) };
}

// ------------------------------------------------------------------ socket.io
const io = new Server(httpServer, {
  // Default path "/socket.io" — REST routes at /api/* pass through untouched.
  // The Caddy gateway forwards everything with ?XTransformPort=3010 to this port,
  // and the frontend connects via io("/?XTransformPort=3010") (default client path).
  cors: { origin: "*", methods: ["GET", "POST"] },
  pingTimeout: 60000,
  pingInterval: 25000,
});
const replay = new ReplayManager(io, state.header, state.testRows);
replay.setMainEngine(state.engine);
replay.attach(io);

io.on("connection", (socket) => {
  socket.emit("hello", { service: "soc-engine", replayApi: true });
});

httpServer.listen(PORT, () => {
  console.log(`[soc-engine] listening on port ${PORT} (REST /api/* + socket.io at path /)`);
});
