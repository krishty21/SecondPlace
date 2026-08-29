/**
 * Real-Time Detection Replay engine.
 * Replays REAL UNSW-NB15 test events (deterministic selection) through the REAL
 * trained models, with SIMULATED timestamps (clearly labeled). Streams ticks
 * over socket.io; supports play/pause/speed control and REST polling fallback.
 */
import { Server, type Socket } from "socket.io";
import type { HttpServer } from "http";
import { loadArtifacts, type RawEvent } from "./artifacts.ts";
import { DetectionEngine } from "./engine.ts";
import { CorrelationEngine } from "./correlate.ts";
import { csvRowToEvent } from "./state.ts";
import type { Incident, ReplayStats, ReplayTick, ScoredEvent } from "./types.ts";

interface ReplaySession {
  id: string;
  engine: DetectionEngine;
  correlator: CorrelationEngine;
  scored: ScoredEvent[];
  xs: Float64Array[];
  cursor: number;
  virtualTime: number;
  playing: boolean;
  speed: number; // multiplier over 8 events/sec virtual baseline
  createdAt: number;
  stats: ReplayStats;
  timer: ReturnType<typeof setTimeout> | null;
  done: boolean;
  touchedIncidents: Set<string>;
}

const TICK_MS = 300;

export class ReplayManager {
  private sessions = new Map<string, ReplaySession>();
  private io: Server;
  private artifacts: ReturnType<typeof loadArtifacts>;
  private header: string[];
  private testRows: string[][];
  private idCounter = 1;

  constructor(io: Server, header: string[], testRows: string[][]) {
    this.io = io;
    this.artifacts = loadArtifacts();
    this.header = header;
    this.testRows = testRows;
  }

  start(opts: { speed?: number; engine: DetectionEngine }): { replayId: string; total: number; durationMs: number } {
    const id = `RP-${Date.now().toString(36)}-${this.idCounter++}`;
    const engine = opts.engine;
    const seq = this.artifacts.demoSequence;
    // precompute scores for all replay events (real model, real rows)
    const scored: ScoredEvent[] = [];
    const xs: Float64Array[] = [];
    const correlator = new CorrelationEngine(this.artifacts.clustering, this.artifacts.features.feature_names, { idPrefix: "RINC" });
    for (const e of seq.events) {
      const ev = csvRowToEvent(this.header, this.testRows[e.i]);
      const { event, x } = engine.scoreWithFeatures(ev, {
        t: e.t, rowIndex: e.i, eventId: `R-${e.i}`, wave: e.wave, includeGroundTruth: true,
      });
      scored.push(event);
      xs.push(x);
    }
    const session: ReplaySession = {
      id,
      engine,
      correlator,
      scored,
      xs,
      cursor: 0,
      virtualTime: 0,
      playing: false,
      speed: opts.speed ?? 1,
      createdAt: Date.now(),
      timer: null,
      done: false,
      touchedIncidents: new Set(),
      stats: {
        processed: 0, total: scored.length, alerts: 0, normals: 0, incidents: 0,
        criticalIncidents: 0, riskMax: 0, byCategory: {}, throughput: 0,
      },
    };
    this.sessions.set(id, session);
    return { replayId: id, total: scored.length, durationMs: seq.duration_ms };
  }

  control(id: string, action: "play" | "pause" | "speed" | "seek" | "stop", value?: number): boolean {
    const s = this.sessions.get(id);
    if (!s) return false;
    switch (action) {
      case "play":
        if (!s.done && !s.playing) { s.playing = true; this.scheduleTick(s); }
        return true;
      case "pause":
        s.playing = false;
        if (s.timer) { clearTimeout(s.timer); s.timer = null; }
        return true;
      case "speed":
        s.speed = Math.max(0.5, Math.min(16, value ?? 1));
        return true;
      case "seek": {
        const frac = Math.max(0, Math.min(1, (value ?? 0) / 100));
        this.resetSession(s);
        s.cursor = Math.floor(frac * s.scored.length);
        for (let i = 0; i < s.cursor; i++) this.processIndex(s, i, false);
        return true;
      }
      case "stop":
        s.playing = false;
        if (s.timer) { clearTimeout(s.timer); s.timer = null; }
        this.resetSession(s);
        return true;
    }
  }

  private resetSession(s: ReplaySession) {
    s.cursor = 0;
    s.virtualTime = 0;
    s.done = false;
    s.correlator = new CorrelationEngine(this.artifacts.clustering, this.artifacts.features.feature_names, { idPrefix: "RINC" });
    s.touchedIncidents.clear();
    s.stats = { processed: 0, total: s.scored.length, alerts: 0, normals: 0, incidents: 0, criticalIncidents: 0, riskMax: 0, byCategory: {}, throughput: 0 };
  }

  private scheduleTick(s: ReplaySession) {
    if (s.timer) { clearTimeout(s.timer); s.timer = null; }
    s.timer = setTimeout(() => this.tick(s), TICK_MS);
  }

  /** Advance the virtual clock by TICK_MS * speed of REAL time... but the demo
   * sequence defines event times in virtual ms; default replay ~8 ev/s. We map
   * virtual ms -> real ms at 1:1 * speed. */
  private tick(s: ReplaySession) {
    if (!s.playing) return;
    try {
      const dtVirtual = TICK_MS * s.speed;
      s.virtualTime += dtVirtual;
      const startCursor = s.cursor;
      while (s.cursor < s.scored.length && s.scored[s.cursor].t <= s.virtualTime) {
        this.processIndex(s, s.cursor, true);
        s.cursor++;
      }
      if (s.cursor > startCursor || s.done) {
        const changedIncidents = [...s.touchedIncidents]
          .map((id) => s.correlator.byId(id))
          .filter((i): i is Incident => Boolean(i));
        this.emitTick(s, s.scored.slice(startCursor, s.cursor), changedIncidents);
      }
      if (s.cursor >= s.scored.length) {
        if (!s.done) {
          s.done = true;
          s.playing = false;
          s.correlator.finalizeAll();
          this.emitTick(s, [], [...s.correlator.all], true);
        }
        if (s.timer) { clearTimeout(s.timer); s.timer = null; }
        return;
      }
      this.scheduleTick(s);
    } catch (e) {
      console.error("[replay] tick error:", e);
      s.playing = false;
      if (s.timer) { clearTimeout(s.timer); s.timer = null; }
      this.io.to(s.id).emit("replay:error", { message: String(e) });
    }
  }

  private processIndex(s: ReplaySession, i: number, markTouched: boolean) {
    const e = s.scored[i];
    const x = s.xs[i];
    s.stats.processed++;
    if (e.binaryVerdict === "Attack") {
      s.stats.alerts++;
      s.stats.byCategory[e.category] = (s.stats.byCategory[e.category] ?? 0) + 1;
      const inc = s.correlator.ingest(e, x, (related) =>
        s.engine.riskScore(e.attackProbability, e.anomalyScore, e.category, related)
      );
      if (inc && markTouched) s.touchedIncidents.add(inc.incidentId);
      if (inc) {
        s.stats.riskMax = Math.max(s.stats.riskMax, inc.riskScore);
      }
    } else {
      s.stats.normals++;
    }
    s.virtualTime = Math.max(s.virtualTime, e.t);
    // recompute incident count stats periodically
    const all = s.correlator.all;
    s.stats.incidents = all.length;
    s.stats.criticalIncidents = all.filter((i) => i.severity === "Critical").length;
    s.stats.throughput = s.virtualTime > 0 ? Number(((s.stats.processed / s.virtualTime) * 1000).toFixed(1)) : 0;
  }

  private emitTick(s: ReplaySession, events: ScoredEvent[], incidents: Incident[], done = false) {
    const payload: ReplayTick & { incidentsChanged: boolean } = {
      replayId: s.id,
      virtualTime: s.virtualTime,
      events: events.map((e) => ({ ...e, explanation: { ...e.explanation, topPositive: e.explanation.topPositive.slice(0, 3), topNegative: [] } })),
      incidents: incidents.map((i) => ({ ...i, events: i.events.slice(-4) })),
      incidentsChanged: incidents.length > 0,
      stats: s.stats,
      done,
    };
    this.io.to(s.id).emit("replay:tick", payload);
  }

  /** REST fallback snapshot */
  snapshot(id: string, since: number): ReplayTick | null {
    const s = this.sessions.get(id);
    if (!s) return null;
    const events = s.scored.slice(Math.min(since, s.cursor), s.cursor);
    return {
      replayId: s.id,
      virtualTime: s.virtualTime,
      events: events.slice(-200),
      incidents: [...s.correlator.all].sort((a, b) => b.riskScore - a.riskScore).slice(0, 20),
      stats: s.stats,
      done: s.done,
    };
  }

  getSession(id: string) {
    return this.sessions.get(id);
  }

  /** Wire socket.io handlers */
  attach(io: Server) {
    io.on("connection", (socket: Socket) => {
      socket.on("replay:join", (data: { replayId: string }) => {
        if (this.sessions.has(data.replayId)) socket.join(data.replayId);
      });
      socket.on("replay:control", (data: { replayId: string; action: "play" | "pause" | "speed" | "seek" | "stop"; value?: number }) => {
        const ok = this.control(data.replayId, data.action, data.value);
        if (!ok) socket.emit("replay:error", { message: `unknown replay ${data.replayId}` });
      });
      socket.on("replay:start", (data: { speed?: number }) => {
        // engine injected at manager construction time — reuse main engine
        const res = this.start({ speed: data.speed ?? 1, engine: (this as unknown as { mainEngine: DetectionEngine }).mainEngine });
        socket.emit("replay:started", res);
        socket.join(res.replayId);
      });
    });
  }

  setMainEngine(e: DetectionEngine) {
    (this as unknown as { mainEngine: DetectionEngine }).mainEngine = e;
  }
}
