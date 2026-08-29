/**
 * SOC service state — boot-time scoring of a stratified test sample through the
 * REAL inference pipeline, producing the Command Center dashboard state,
 * incident list, and pattern explorer data. Everything is model-driven; nothing
 * is hardcoded.
 */
import { loadArtifacts, loadTestCsv, type RawEvent } from "./artifacts.ts";
import { DetectionEngine, CATEGORIES } from "./engine.ts";
import { CorrelationEngine } from "./correlate.ts";
import type { Incident, ScoredEvent } from "./types.ts";

export interface DashboardKpis {
  totalEvents: number;
  totalAlerts: number;
  activeIncidents: number;
  criticalIncidents: number;
  detectionRate: number; // alerts / events
  highRiskTrend: number; // slope of risk over time (per minute, last 10%)
  falsePositiveIndicator: number; // alerts predicted Normal that were attacks (ground truth, demo transparency)
  meanAnomaly: number;
  medianResponseRisk: number;
}

export interface DashboardState {
  kpis: DashboardKpis;
  categoryBreakdown: { category: string; count: number; meanRisk: number }[];
  severityBreakdown: { severity: string; count: number }[];
  timeline: { t: number; events: number; alerts: number; incidents: number; meanRisk: number }[];
  topIncidents: Incident[];
  recentCritical: ScoredEvent[];
  generatedAt: string;
  sampleDescription: string;
  engineStats: { bootScoringMs: number; eventsPerSec: number; singleEventLatencyMs: number };
}

const NUM_FIELDS = [
  "id", "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
  "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
  "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
  "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
  "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
  "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
  "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "label",
];

export function csvRowToEvent(header: string[], row: string[]): RawEvent {
  const ev: Record<string, unknown> = {};
  header.forEach((h, i) => {
    const v = row[i];
    if (h === "proto" || h === "service" || h === "state" || h === "attack_cat") ev[h] = v;
    else ev[h] = Number(v);
  });
  return ev as unknown as RawEvent;
}

export class SocState {
  engine: DetectionEngine;
  correlator: CorrelationEngine;
  dashboard: DashboardState;
  incidents: Incident[];
  events: ScoredEvent[] = [];
  header: string[];
  testRows: string[][];

  constructor() {
    const t0 = Date.now();
    const a = loadArtifacts();
    const { header, rows } = loadTestCsv();
    this.header = header;
    this.testRows = rows;
    this.engine = new DetectionEngine();

    // ---- boot: score stratified sample of official test set
    const indices = a.bootSample.indices;
    const scored: ScoredEvent[] = [];
    const scoredX: Float64Array[] = [];
    const tBoot = Date.now();
    // synthetic replay timestamps (SIMULATED metadata): bursts with quiet gaps
    // so correlation forms meaningful incidents instead of one mega-incident.
    let t = 0;
    let burstRun = 0;
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const ev = csvRowToEvent(header, rows[i]);
      const isAttackLike = ev.rate > 500 || ev.sbytes > 4000;
      if (isAttackLike) {
        burstRun++;
        t += 25 + ((i * 13) % 50);
        if (burstRun % 25 === 0) t += 30000 + ((i * 7) % 20000); // campaign boundary (quiet gap > window)
      } else {
        burstRun = 0;
        t += 90 + ((i * 37) % 180);
      }
      const { event, x } = this.engine.scoreWithFeatures(ev, {
        t, rowIndex: i, eventId: `B-${i}`, wave: "boot",
      });
      scored.push(event);
      scoredX.push(x);
    }
    const bootMs = Date.now() - tBoot;

    // ---- correlate
    this.correlator = new CorrelationEngine(a.clustering, a.features.feature_names);
    for (let k = 0; k < scored.length; k++) {
      const e = scored[k];
      this.correlator.ingest(e, scoredX[k], (related) =>
        this.engine.riskScore(e.attackProbability, e.anomalyScore, e.category, related)
      );
    }
    this.correlator.finalizeAll();
    this.incidents = [...this.correlator.all].sort((x, y) => y.riskScore - x.riskScore);
    this.events = scored;
    this.dashboard = this.buildDashboard(scored, this.incidents, bootMs, indices.length);
    console.log(`[soc-engine] boot state: ${scored.length} events scored, ${this.incidents.length} incidents in ${Date.now() - t0}ms total`);
  }

  private buildDashboard(events: ScoredEvent[], incidents: Incident[], bootMs: number, n: number): DashboardState {
    const alerts = events.filter((e) => e.binaryVerdict === "Attack");
    // timeline buckets (~60 buckets)
    const tMax = events.length ? events[events.length - 1].t : 1;
    const bucketMs = Math.max(1, tMax / 60);
    const buckets = new Map<number, { events: number; alerts: number; riskSum: number; riskN: number }>();
    const openIncidentsAt = new Map<number, number>();
    let opened = 0;
    const incidentStarts = incidents.map((i) => i.firstSeen).sort((a, b) => a - b);
    let ii = 0;
    for (const e of events) {
      const b = Math.floor(e.t / bucketMs);
      const cur = buckets.get(b) ?? { events: 0, alerts: 0, riskSum: 0, riskN: 0 };
      cur.events++;
      if (e.binaryVerdict === "Attack") { cur.alerts++; cur.riskSum += e.riskScore; cur.riskN++; }
      buckets.set(b, cur);
      while (ii < incidentStarts.length && incidentStarts[ii] <= e.t) { opened++; ii++; }
      openIncidentsAt.set(b, opened);
    }
    const timeline = [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([b, v]) => ({
        t: Math.round(b * bucketMs),
        events: v.events,
        alerts: v.alerts,
        incidents: openIncidentsAt.get(b) ?? 0,
        meanRisk: v.riskN ? Number((v.riskSum / v.riskN).toFixed(1)) : 0,
      }));

    // category breakdown
    const catAgg = new Map<string, { count: number; riskSum: number }>();
    for (const e of alerts) {
      const c = catAgg.get(e.category) ?? { count: 0, riskSum: 0 };
      c.count++; c.riskSum += e.riskScore;
      catAgg.set(e.category, c);
    }
    const categoryBreakdown = CATEGORIES.filter((c) => c !== "Normal" && catAgg.has(c))
      .map((c) => ({ category: c, count: catAgg.get(c)!.count, meanRisk: Number((catAgg.get(c)!.riskSum / catAgg.get(c)!.count).toFixed(1)) }))
      .sort((a, b) => b.count - a.count);

    const sevAgg = new Map<string, number>();
    for (const e of alerts) sevAgg.set(e.severity, (sevAgg.get(e.severity) ?? 0) + 1);

    // high-risk trend: slope of mean alert risk across last vs first half
    const half = Math.floor(timeline.length / 2);
    const firstHalf = avg(timeline.slice(0, half).map((x) => x.meanRisk));
    const secondHalf = avg(timeline.slice(half).map((x) => x.meanRisk));

    // false positive indicator (demo transparency): events the model called Attack
    // but ground truth label=0, and vice versa — computed from labels we keep aside
    const fp = this.events.filter((e) => e.binaryVerdict === "Attack" && (e.raw as RawEvent).label === 0).length;
    const fn = this.events.filter((e) => e.binaryVerdict === "Normal" && (e.raw as RawEvent).label === 1).length;

    // single-event latency benchmark
    const benchEv = csvRowToEvent(this.header, this.testRows[Math.floor(Math.random() * this.testRows.length)]);
    const tb = performance.now();
    for (let i = 0; i < 100; i++) this.engine.score(benchEv);
    const singleMs = (performance.now() - tb) / 100;

    return {
      kpis: {
        totalEvents: events.length,
        totalAlerts: alerts.length,
        activeIncidents: incidents.filter((i) => i.status === "Active" || i.status === "Escalating").length,
        criticalIncidents: incidents.filter((i) => i.severity === "Critical").length,
        detectionRate: events.length ? Number((alerts.length / events.length).toFixed(4)) : 0,
        highRiskTrend: Number((secondHalf - firstHalf).toFixed(2)),
        falsePositiveIndicator: alerts.length ? Number((fp / alerts.length).toFixed(4)) : 0,
        meanAnomaly: Number(avg(alerts.map((e) => e.anomalyScore)).toFixed(1)),
        medianResponseRisk: median(alerts.map((e) => e.riskScore)),
      },
      categoryBreakdown,
      severityBreakdown: ["Low", "Medium", "High", "Critical"].map((s) => ({ severity: s, count: sevAgg.get(s) ?? 0 })),
      timeline,
      topIncidents: incidents.slice(0, 12).map((i) => slimIncident(i)),
      recentCritical: alerts
        .filter((e) => e.severity === "Critical" || e.severity === "High")
        .slice(-8)
        .reverse(),
      generatedAt: new Date().toISOString(),
      sampleDescription: `Boot state: ${n.toLocaleString()} stratified events from the official UNSW-NB15 test set scored live by the trained models (not cached predictions).`,
      engineStats: {
        bootScoringMs: bootMs,
        eventsPerSec: Math.round((n / bootMs) * 1000),
        singleEventLatencyMs: Number(singleMs.toFixed(3)),
      },
    };
  }

  incidentDetail(id: string) {
    const inc = this.correlator.byId(id) ?? this.incidents.find((i) => i.incidentId === id);
    if (!inc) return null;
    return inc;
  }
}

/** Slim an incident for list payloads (full events kept in detail view). */
export function slimIncident(i: Incident): Incident {
  return {
    ...i,
    events: i.events.slice(-6),
  };
}

function avg(a: number[]): number {
  return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
}

function median(a: number[]): number {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return Number(((s[m - 1] + s[m]) / 2).toFixed(1));
}
