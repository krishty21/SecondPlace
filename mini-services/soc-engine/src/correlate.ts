/**
 * Incident Correlation Engine — turns noisy alerts into prioritized incidents
 * with attack stories. Operates on model outputs ONLY (predicted category,
 * behavioral feature vectors, risk trajectory) plus SIMULATED replay timestamps
 * (clearly labeled — the supplied CSVs contain no timestamps).
 *
 * Correlation key: same predicted-category family + temporal proximity (window)
 * + behavioral similarity (euclidean distance on the standardized behavioral
 * subspace) + simulated pseudo-entity overlap.
 */
import type { ScoredEvent, Incident, StoryStage, Severity } from "./types.ts";
import type { ClusteringArt } from "./artifacts.ts";

const WINDOW_MS = 45 * 1000; // 45-second sliding window (simulated time)
const MAX_EVENTS_PER_INCIDENT = 400;
const MAX_ALERTS_PER_INCIDENT = 80; // prevents mega-incidents in continuous bursts

export interface CorrelationConfig {
  windowMs: number;
  behavioralDistanceThreshold: number;
  minAlertsForIncident: number;
}

export class CorrelationEngine {
  private config: CorrelationConfig;
  private clustering: ClusteringArt;
  private featureNames: string[];
  private incidents: Incident[] = [];
  private openIncidents: Incident[] = [];
  private idCounter = 1;
  private idPrefix: string;
  private bvecCache = new Map<string, Float64Array>();

  constructor(clustering: ClusteringArt, featureNames: string[], config?: Partial<CorrelationConfig> & { idPrefix?: string }) {
    this.clustering = clustering;
    this.featureNames = featureNames;
    this.idPrefix = config?.idPrefix ?? "INC";
    this.config = {
      windowMs: config?.windowMs ?? WINDOW_MS,
      behavioralDistanceThreshold: config?.behavioralDistanceThreshold ?? 42,
      minAlertsForIncident: config?.minAlertsForIncident ?? 2,
    };
  }

  /** standardized behavioral vector (from the full transformed feature space). */
  private bvec(e: ScoredEvent, x?: Float64Array): Float64Array | null {
    let v = this.bvecCache.get(e.eventId);
    if (!v) {
      if (!x) return null;
      const idx = this.clustering.feature_indices;
      v = new Float64Array(idx.length);
      for (let i = 0; i < idx.length; i++) {
        v[i] = (x[idx[i]] - this.clustering.scaler_mean[i]) / (this.clustering.scaler_std[i] || 1);
      }
      if (this.bvecCache.size > 20000) this.bvecCache.clear();
      this.bvecCache.set(e.eventId, v);
    }
    return v;
  }

  private distance(a: Float64Array, b: Float64Array): number {
    let d = 0;
    for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; d += t * t; }
    return Math.sqrt(d);
  }

  /** Feed one scored event (+ its transformed feature vector) into the streaming correlation. */
  ingest(e: ScoredEvent, x: Float64Array, engineRisk: (related: number) => number): Incident | null {
    if (e.binaryVerdict !== "Attack") return null;
    const cat = e.category;
    const bv = this.bvec(e, x);

    let bestIncident: Incident | null = null;
    let bestScore = -Infinity;
    for (const inc of this.openIncidents) {
      if (e.t - inc.lastSeen > this.config.windowMs) continue;
      if (inc.alertCount >= MAX_ALERTS_PER_INCIDENT) continue; // full — start a new one
      const sameFamily = inc.category === cat || isRelatedCategory(inc.category, cat);
      const timePenalty = (e.t - inc.lastSeen) / this.config.windowMs;
      const lastBv = this.bvec(inc.events[inc.events.length - 1]);
      const behavioralDist = lastBv && bv ? this.distance(bv, lastBv) : this.config.behavioralDistanceThreshold;
      const entityOverlap = inc.entities.includes(e.entity) ? 1 : 0;
      const s =
        (sameFamily ? 1.0 : 0.15) +
        entityOverlap * 0.5 -
        timePenalty * 0.35 -
        behavioralDist / this.config.behavioralDistanceThreshold;
      if (s > bestScore) { bestScore = s; bestIncident = inc; }
    }

    let incident: Incident;
    if (bestIncident && bestScore > 0.45) {
      incident = bestIncident;
      incident.events.push(e);
      if (incident.events.length > MAX_EVENTS_PER_INCIDENT) incident.events.shift();
      incident.lastSeen = e.t;
      incident.eventCount++;
      incident.alertCount++;
      if (!incident.entities.includes(e.entity)) incident.entities.push(e.entity);
      incident.categoryMix[cat] = (incident.categoryMix[cat] ?? 0) + 1;
      incident.meanConfidence = (incident.meanConfidence * (incident.alertCount - 1) + e.attackConfidence) / incident.alertCount;
      incident.meanAnomaly = (incident.meanAnomaly * (incident.alertCount - 1) + e.anomalyScore) / incident.alertCount;
      incident.peakAnomaly = Math.max(incident.peakAnomaly, e.anomalyScore);
      // category transition => escalation
      if (cat !== incident.category && e.riskScore > incident.riskScore) {
        incident.category = cat; // dominant may shift on escalation
        incident.status = "Escalating";
      }
    } else {
      incident = {
        incidentId: `${this.idPrefix}-${String(this.idCounter++).padStart(4, "0")}`,
        title: `${cat} activity pattern`,
        status: "Active",
        category: cat,
        categoryMix: { [cat]: 1 },
        firstSeen: e.t,
        lastSeen: e.t,
        eventCount: 1,
        alertCount: 1,
        entities: [e.entity],
        riskScore: 0,
        riskTrajectory: [],
        severity: "Low",
        meanConfidence: e.attackConfidence,
        meanAnomaly: e.anomalyScore,
        peakAnomaly: e.anomalyScore,
        topContributors: [],
        events: [e],
        story: [],
        containmentPlaybook: [],
      };
      this.incidents.push(incident);
      this.openIncidents.push(incident);
    }

    // risk evolves with correlation boost
    const risk = engineRisk(incident.alertCount);
    incident.riskScore = Math.max(incident.riskScore, risk);
    incident.severity = severityOf(incident.riskScore);
    incident.riskTrajectory.push({ t: e.t, risk: Number(risk.toFixed(1)), count: incident.alertCount });
    if (incident.riskTrajectory.length > 200) incident.riskTrajectory.shift();
    if (incident.riskScore >= 75) incident.status = incident.status === "Contained" ? "Contained" : "Escalating";

    // close stale incidents
    for (let i = this.openIncidents.length - 1; i >= 0; i--) {
      if (e.t - this.openIncidents[i].lastSeen > this.config.windowMs) {
        const inc = this.openIncidents[i];
        inc.status = inc.status === "Escalating" ? "Contained" : inc.status === "Active" ? "Closed" : inc.status;
        this.openIncidents.splice(i, 1);
      }
    }
    return incident;
  }

  /** Finalize stories for all incidents (call after replay completes or on-demand). */
  finalizeAll(): void {
    for (const inc of this.incidents) this.buildStory(inc);
  }

  buildStory(inc: Incident): void {
    const evs = inc.events;
    const stages: StoryStage[] = [];
    if (evs.length === 0) return;
    const first = evs[0];
    stages.push({
      index: 1,
      title: "Initial suspicious behavior",
      detail: `First anomalous flow detected: predicted ${first.category} (confidence ${(first.attackConfidence * 100).toFixed(0)}%, anomaly ${first.anomalyScore.toFixed(0)}/100, risk ${first.riskScore.toFixed(0)}).`,
      timestamp: first.t,
      evidenceEventIds: [first.eventId],
      epistemics: "Observed",
    });
    // repeated anomalies
    const repeated = evs.filter((e) => e.anomalyScore >= 40);
    if (repeated.length >= 3) {
      const third = repeated[2];
      stages.push({
        index: 2,
        title: "Repeated anomalous behavior",
        detail: `${repeated.length} correlated flows exceeded the normal-behavior anomaly threshold (40/100), indicating a sustained pattern rather than a single outlier.`,
        timestamp: third.t,
        evidenceEventIds: repeated.slice(0, 5).map((e) => e.eventId),
        epistemics: "Observed",
      });
    }
    // category transition
    const cats = [...new Set(evs.map((e) => e.category))];
    if (cats.length > 1) {
      const transition = evs.find((e) => e.category !== cats[0]);
      if (transition) {
        stages.push({
          index: 3,
          title: "Attack pattern transition",
          detail: `Behavioral pattern shifted from ${cats[0]} to ${transition.category}. The campaign may be progressing through multiple attack stages (observed across ${cats.length} predicted categories: ${cats.join(", ")}).`,
          timestamp: transition.t,
          evidenceEventIds: [transition.eventId],
          epistemics: "Inferred",
        });
      }
    }
    // peak
    const peak = evs.reduce((a, b) => (b.riskScore > a.riskScore ? b : a));
    stages.push({
      index: 4,
      title: "Incident peak",
      detail: `Highest-risk flow observed: risk ${peak.riskScore.toFixed(0)}/100 (${peak.severity}), anomaly ${peak.anomalyScore.toFixed(0)}/100, predicted ${peak.category} with ${(peak.attackConfidence * 100).toFixed(0)}% confidence.`,
      timestamp: peak.t,
      evidenceEventIds: [peak.eventId],
      epistemics: "Observed",
    });
    stages.push({
      index: 5,
      title: "Recommended response",
      detail: `${inc.alertCount} correlated alerts consolidated into this incident across ${inc.entities.length} simulated pseudo-entities over ${((inc.lastSeen - inc.firstSeen) / 1000).toFixed(0)}s of replay time.`,
      timestamp: inc.lastSeen,
      evidenceEventIds: [],
      epistemics: "Prescriptive",
    });
    inc.story = stages;
    inc.containmentPlaybook = playbookFor(inc.category);
    inc.topContributors = aggregateContributors(evs);
  }

  get all(): Incident[] {
    return this.incidents;
  }

  byId(id: string): Incident | undefined {
    return this.incidents.find((i) => i.incidentId === id);
  }
}

function severityOf(risk: number): Severity {
  if (risk <= 24) return "Low";
  if (risk <= 49) return "Medium";
  if (risk <= 74) return "High";
  return "Critical";
}

function isRelatedCategory(a: string, b: string): boolean {
  const families: string[][] = [
    ["Exploits", "Shellcode", "Backdoor"],
    ["Reconnaissance", "Analysis"],
    ["DoS", "Generic"],
    ["Fuzzers", "Analysis"],
  ];
  return families.some((f) => f.includes(a) && f.includes(b));
}

function playbookFor(category: string): string[] {
  const playbooks: Record<string, string[]> = {
    Exploits: [
      "Isolate affected hosts (simulated entities) and capture full packet data for forensic review",
      "Review service/version exposure on targeted ports and apply vendor patches",
      "Hunt for follow-on persistence: unusual outbound flows and repeated exploit retries",
    ],
    Shellcode: [
      "Capture memory forensics on endpoints linked to the implicated pseudo-entities",
      "Block originating flow signatures at the perimeter and monitor for repeats",
      "Check for code-execution follow-up: unexpected child processes, new listeners",
    ],
    Reconnaissance: [
      "Compare scanning sources against known recon patterns; rate-limit repeated probes",
      "Verify exposure of scanned services; close unused ports",
      "Escalate to active monitoring — recon often precedes exploitation",
    ],
    DoS: [
      "Enable upstream traffic scrubbing / rate limiting for targeted services",
      "Check service health and capacity; confirm degradation vs. failure",
      "Correlate with volumetric baselines to distinguish attack bursts from load spikes",
    ],
    Generic: [
      "Validate whether generic blocking rules already cover this signature family",
      "Sample flows for manual review to confirm malicious intent",
    ],
    Backdoor: [
      "Inspect implicated pseudo-entity flow histories for command-and-control periodicity",
      "Quarantine suspected hosts; rotate credentials that transited affected services",
    ],
    Analysis: [
      "Review analyst tooling noise vs. genuine probing (this class contains port sweeps)",
      "Cross-check with firewall logs for attempted DNS/VPN tunneling",
    ],
    Fuzzers: [
      "Check targeted service crash logs; fuzzing often precedes exploit development",
      "Harden input validation on the affected services",
    ],
    Worms: [
      "Immediate containment: isolate implicated entities (self-propagating behavior)",
      "Patch the propagation vector across the estate; verify removal artifacts",
    ],
    Normal: ["No action required"],
  };
  return playbooks[category] ?? ["Triage alert manually"];
}

function aggregateContributors(evs: ScoredEvent[]): { feature: string; contribution: number }[] {
  const agg = new Map<string, number>();
  for (const e of evs) {
    for (const a of e.explanation.topPositive) {
      agg.set(a.feature, (agg.get(a.feature) ?? 0) + a.contribution);
    }
  }
  return [...agg.entries()]
    .map(([feature, contribution]) => ({ feature, contribution: Number(contribution.toFixed(4)) }))
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, 8);
}
