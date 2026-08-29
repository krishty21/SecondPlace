/**
 * CipherMind Sentinel — typed API client for the soc-engine (port 3010)
 * and the Next.js LLM endpoints (same origin).
 *
 * Engine routing:
 * - Sandbox/preview (behind the gateway): relative paths + `XTransformPort=3010`.
 * - Local development: `NEXT_PUBLIC_ENGINE_URL=http://localhost:3010` in `.env`
 *   makes the client call the engine directly (CORS is enabled engine-side).
 */

// ------------------------------------------------------------------ types

export type Severity = "Low" | "Medium" | "High" | "Critical";
export type IncidentStatus = "Active" | "Escalating" | "Contained" | "Closed";
export type Epistemics = "Observed" | "Inferred" | "Simulated" | "Prescriptive";

/** A raw UNSW-NB15 flow record (45 CSV fields incl. attack_cat & label). */
export interface RawEvent {
  id: number;
  dur: number;
  proto: string;
  service: string;
  state: string;
  spkts: number;
  dpkts: number;
  sbytes: number;
  dbytes: number;
  rate: number;
  sttl: number;
  dttl: number;
  sload: number;
  dload: number;
  sloss: number;
  dloss: number;
  sinpkt: number;
  dinpkt: number;
  sjit: number;
  djit: number;
  swin: number;
  stcpb: number;
  dtcpb: number;
  dwin: number;
  tcprtt: number;
  synack: number;
  ackdat: number;
  smean: number;
  dmean: number;
  trans_depth: number;
  response_body_len: number;
  ct_srv_src: number;
  ct_state_ttl: number;
  ct_dst_ltm: number;
  ct_src_dport_ltm: number;
  ct_dst_sport_ltm: number;
  ct_dst_src_ltm: number;
  is_ftp_login: number;
  ct_ftp_cmd: number;
  ct_flw_http_mthd: number;
  ct_src_ltm: number;
  ct_srv_dst: number;
  is_sm_ips_ports: number;
  attack_cat: string;
  label: number;
  [key: string]: string | number;
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  contribution: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface Explanation {
  method: "exact-treeshap" | "saabas-path";
  baseline: number;
  topPositive: FeatureAttribution[];
  topNegative: FeatureAttribution[];
  narrative: string;
}

export interface ScoredEvent {
  eventId: string;
  rowIndex: number | null;
  t: number;
  wave?: string;
  raw: RawEvent;
  attackProbability: number;
  binaryVerdict: "Attack" | "Normal";
  attackConfidence: number;
  category: string;
  categoryProbs: Record<string, number>;
  anomalyRaw: number;
  anomalyScore: number;
  riskScore: number;
  severity: Severity;
  cluster: number | null;
  entity: string;
  explanation: Explanation;
  groundTruth?: string;
}

export interface StoryStage {
  index: number;
  title: string;
  detail: string;
  timestamp: number;
  evidenceEventIds: string[];
  epistemics: Epistemics;
}

export interface RiskPoint {
  t: number;
  risk: number;
  count: number;
}

export interface Incident {
  incidentId: string;
  title: string;
  status: IncidentStatus;
  category: string;
  categoryMix: Record<string, number>;
  firstSeen: number;
  lastSeen: number;
  eventCount: number;
  alertCount: number;
  entities: string[];
  riskScore: number;
  riskTrajectory: RiskPoint[];
  severity: Severity;
  meanConfidence: number;
  meanAnomaly: number;
  peakAnomaly: number;
  topContributors: { feature: string; contribution: number }[];
  events: ScoredEvent[];
  story: StoryStage[];
  containmentPlaybook: string[];
}

export interface EngineStats {
  bootScoringMs: number;
  eventsPerSec: number;
  singleEventLatencyMs: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  modelsLoaded: boolean;
  version: string;
  trainedAt: string;
  uptimeSec: number;
  incidentsTracked: number;
  engine: EngineStats;
}

export interface DashboardKpis {
  totalEvents: number;
  totalAlerts: number;
  activeIncidents: number;
  criticalIncidents: number;
  detectionRate: number;
  highRiskTrend: number;
  falsePositiveIndicator: number;
  meanAnomaly: number;
  medianResponseRisk: number;
}

export interface DashboardResponse {
  kpis: DashboardKpis;
  categoryBreakdown: { category: string; count: number; meanRisk: number }[];
  severityBreakdown: { severity: string; count: number }[];
  timeline: {
    t: number;
    events: number;
    alerts: number;
    incidents: number;
    meanRisk: number;
  }[];
  topIncidents: Incident[];
  recentCritical: ScoredEvent[];
  generatedAt: string;
  sampleDescription: string;
  engineStats: EngineStats;
}

export interface IncidentsResponse {
  total: number;
  incidents: Incident[];
}

export interface IncidentDetailResponse {
  incident: Incident;
  groundTruthMix: Record<string, number>;
}

export interface ClusterProfile {
  cluster: number;
  size: number;
  dominant_category: string;
  category_distribution: Record<string, number>;
  top_features: { feature: string; z_score: number }[];
  liveAlertCount: number;
  centroidPca: { x: number; y: number };
}

export interface ScatterPoint {
  x: number;
  y: number;
  cluster: number;
  category: string;
}

export interface PatternsResponse {
  clusters: ClusterProfile[];
  scatter: ScatterPoint[];
  pcaExplainedVariance: number[];
  featuresUsed: string[];
  notes: string;
}

export interface ReliabilityBin {
  bin_low: number;
  bin_high: number;
  mean_predicted: number;
  fraction_positive: number;
  count: number;
}

export interface ThresholdPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ExplainGlobalResponse {
  shapGlobal: {
    expected_value: number;
    method: string;
    features: { feature: string; mean_abs_shap: number }[];
  };
  multiclassGain: { features: { feature: string; gain: number }[] };
  calibration: {
    platt: { a: number; b: number };
    temperature: number;
    chosen_threshold: number;
    threshold_curve: ThresholdPoint[];
    oof_reliability: ReliabilityBin[];
    oof_brier_raw: number;
    oof_brier_platt: number;
  };
  testEvaluation: Record<string, unknown>;
  datasetProfile: Record<string, unknown>;
  methodology: { global: string; local: string };
}

export interface ComparisonRow {
  model: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  pr_auc?: number;
  specificity?: number;
  fit_seconds?: number;
  macro_f1?: number;
  macro_precision?: number;
  macro_recall?: number;
  weighted_f1?: number;
  balanced_accuracy?: number;
  best_iteration?: number;
  full_pipeline_f1?: number;
  [key: string]: unknown;
}

export interface ModelRegistry {
  name: string;
  version: string;
  trained_at: string;
  seed: number;
  dataset: {
    train_file: string;
    test_file: string;
    train_rows: number;
    test_rows: number;
  };
  threshold: number;
  feature_count: number;
  [key: string]: unknown;
}

export interface ModelInfoResponse {
  registry: ModelRegistry;
  comparison: { binary: ComparisonRow[]; multiclass: ComparisonRow[] };
  featureAblation: {
    raw_only_f1?: number;
    full_pipeline_f1?: number;
    delta?: number;
    [key: string]: unknown;
  };
}

export interface EventsResponse {
  category: string;
  count: number;
  events: ScoredEvent[];
}

export interface ReplayStartResponse {
  replayId: string;
  total: number;
  durationMs: number;
}

export interface ReplayStats {
  processed: number;
  total: number;
  alerts: number;
  normals: number;
  incidents: number;
  criticalIncidents: number;
  riskMax: number;
  byCategory: Record<string, number>;
  throughput: number;
}

export interface ReplayTick {
  replayId: string;
  virtualTime: number;
  done: boolean;
  events: ScoredEvent[];
  incidents: Incident[];
  incidentsChanged?: boolean;
  stats: ReplayStats;
}

export interface ReplayStateResponse extends ReplayTick {
  cursor: number;
}

// ------------------------------------------------------------ LLM endpoints

export interface IncidentSummarySections {
  executiveSummary: string;
  technicalAnalysis: string;
  whyThisMatters: string;
  recommendedInvestigation: string[];
  suggestedContainment: string[];
  confidenceCaveats: string;
}

export interface IncidentSummaryResponse {
  source: "llm" | "fallback";
  sections: IncidentSummarySections;
}

export interface AnalystChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AnalystChatResponse {
  reply: string;
  source: "llm" | "fallback";
}

// ---------------------------------------------------------------- client

export class EngineError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "EngineError";
    this.status = status;
  }
}

/**
 * Engine base URL resolution:
 * - Sandbox/preview (behind the gateway): requests stay RELATIVE and carry
 *   `?XTransformPort=3010` so the gateway forwards them to the soc-engine.
 * - Local development: set `NEXT_PUBLIC_ENGINE_URL=http://localhost:3010` in
 *   `.env` and the client calls the engine directly (the engine enables CORS).
 */
const ENGINE_PORT = "3010";
const ENGINE_BASE = (process.env.NEXT_PUBLIC_ENGINE_URL ?? "").replace(/\/+$/, "");

function engineUrl(path: string): string {
  if (ENGINE_BASE) return `${ENGINE_BASE}${path}`;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}XTransformPort=${ENGINE_PORT}`;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(engineUrl(path), { signal, cache: "no-store" });
  if (!res.ok) {
    let message = `engine request failed (${res.status})`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) message = body.error;
    } catch {
      /* body not json — keep default message */
    }
    throw new EngineError(message, res.status);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(engineUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (!res.ok) {
    let message = `engine request failed (${res.status})`;
    try {
      const parsed = (await res.json()) as { error?: string };
      if (parsed?.error) message = parsed.error;
    } catch {
      /* keep default message */
    }
    throw new EngineError(message, res.status);
  }
  return (await res.json()) as T;
}

/** LLM endpoints live on the Next.js server (same origin, no port param). */
async function postAi<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (!res.ok) {
    throw new EngineError(`AI endpoint failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export const socApi = {
  health: (signal?: AbortSignal) => getJson<HealthResponse>("/api/health", signal),
  dashboard: (signal?: AbortSignal) => getJson<DashboardResponse>("/api/dashboard", signal),
  incidents: (params: { limit?: number; severity?: string } = {}, signal?: AbortSignal) => {
    const q = new URLSearchParams();
    if (params.limit) q.set("limit", String(params.limit));
    if (params.severity) q.set("severity", params.severity);
    const qs = q.toString();
    return getJson<IncidentsResponse>(qs ? `/api/incidents?${qs}` : "/api/incidents", signal);
  },
  incidentDetail: (id: string, signal?: AbortSignal) =>
    getJson<IncidentDetailResponse>(`/api/incidents/${encodeURIComponent(id)}`, signal),
  patterns: (signal?: AbortSignal) => getJson<PatternsResponse>("/api/patterns", signal),
  explainGlobal: (signal?: AbortSignal) => getJson<ExplainGlobalResponse>("/api/explain/global", signal),
  modelInfo: (signal?: AbortSignal) => getJson<ModelInfoResponse>("/api/model/info", signal),
  events: (params: { category: string; n?: number; seed?: number }, signal?: AbortSignal) => {
    const q = new URLSearchParams({ category: params.category });
    if (params.n) q.set("n", String(params.n));
    if (params.seed !== undefined) q.set("seed", String(params.seed));
    return getJson<EventsResponse>(`/api/events?${q.toString()}`, signal);
  },
  explainEvent: (event: RawEvent, signal?: AbortSignal) =>
    postJson<ScoredEvent>("/api/explain", { event }, signal),

  replayStart: (speed: number, signal?: AbortSignal) =>
    postJson<ReplayStartResponse>("/api/replay/start", { speed }, signal),
  replayState: (id: string, cursor: number, signal?: AbortSignal) =>
    getJson<ReplayStateResponse>(
      `/api/replay/${encodeURIComponent(id)}/state?cursor=${cursor}`,
      signal
    ),
  replayControl: (id: string, action: string, value?: number, signal?: AbortSignal) =>
    postJson<{ ok: boolean }>(
      `/api/replay/${encodeURIComponent(id)}/control`,
      { action, value },
      signal
    ),

  aiIncidentSummary: (evidence: Record<string, unknown>, signal?: AbortSignal) =>
    postAi<IncidentSummaryResponse>("/api/ai/incident-summary", { evidence }, signal),
  aiAnalystChat: (
    messages: AnalystChatMessage[],
    context: Record<string, unknown>,
    signal?: AbortSignal
  ) => postAi<AnalystChatResponse>("/api/ai/analyst-chat", { messages, context }, signal),
};

/** The 10 UNSW-NB15 class labels (dataset-fixed). */
export const UNSW_CATEGORIES = [
  "Normal",
  "Generic",
  "Exploits",
  "Fuzzers",
  "DoS",
  "Reconnaissance",
  "Analysis",
  "Backdoor",
  "Shellcode",
  "Worms",
] as const;
