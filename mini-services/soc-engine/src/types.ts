/** Shared types for the CipherMind Sentinel inference engine. */

export interface RawEvent {
  id: number;
  dur: number;
  proto: string;
  service: string;
  state: string;
  spkts: number; dpkts: number; sbytes: number; dbytes: number;
  rate: number; sttl: number; dttl: number; sload: number; dload: number;
  sloss: number; dloss: number; sinpkt: number; dinpkt: number;
  sjit: number; djit: number; swin: number; stcpb: number; dtcpb: number; dwin: number;
  tcprtt: number; synack: number; ackdat: number; smean: number; dmean: number;
  trans_depth: number; response_body_len: number;
  ct_srv_src: number; ct_state_ttl: number; ct_dst_ltm: number;
  ct_src_dport_ltm: number; ct_dst_sport_ltm: number; ct_dst_src_ltm: number;
  is_ftp_login: number; ct_ftp_cmd: number; ct_flw_http_mthd: number;
  ct_src_ltm: number; ct_srv_dst: number; is_sm_ips_ports: number;
  attack_cat: string;
  label: number;
}

export type Severity = "Low" | "Medium" | "High" | "Critical";

export interface FeatureAttribution {
  feature: string;
  value: number;
  contribution: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface ScoredEvent {
  eventId: string;
  rowIndex: number | null; // index into test CSV when replaying real data
  raw: RawEvent;
  /** synthetic replay timestamp (ms offset) — clearly simulated metadata */
  t: number;
  attackProbability: number; // calibrated
  binaryVerdict: "Attack" | "Normal";
  attackConfidence: number;
  category: string; // final predicted category (Normal if verdict normal)
  categoryProbs: Record<string, number>;
  anomalyRaw: number;
  anomalyScore: number; // 0-100 normalized
  riskScore: number;
  severity: Severity;
  cluster: number | null;
  entity: string; // simulated pseudo-entity (clearly labeled)
  explanation: {
    method: "exact-treeshap" | "saabas-path";
    baseline: number;
    topPositive: FeatureAttribution[];
    topNegative: FeatureAttribution[];
    narrative: string;
  };
  wave?: string;
  groundTruth?: string; // for demo transparency (not used by the engine)
}

export interface Alert extends ScoredEvent {
  alertId: string;
  incidentId: string | null;
}

export interface StoryStage {
  index: number;
  title: string;
  detail: string;
  timestamp: number;
  evidenceEventIds: string[];
  epistemics: "Observed" | "Inferred" | "Simulated" | "Prescriptive";
}

export interface Incident {
  incidentId: string;
  title: string;
  status: "Active" | "Escalating" | "Contained" | "Closed";
  category: string;
  categoryMix: Record<string, number>;
  firstSeen: number;
  lastSeen: number;
  eventCount: number;
  alertCount: number;
  entities: string[];
  riskScore: number; // incident-level peak
  riskTrajectory: { t: number; risk: number; count: number }[];
  severity: Severity;
  meanConfidence: number;
  meanAnomaly: number;
  peakAnomaly: number;
  topContributors: { feature: string; contribution: number }[];
  events: ScoredEvent[]; // capped
  story: StoryStage[];
  containmentPlaybook: string[];
}

export interface ReplayTick {
  replayId: string;
  virtualTime: number;
  events: ScoredEvent[];
  incidents: Incident[];
  stats: ReplayStats;
  done: boolean;
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
  throughput: number; // events per virtual second
}
