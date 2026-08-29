/**
 * Detection engine — orchestrates the full scoring pipeline per event:
 * features -> Model A (binary, Platt-calibrated) -> Model B (multiclass, temperature)
 * -> Model C (anomaly) -> Model D (transparent risk) -> explanation + clustering.
 */
import { loadArtifacts, type RawEvent } from "./artifacts.ts";
import { FeaturePipeline, FEATURE_LABELS } from "./features.ts";
import { LgbmModel, sigmoid, softmax } from "./lgbm.ts";
import { AnomalyDetector } from "./iforest.ts";
import type { FeatureAttribution, ScoredEvent, Severity } from "./types.ts";

const CATS = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
  "Normal", "Reconnaissance", "Shellcode", "Worms"];

function hash32(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export class DetectionEngine {
  bin: LgbmModel;
  mc: LgbmModel;
  anomaly: AnomalyDetector;
  pipeline: FeaturePipeline;
  threshold: number;
  plattA: number; plattB: number; temperature: number;
  nFeatures: number;
  riskWeights: Record<string, number>;
  categorySeverity: Record<string, number>;
  rarityWeights: Record<string, number>;
  corrSaturation: number;
  featureNames: string[];
  clustering: ReturnType<typeof loadArtifacts>["clustering"];
  shapCache: Record<string, { b: number[]; mc?: { class: string; top: [string, number][] } }>;
  shapExpected: number;

  constructor() {
    const a = loadArtifacts();
    this.bin = new LgbmModel(a.binary);
    this.mc = new LgbmModel(a.multiclass);
    this.anomaly = new AnomalyDetector(a.iforest);
    this.pipeline = new FeaturePipeline(a.features);
    this.threshold = a.registry.threshold;
    this.plattA = a.registry.platt.a;
    this.plattB = a.registry.platt.b;
    this.temperature = a.registry.temperature;
    this.nFeatures = a.features.feature_names.length;
    this.featureNames = a.features.feature_names;
    this.riskWeights = a.registry.risk_config.weights;
    this.categorySeverity = a.registry.risk_config.category_severity;
    this.rarityWeights = a.registry.risk_config.rarity_weights ?? {};
    this.corrSaturation = a.registry.risk_config.correlation_alert_saturation;
    this.clustering = a.clustering;
    this.shapCache = a.shapCache;
    this.shapExpected = a.shapGlobal.expected_value;
  }

  binaryProb(x: Float64Array): { logit: number; prob: number } {
    const logit = this.bin.predictRaw(x);
    return { logit, prob: sigmoid(this.plattA * logit + this.plattB) };
  }

  categoryProbs(x: Float64Array): { logits: Float64Array; probs: Float64Array } {
    const logits = new Float64Array(CATS.length);
    this.mc.predictRawMulti(x, logits);
    return { logits, probs: softmax(logits, this.temperature) };
  }

  riskScore(attackProb: number, anomalyNorm: number, category: string, relatedAlerts: number): number {
    const w = this.riskWeights;
    const sev = this.categorySeverity[category] ?? 0.3;
    const rarity = this.rarityWeights[category] ?? 0.2;
    const unc = 2 * attackProb * (1 - attackProb) * 2;
    const corr = Math.min(1, relatedAlerts / this.corrSaturation);
    const risk =
      w.attack_confidence * attackProb +
      w.anomaly * (anomalyNorm / 100) +
      w.category_severity * sev +
      w.rarity * rarity +
      w.uncertainty * unc +
      w.correlation * corr;
    return Math.min(100, Math.max(0, risk));
  }

  severityBand(risk: number): Severity {
    if (risk <= 24) return "Low";
    if (risk <= 49) return "Medium";
    if (risk <= 74) return "High";
    return "Critical";
  }

  /** Simulated pseudo-entity (deterministic hash of flow-level attributes). */
  entityOf(ev: RawEvent): string {
    const key = `${ev.proto}|${ev.service}|${ev.sttl >> 2}|${Math.min(15, ev.ct_dst_src_ltm)}|${Math.min(15, ev.ct_srv_src)}|${ev.dttl >> 2}`;
    return `ENT-${(hash32(key) % 9000 + 1000).toString()}`;
  }

  nearestCluster(x: Float64Array): number {
    const c = this.clustering;
    const bIdx = c.feature_indices;
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < c.kmeans_centroids.length; k++) {
      const cen = c.kmeans_centroids[k];
      let d = 0;
      for (let i = 0; i < bIdx.length; i++) {
        const v = (x[bIdx[i]] - c.scaler_mean[i]) / (c.scaler_std[i] || 1);
        const diff = v - cen[i];
        d += diff * diff;
      }
      if (d < bestD) { bestD = d; best = k; }
    }
    return best;
  }

  pcaProject(x: Float64Array): { x: number; y: number } {
    const c = this.clustering;
    const bIdx = c.feature_indices;
    const p1 = c.pca_components[0];
    const p2 = c.pca_components[1];
    let x1 = 0, x2 = 0;
    for (let i = 0; i < bIdx.length; i++) {
      const v = (x[bIdx[i]] - c.scaler_mean[i]) / (c.scaler_std[i] || 1);
      x1 += (v - c.pca_mean[i]) * p1[i];
      x2 += (v - c.pca_mean[i]) * p2[i];
    }
    return { x: x1, y: x2 };
  }

  /** Full scoring of one raw event. */
  score(ev: RawEvent, opts: { t?: number; eventId?: string; rowIndex?: number | null; wave?: string; includeGroundTruth?: boolean } = {}): ScoredEvent {
    return this.scoreWithFeatures(ev, opts).event;
  }

  /** Scoring that also returns the transformed feature vector (for correlation). */
  scoreWithFeatures(ev: RawEvent, opts: { t?: number; eventId?: string; rowIndex?: number | null; wave?: string; includeGroundTruth?: boolean } = {}): { event: ScoredEvent; x: Float64Array } {
    const x = this.pipeline.transform(ev);
    const { prob } = this.binaryProb(x);
    const { probs } = this.categoryProbs(x);
    const anomalyRaw = this.anomaly.scoreRaw(x);
    const anomalyNorm = this.anomaly.normalize(anomalyRaw);
    const verdict = prob >= this.threshold ? "Attack" : "Normal";
    const predClassIdx = argmax(probs);
    const category = verdict === "Attack" ? CATS[predClassIdx] : "Normal";
    const risk = this.riskScore(prob, anomalyNorm, category, 0);

    const explanation = this.explain(ev, x, prob, category, rowIndexKey(opts.rowIndex));
    const cluster = verdict === "Attack" ? this.nearestCluster(x) : null;

    const catProbs: Record<string, number> = {};
    CATS.forEach((c, i) => (catProbs[c] = Number(probs[i].toFixed(6))));

    const event: ScoredEvent = {
      eventId: opts.eventId ?? `EVT-${hash32(`${ev.id}|${ev.dur}|${ev.sbytes}|${ev.rate}`).toString(16).slice(0, 8)}`,
      rowIndex: opts.rowIndex ?? null,
      raw: ev,
      t: opts.t ?? 0,
      attackProbability: Number(prob.toFixed(6)),
      binaryVerdict: verdict,
      attackConfidence: Number((verdict === "Attack" ? prob : 1 - prob).toFixed(6)),
      category,
      categoryProbs: catProbs,
      anomalyRaw: Number(anomalyRaw.toFixed(6)),
      anomalyScore: Number(anomalyNorm.toFixed(2)),
      riskScore: Number(risk.toFixed(2)),
      severity: this.severityBand(risk),
      cluster,
      entity: this.entityOf(ev),
      explanation,
      wave: opts.wave,
      ...(opts.includeGroundTruth ? { groundTruth: ev.attack_cat } : {}),
    };
    return { event, x };
  }

  /** Local explanation: exact TreeSHAP if precomputed for this test row, else Saabas path attributions. */
  explain(ev: RawEvent, x: Float64Array, prob: number, category: string, rowKey: string | null) {
    const cached = rowKey !== null ? this.shapCache[rowKey] : undefined;
    let contribs: Float64Array;
    let baseline: number;
    let method: "exact-treeshap" | "saabas-path";
    if (cached) {
      contribs = new Float64Array(this.nFeatures);
      for (let i = 0; i < this.nFeatures; i++) contribs[i] = cached.b[i] ?? 0;
      baseline = cached.b[this.nFeatures] ?? this.shapExpected;
      method = "exact-treeshap";
    } else {
      const s = this.bin.saabas(x, this.nFeatures);
      contribs = s.contribs;
      baseline = s.baseline;
      method = "saabas-path";
    }
    const pos: FeatureAttribution[] = [];
    const neg: FeatureAttribution[] = [];
    for (let i = 0; i < this.nFeatures; i++) {
      const c = contribs[i];
      if (Math.abs(c) < 1e-4) continue;
      const name = this.featureNames[i];
      const { display } = this.pipeline.featureDisplayValue(ev, name);
      const item: FeatureAttribution = {
        feature: name,
        value: Number(display.replace(/[^\d.\-eE+]/g, "")) || 0,
        contribution: Number(c.toFixed(5)),
        direction: c > 0 ? "increases_risk" : "decreases_risk",
      };
      (c > 0 ? pos : neg).push(item);
    }
    pos.sort((a, b) => b.contribution - a.contribution);
    neg.sort((a, b) => a.contribution - b.contribution);
    const topPositive = pos.slice(0, 6);
    const topNegative = neg.slice(0, 4);
    const narrative = this.narrative(prob, category, topPositive, ev);
    return { method, baseline: Number(baseline.toFixed(4)), topPositive, topNegative, narrative };
  }

  private narrative(prob: number, category: string, top: FeatureAttribution[], ev: RawEvent): string {
    const label = (f: string) => FEATURE_LABELS[f] ?? f;
    const parts = top.slice(0, 3).map((a) => {
      const l = label(a.feature);
      return a.feature.startsWith("cat_")
        ? `${l} = ${a.feature === "cat_proto" ? ev.proto : a.feature === "cat_service" ? ev.service : ev.state}`
        : `unusual ${l}`;
    });
    const strength = prob > 0.9 ? "High" : prob > 0.7 ? "Elevated" : "Moderate";
    return `${strength} attack confidence (${(prob * 100).toFixed(1)}%) driven primarily by ${parts.join(", ")}. ` +
      `Predicted behavior pattern: ${category}. Attribution method: model feature contributions on this flow record only.`;
  }

  /** Batch score for replay/boot. */
  scoreBatch(events: { ev: RawEvent; t: number; rowIndex: number; wave?: string }[]): ScoredEvent[] {
    return events.map((e) =>
      this.score(e.ev, { t: e.t, rowIndex: e.rowIndex, wave: e.wave, eventId: `R-${e.rowIndex}`, includeGroundTruth: false })
    );
  }
}

function rowIndexKey(rowIndex: number | null | undefined): string | null {
  return rowIndex !== null && rowIndex !== undefined ? String(rowIndex) : null;
}

function argmax(a: Float64Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

export const CATEGORIES = CATS;
