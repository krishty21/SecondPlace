/** Artifact loaders — reads the JSON model package produced by ml/scripts/train.py. */
import * as fs from "fs";
import * as path from "path";

const ROOT = path.resolve(import.meta.dir, "..", "..", "..");
export const ARTIFACTS = path.join(ROOT, "ml", "artifacts");
export const TEST_CSV = path.join(ROOT, "dataset", "Training and Testing Sets", "UNSW_NB15_testing-set.csv");
export const TRAIN_CSV = path.join(ROOT, "dataset", "Training and Testing Sets", "UNSW_NB15_training-set.csv");

function readJson<T>(rel: string): T {
  const p = path.join(ARTIFACTS, rel);
  return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
}

// ---------------------------------------------------------------- types
export interface LgbmNode {
  split_index?: number;
  split_feature?: number;
  threshold?: number;
  decision_type?: string;
  internal_value?: number;
  internal_count?: number;
  left_child?: LgbmNode;
  right_child?: LgbmNode;
  leaf_value?: number;
  value?: number;
}

export interface LgbmDump {
  max_feature_idx?: number;
  num_class: number;
  num_tree_per_iteration: number;
  tree_info: { tree_index: number; num_leaves: number; tree_structure: LgbmNode }[];
  // added by our loader:
  trees?: LgbmNode[];
}

export interface FeatureConfig {
  raw_numeric: string[];
  categorical: string[];
  medians: Record<string, number>;
  log_cols: string[];
  cat_maps: Record<string, Record<string, number>>;
  unknown_code: Record<string, number>;
  derived: { name: string; formula: string }[];
  eps: number;
  feature_names: string[];
}

export interface IForestArt {
  n_estimators: number;
  subsample_size: number;
  c_n: number;
  trees: number[][][]; // tree -> nodes -> [f, thr, l, r, n]
  norm_anchors: Record<string, number>;
  norm_anchor_targets: Record<string, number>;
}

export interface ClusteringArt {
  features: string[];
  feature_indices: number[];
  scaler_mean: number[];
  scaler_std: number[];
  kmeans_centroids: number[][];
  pca_components: number[][];
  pca_explained_variance: number[];
  pca_mean: number[];
  profiles: {
    cluster: number; size: number; dominant_category: string;
    category_distribution: Record<string, number>;
    top_features: { feature: string; z_score: number }[];
  }[];
  sample_points: { x: number; y: number; cluster: number; category: string }[];
}

export interface Registry {
  name: string; version: string; trained_at: string; seed: number;
  dataset: { train_file: string; test_file: string; train_rows: number; test_rows: number };
  models: Record<string, { algorithm: string; artifact: string; [k: string]: unknown }>;
  threshold: number;
  platt: { a: number; b: number };
  temperature: number;
  risk_config: RiskConfig;
  feature_count: number;
  class_mapping: Record<string, number>;
  software: Record<string, string>;
}

export interface RiskConfig {
  weights: Record<string, number>;
  category_severity: Record<string, number>;
  severity_bands: { low: number; medium: number; high: number };
  correlation_alert_saturation: number;
  rarity_weights?: Record<string, number>;
}

// ---------------------------------------------------------------- loader
let _cache: {
  binary: LgbmDump; multiclass: LgbmDump; iforest: IForestArt;
  features: FeatureConfig; clustering: ClusteringArt; registry: Registry;
  demoSequence: { events: { i: number; t: number; wave: string }[]; default_speed_events_per_sec: number; total_events: number; duration_ms: number; label: string; simulation_note: string };
  bootSample: { indices: number[] };
  shapCache: Record<string, { b: number[]; mc?: { class: string; top: [string, number][] } }>;
  shapGlobal: { expected_value: number; method: string; features: { feature: string; mean_abs_shap: number }[] };
  mcGain: { features: { feature: string; gain: number }[] };
  calibration: { platt: { a: number; b: number }; temperature: number; chosen_threshold: number; threshold_curve: { threshold: number; precision: number; recall: number; f1: number }[]; oof_reliability: { bin_low: number; bin_high: number; mean_predicted: number; fraction_positive: number; count: number }[]; oof_brier_raw: number; oof_brier_platt: number };
  testEval: Record<string, unknown>;
  modelComparison: Record<string, unknown>;
  operational: Record<string, unknown>;
  evalSummary: Record<string, unknown>;
  datasetProfile: Record<string, unknown>;
} | null = null;

export function loadArtifacts() {
  if (_cache) return _cache;
  const t0 = Date.now();
  const binary = readJson<LgbmDump>("models/binary_lightgbm.json");
  const multiclass = readJson<LgbmDump>("models/multiclass_lightgbm.json");
  _cache = {
    binary,
    multiclass,
    iforest: readJson<IForestArt>("models/isolation_forest.json"),
    features: readJson<FeatureConfig>("preprocessor/feature_config.json"),
    clustering: readJson<ClusteringArt>("preprocessor/clustering.json"),
    registry: readJson<Registry>("metadata/model_registry.json"),
    demoSequence: readJson("replay/demo_sequence.json"),
    bootSample: readJson("replay/boot_sample.json"),
    shapCache: readJson("explainability/shap_cache.json"),
    shapGlobal: readJson("metrics/shap_global.json"),
    mcGain: readJson("metrics/multiclass_gain.json"),
    calibration: readJson("metrics/calibration.json"),
    testEval: readJson("metrics/test_evaluation.json"),
    modelComparison: readJson("metrics/model_comparison.json"),
    operational: readJson("metrics/operational.json"),
    evalSummary: readJson("reports/eval_summary.json"),
    datasetProfile: readJson("metrics/dataset_profile.json"),
  };
  console.log(`[soc-engine] artifacts loaded in ${Date.now() - t0}ms (binary ${binary.tree_info.length} trees, multiclass ${multiclass.tree_info.length} trees, shap cache ${Object.keys(_cache.shapCache).length} events)`);
  return _cache;
}

/** Parse the official test CSV into typed rows (index-aligned). */
export function loadTestCsv(): { header: string[]; rows: string[][] } {
  const t0 = Date.now();
  const text = fs.readFileSync(TEST_CSV, "utf-8");
  const lines = text.split("\n");
  const header = lines[0].replace(/^\uFEFF/, "").replace(/\r$/, "").split(",").map((h) => h.trim());
  const rows: string[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].replace(/\r$/, "");
    if (!line) continue;
    const cells = line.split(",").map((c) => c.trim());
    rows.push(cells);
  }
  console.log(`[soc-engine] test CSV parsed: ${rows.length} rows in ${Date.now() - t0}ms`);
  return { header, rows };
}
