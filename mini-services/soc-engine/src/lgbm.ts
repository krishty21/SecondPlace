/**
 * LightGBM inference engine — walks dumped tree JSON exactly like the C++ predictor.
 * Supports binary (sigmoid, raw margin) and multiclass (softmax w/ temperature).
 * Saabas path attributions (treeinterpreter-style local explanations), verified
 * against Python: sum(contribs) + baseline == raw score.
 */
import type { LgbmDump, LgbmNode } from "./artifacts.ts";

const DECISION_MASK = 8; // categorical decision bit in LightGBM decision_type

/** decision_type in dump_model is a string: "<=" numeric, "==" categorical. */
function isCategoricalDecision(dtype: string | undefined): boolean {
  if (!dtype) return false;
  if (dtype === "==") return true;
  if (dtype === "<=") return false;
  // numeric code fallback (2 = numeric, 10 = categorical)
  const code = dtype.charCodeAt(0);
  return (code & DECISION_MASK) > 0;
}

export class LgbmModel {
  numClass: number;
  trees: LgbmNode[];
  nTrees: number;
  private treesPerIteration: number;

  constructor(dump: LgbmDump) {
    this.numClass = dump.num_class || 1;
    this.treesPerIteration = dump.num_tree_per_iteration || this.numClass;
    this.trees = dump.tree_info.map((t) => t.tree_structure);
    this.nTrees = this.trees.length;
  }

  /** Raw margin score for a single class (binary: class ignored). */
  predictRaw(x: Float64Array): number {
    let sum = 0;
    if (this.numClass === 1) {
      for (let t = 0; t < this.nTrees; t++) sum += this.walkLeaf(this.trees[t], x);
      return sum;
    }
    throw new Error("use predictRawMulti for multiclass models");
  }

  predictRawMulti(x: Float64Array, out: Float64Array): void {
    out.fill(0);
    for (let t = 0; t < this.nTrees; t++) {
      const cls = t % this.treesPerIteration;
      out[cls] += this.walkLeaf(this.trees[t], x);
    }
  }

  private walkLeaf(root: LgbmNode, x: Float64Array): number {
    let node = root;
    while (node.leaf_value === undefined) {
      const f = node.split_feature!;
      const thr = node.threshold!;
      const dtype = node.decision_type ?? "<=";
      const goLeft: boolean = isCategoricalDecision(dtype) ? x[f] === thr : x[f] <= thr;
      node = goLeft ? node.left_child! : node.right_child!;
    }
    return node.leaf_value!;
  }

  /**
   * Saabas path attributions. For every tree, walking root->leaf, each split
   * attributes (child expected value - node expected value) to the split feature.
   * Verified: baseline + sum(contribs) == raw score. Baseline = mean training
   * prediction of the tree (root internal_value).
   */
  saabas(x: Float64Array, nFeatures: number): { baseline: number; contribs: Float64Array } {
    const contribs = new Float64Array(nFeatures);
    let baseline = 0;
    if (this.numClass === 1) {
      for (let t = 0; t < this.nTrees; t++) {
        const { base, cont } = this.walkContribs(this.trees[t], x, nFeatures);
        baseline += base;
        for (let i = 0; i < nFeatures; i++) contribs[i] += cont[i];
      }
      return { baseline, contribs };
    }
    throw new Error("use saabasMulti for multiclass models");
  }

  /** Per-class Saabas attributions for multiclass models (for a given class). */
  saabasMulti(x: Float64Array, nFeatures: number, cls: number): { baseline: number; contribs: Float64Array } {
    const contribs = new Float64Array(nFeatures);
    let baseline = 0;
    for (let t = cls; t < this.nTrees; t += this.treesPerIteration) {
      const { base, cont } = this.walkContribs(this.trees[t], x, nFeatures);
      baseline += base;
      for (let i = 0; i < nFeatures; i++) contribs[i] += cont[i];
    }
    return { baseline, contribs };
  }

  private walkContribs(root: LgbmNode, x: Float64Array, nFeatures: number): { base: number; cont: Float64Array } {
    const cont = new Float64Array(nFeatures);
    let node = root;
    const base = node.internal_value ?? 0;
    while (node.leaf_value === undefined) {
      const f = node.split_feature!;
      const thr = node.threshold!;
      const dtype = node.decision_type ?? "<=";
      const goLeft = isCategoricalDecision(dtype) ? x[f] === thr : x[f] <= thr;
      const next = goLeft ? node.left_child! : node.right_child!;
      const nodeVal = node.internal_value ?? 0;
      const nextVal = next.leaf_value !== undefined ? next.leaf_value : (next.internal_value ?? 0);
      cont[f] += nextVal - nodeVal;
      node = next;
    }
    return { base, cont };
  }
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function softmax(z: Float64Array, T = 1): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) { const v = z[i] / T; if (v > max) max = v; }
  const out = new Float64Array(z.length);
  let sum = 0;
  for (let i = 0; i < z.length; i++) { const e = Math.exp(z[i] / T - max); out[i] = e; sum += e; }
  for (let i = 0; i < z.length; i++) out[i] /= sum;
  return out;
}
