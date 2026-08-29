/**
 * Isolation Forest scorer — implements the original paper's anomaly score
 * s(x) = 2^(-E(h(x)) / c(n)) over serialized sklearn trees, identical to the
 * Python reference (ml/scripts/train.py::iforest_score).
 */
import type { IForestArt } from "./artifacts.ts";

function c(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  const EULER_GAMMA = 0.5772156649015329;
  return 2 * (Math.log(n - 1) + EULER_GAMMA) - (2 * (n - 1)) / n;
}

export class AnomalyDetector {
  art: IForestArt;
  private trees: number[][][];

  constructor(art: IForestArt) {
    this.art = art;
    this.trees = art.trees;
  }

  /** Raw anomaly score in (0, 1]; higher = more anomalous. */
  scoreRaw(x: Float64Array): number {
    let depthSum = 0;
    for (let t = 0; t < this.trees.length; t++) {
      const nodes = this.trees[t];
      let i = 0;
      let depth = 0;
      for (;;) {
        const node = nodes[i];
        if (node[0] === -1) { // leaf
          depthSum += depth + c(node[4] - 1);
          break;
        }
        i = x[node[0]] <= node[1] ? node[2] : node[3];
        depth++;
      }
    }
    return Math.pow(2, -depthSum / (this.trees.length * this.art.c_n));
  }

  /** Deterministic 0-100 normalization via training-normal percentile anchors. */
  normalize(raw: number): number {
    const a = this.art.norm_anchors;
    const t = this.art.norm_anchor_targets;
    const xs = [0, a.p50, a.p90, a.p99, a.p999, 1];
    const ys = [0, t.p50, t.p90, t.p99, t.p999, 100];
    if (raw <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) {
      if (raw <= xs[i]) {
        const f = (raw - xs[i - 1]) / (xs[i] - xs[i - 1] || 1);
        return ys[i - 1] + f * (ys[i] - ys[i - 1]);
      }
    }
    return 100;
  }
}
