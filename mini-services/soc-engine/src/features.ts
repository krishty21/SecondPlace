/**
 * Feature pipeline — implements the EXACT transformations defined by
 * ml/artifacts/preprocessor/feature_config.json (fitted on training data only).
 * Mirrors ml/training/features.py.
 */
import type { FeatureConfig, RawEvent } from "./artifacts.ts";

const EPS = 1e-9;

export class FeaturePipeline {
  cfg: FeatureConfig;
  nFeatures: number;
  private rawIdx: number[] = [];
  private medians: Float64Array;
  private isLog: boolean[] = [];
  private protoMap: Record<string, number>;
  private serviceMap: Record<string, number>;
  private stateMap: Record<string, number>;
  private protoUnknown: number; private serviceUnknown: number; private stateUnknown: number;
  // derived feature start index in raw-event field space
  private static DERIVED_START = 39; // after 39 raw numeric fields

  constructor(cfg: FeatureConfig) {
    this.cfg = cfg;
    this.nFeatures = cfg.feature_names.length;
    const rawEventFields = [
      "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
      "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
      "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
      "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
      "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
      "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
      "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    ];
    this.rawIdx = rawEventFields.map((f) => cfg.raw_numeric.indexOf(f));
    this.medians = new Float64Array(cfg.raw_numeric.length);
    cfg.raw_numeric.forEach((c, i) => (this.medians[i] = cfg.medians[c] ?? 0));
    this.isLog = cfg.raw_numeric.map((c) => cfg.log_cols.includes(c));
    this.protoMap = cfg.cat_maps["proto"];
    this.serviceMap = cfg.cat_maps["service"];
    this.stateMap = cfg.cat_maps["state"];
    this.protoUnknown = cfg.unknown_code["proto"];
    this.serviceUnknown = cfg.unknown_code["service"];
    this.stateUnknown = cfg.unknown_code["state"];
  }

  /** Numeric sanitation identical to python: NaN/Inf -> train median. */
  private num(v: number, i: number): number {
    if (!Number.isFinite(v) || Number.isNaN(v)) return this.medians[i];
    return v;
  }

  transform(ev: RawEvent): Float64Array {
    const out = new Float64Array(this.nFeatures);
    const rawVals = [
      ev.dur, ev.spkts, ev.dpkts, ev.sbytes, ev.dbytes, ev.rate, ev.sttl, ev.dttl,
      ev.sload, ev.dload, ev.sloss, ev.dloss, ev.sinpkt, ev.dinpkt, ev.sjit, ev.djit,
      ev.swin, ev.stcpb, ev.dtcpb, ev.dwin, ev.tcprtt, ev.synack, ev.ackdat,
      ev.smean, ev.dmean, ev.trans_depth, ev.response_body_len, ev.ct_srv_src,
      ev.ct_state_ttl, ev.ct_dst_ltm, ev.ct_src_dport_ltm, ev.ct_dst_sport_ltm,
      ev.ct_dst_src_ltm, ev.is_ftp_login, ev.ct_ftp_cmd, ev.ct_flw_http_mthd,
      ev.ct_src_ltm, ev.ct_srv_dst, ev.is_sm_ips_ports,
    ];
    // 1) raw numeric (sanitize + impute + log1p)
    for (let i = 0; i < rawVals.length; i++) {
      let v = this.num(rawVals[i], i);
      if (this.isLog[i]) v = Math.log1p(Math.max(0, v));
      out[i] = v;
    }
    const base = FeaturePipeline.DERIVED_START;
    // 2) derived (sanitized: non-finite -> 0 like python)
    const sbytes = this.num(ev.sbytes, 3), dbytes = this.num(ev.dbytes, 4);
    const spkts = this.num(ev.spkts, 1), dpkts = this.num(ev.dpkts, 2);
    const sload = this.num(ev.sload, 8), dload = this.num(ev.dload, 9);
    out[base + 0] = sbytes / (dbytes + EPS);
    out[base + 1] = spkts / (dpkts + EPS);
    out[base + 2] = sbytes + dbytes;
    out[base + 3] = spkts + dpkts;
    out[base + 4] = this.num(ev.sloss, 10) + this.num(ev.dloss, 11);
    out[base + 5] = (sbytes + dbytes) / (spkts + dpkts + EPS);
    out[base + 6] = sload / (dload + EPS);
    out[base + 7] = this.num(ev.smean, 23) / (this.num(ev.dmean, 24) + EPS);
    out[base + 8] = this.num(ev.synack, 21) / (this.num(ev.ackdat, 22) + EPS);
    out[base + 9] = this.num(ev.sjit, 14) / (this.num(ev.djit, 15) + EPS);
    out[base + 10] = this.num(ev.sinpkt, 12) / (this.num(ev.dinpkt, 13) + EPS);
    out[base + 11] = (sbytes - dbytes) / (sbytes + dbytes + EPS);
    // 3) categorical ordinal codes with __unknown__ fallback
    out[base + 12] = this.protoMap[ev.proto] ?? this.protoUnknown;
    out[base + 13] = this.serviceMap[ev.service] ?? this.serviceUnknown;
    out[base + 14] = this.stateMap[ev.state] ?? this.stateUnknown;
    // sanitize derived (python fills NaN with 0)
    for (let i = base; i < base + 12; i++) {
      if (!Number.isFinite(out[i])) out[i] = 0;
    }
    return out;
  }

  /** Human-readable value of a feature for explanation UIs. */
  featureDisplayValue(ev: RawEvent, feature: string): { display: string; raw: number } {
    const strip = (f: string) => (f.startsWith("log1p_") ? f.slice(6) : f);
    if (feature.startsWith("cat_")) {
      const c = feature.slice(4);
      const v = c === "proto" ? ev.proto : c === "service" ? ev.service : ev.state;
      return { display: `${v} (code ${this.cfg.cat_maps[c]?.[v] ?? this.cfg.unknown_code[c]})`, raw: (this.cfg.cat_maps[c]?.[v] ?? this.cfg.unknown_code[c]) };
    }
    const rawName = strip(feature);
    const isDerived = this.cfg.derived.some((d) => d.name === rawName);
    if (isDerived) {
      const x = this.transform(ev);
      const i = this.cfg.feature_names.indexOf(rawName);
      return { display: Number(x[i]).toFixed(3), raw: x[i] };
    }
    const v = (ev as unknown as Record<string, number>)[rawName];
    if (v === undefined) return { display: "n/a", raw: 0 };
    return { display: Number.isInteger(v) ? String(v) : Number(v).toFixed(4), raw: v };
  }
}

/** Pretty names for model features used in narratives. */
export const FEATURE_LABELS: Record<string, string> = {
  dur: "connection duration",
  spkts: "source packet count", dpkts: "destination packet count",
  sbytes: "source-to-destination bytes", dbytes: "destination-to-source bytes",
  rate: "traffic rate", sttl: "source TTL", dttl: "destination TTL",
  sload: "source load", dload: "destination load",
  sloss: "source retransmissions", dloss: "destination retransmissions",
  sinpkt: "source inter-packet time", dinpkt: "destination inter-packet time",
  sjit: "source jitter", djit: "destination jitter",
  swin: "source TCP window", dwin: "destination TCP window",
  stcpb: "source TCP base sequence", dtcpb: "destination TCP base sequence",
  tcprtt: "TCP round-trip time", synack: "SYN/ACK timing", ackdat: "ACK timing",
  smean: "mean source packet size", dmean: "mean destination packet size",
  trans_depth: "transaction depth", response_body_len: "response body length",
  ct_srv_src: "service/source connection count", ct_state_ttl: "state/TTL pattern count",
  ct_dst_ltm: "destination activity (last 100 flows)", ct_src_dport_ltm: "source/dest-port activity",
  ct_dst_sport_ltm: "destination/src-port activity", ct_dst_src_ltm: "destination/source relationship count",
  is_ftp_login: "FTP login indicator", ct_ftp_cmd: "FTP command count",
  ct_flw_http_mthd: "HTTP method/flow count", ct_src_ltm: "source activity (last 100 flows)",
  ct_srv_dst: "service/destination relationship count", is_sm_ips_ports: "same-IP/port indicator",
  byte_ratio: "byte ratio (src/dst)", packet_ratio: "packet ratio (src/dst)",
  total_bytes: "total bytes transferred", total_packets: "total packets",
  total_loss: "total retransmissions", payload_per_packet: "payload per packet",
  load_ratio: "load ratio (src/dst)", size_ratio: "packet size ratio (src/dst)",
  rtt_ratio: "RTT ratio (SYN-ACK/ACK)", jitter_ratio: "jitter ratio (src/dst)",
  interpkt_ratio: "inter-packet timing ratio", flow_asymmetry: "flow direction asymmetry",
  cat_proto: "protocol", cat_service: "network service", cat_state: "connection state",
};
