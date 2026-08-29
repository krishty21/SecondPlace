/**
 * CipherMind Sentinel — shared UI tokens: severity palette, formatters,
 * chart theming and small display helpers.
 */
import type { Epistemics, IncidentStatus, Severity } from "@/lib/soc-api";

// ------------------------------------------------------------------ colors

export const SEVERITY_META: Record<
  Severity,
  { hex: string; label: string; priority: number }
> = {
  Critical: { hex: "#f43f5e", label: "Critical", priority: 4 },
  High: { hex: "#f97316", label: "High", priority: 3 },
  Medium: { hex: "#eab308", label: "Medium", priority: 2 },
  Low: { hex: "#10b981", label: "Low", priority: 1 },
};

export const ACCENT = "#06b6d4";
export const VIOLET = "#8b5cf6";

export function severityMeta(sev: string): { hex: string; label: string } {
  return SEVERITY_META[(sev as Severity) ?? "Low"] ?? { hex: "#64748b", label: sev || "—" };
}

export const STATUS_META: Record<IncidentStatus, { hex: string }> = {
  Escalating: { hex: "#f43f5e" },
  Active: { hex: "#f97316" },
  Contained: { hex: "#10b981" },
  Closed: { hex: "#8b98a9" },
};

export const EPISTEMICS_META: Record<Epistemics, { hex: string; hint: string }> = {
  Observed: { hex: "#10b981", hint: "Directly derived from scored events" },
  Inferred: { hex: "#eab308", hint: "Model / correlation interpretation" },
  Simulated: { hex: "#8b5cf6", hint: "Simulated replay metadata" },
  Prescriptive: { hex: "#06b6d4", hint: "Recommended action, not an observation" },
};

/** Fixed 8-color cluster palette (no indigo/blue). */
export const CLUSTER_PALETTE = [
  "#06b6d4",
  "#14b8a6",
  "#10b981",
  "#eab308",
  "#f97316",
  "#f43f5e",
  "#8b5cf6",
  "#ec4899",
];

export function clusterColor(cluster: number | null | undefined): string {
  if (cluster === null || cluster === undefined) return "#64748b";
  return CLUSTER_PALETTE[cluster % CLUSTER_PALETTE.length];
}

/** Map a 0-100 risk to a severity color (for chart cells / gauges). */
export function riskColor(risk: number): string {
  if (risk >= 75) return "#f43f5e";
  if (risk >= 50) return "#f97316";
  if (risk >= 25) return "#eab308";
  return "#10b981";
}

export function waveColor(wave: string | undefined): string {
  switch (wave) {
    case "campaign1-recon":
      return "#06b6d4";
    case "campaign2-exploit":
      return "#f43f5e";
    case "campaign3-dos":
      return "#eab308";
    case "noise":
      return "#64748b";
    case "background":
      return "#475569";
    case "boot":
      return "#10b981";
    default:
      return "#8b98a9";
  }
}

export function waveLabel(wave: string | undefined): string {
  if (!wave) return "—";
  return wave.replace("campaign", "C").replace("1-recon", "1 · Recon").replace("2-exploit", "2 · Exploit").replace("3-dos", "3 · DoS");
}

// --------------------------------------------------------------- formatting

/** Risk / anomaly scores: one decimal. */
export function fmtRisk(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(1);
}

/** Probabilities (0-1) as percentage. */
export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Simulated replay milliseconds as mm:ss (or h:mm:ss for long spans). */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  const total = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(value).toLocaleString("en-US");
}

/** Compact formatting for large byte / rate values. */
export function fmtBig(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  const n = typeof value === "string" ? Number(value) : value;
  if (Number.isNaN(n)) return String(value);
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  if (abs >= 1) return n.toFixed(abs >= 100 ? 0 : 1);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export function fmtSigned(value: number, digits = 2): string {
  const s = value.toFixed(digits);
  return value >= 0 ? `+${s}` : s;
}

// --------------------------------------------------------------- chart theme

export const CHART = {
  grid: "rgba(255,255,255,0.06)",
  axisLine: "rgba(255,255,255,0.10)",
  axisText: "#7d8a9b",
  cursorFill: "rgba(255,255,255,0.04)",
} as const;

export function pctTick(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}
