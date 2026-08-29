"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { AttributionList, maxContribution } from "@/components/soc/contribution-bar";
import {
  CategoryChip,
  ChartSkeleton,
  SeverityBadge,
  SimulatedTag,
  WaveTag,
} from "@/components/soc/primitives";
import { RiskGauge } from "@/components/soc/risk-gauge";
import type { ScoredEvent } from "@/lib/soc-api";
import { fmtBig, fmtPct, fmtRisk, fmtMs } from "@/lib/soc-ui";
import { cn } from "@/lib/utils";

/** Compact single-line event row used in dense feeds (replay, tables). */
export function EventRow({
  event,
  onClick,
  dimmed = false,
}: {
  event: ScoredEvent;
  onClick?: (event: ScoredEvent) => void;
  dimmed?: boolean;
}) {
  const isAlert = event.binaryVerdict === "Attack";
  const interactive = Boolean(onClick);
  const dotColor = isAlert ? severityHex(event.severity) : "#475569";
  return (
    <button
      type="button"
      disabled={!interactive}
      onClick={onClick ? () => onClick(event) : undefined}
      className={cn(
        "flex w-full items-center gap-2 px-2 py-1 text-left font-mono text-[11px] tabular-nums transition-colors",
        interactive && "hover:bg-accent/40",
        dimmed && "opacity-45"
      )}
      aria-label={`Event ${event.eventId}, ${event.category}, risk ${event.riskScore}`}
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: dotColor }} aria-hidden />
      <span className="w-16 shrink-0 truncate text-muted-foreground">{event.eventId}</span>
      <span className={cn("w-24 shrink-0 truncate", isAlert ? "text-foreground/90" : "text-muted-foreground")}>
        {event.category}
      </span>
      <span className="w-12 shrink-0 text-right" style={{ color: isAlert ? undefined : "#475569" }}>
        {isAlert ? fmtRisk(event.riskScore) : "·"}
      </span>
      <span className="w-12 shrink-0 text-right text-muted-foreground">
        {isAlert ? fmtRisk(event.anomalyScore) : "·"}
      </span>
      <span className="w-12 shrink-0 text-right text-muted-foreground">
        {isAlert ? fmtPct(event.attackConfidence, 0) : "·"}
      </span>
      <span className="w-20 shrink-0 truncate text-muted-foreground">{event.entity}</span>
      <WaveTag wave={event.wave} className="hidden shrink-0 sm:inline-flex" />
    </button>
  );
}

function severityHex(sev: string): string {
  switch (sev) {
    case "Critical":
      return "#f43f5e";
    case "High":
      return "#f97316";
    case "Medium":
      return "#eab308";
    default:
      return "#10b981";
  }
}

const FLOW_FIELDS: [string, string][] = [
  ["dur", "duration s"],
  ["proto", "protocol"],
  ["service", "service"],
  ["state", "state"],
  ["spkts", "src pkts"],
  ["dpkts", "dst pkts"],
  ["sbytes", "src bytes"],
  ["dbytes", "dst bytes"],
  ["rate", "pkts/s"],
  ["sttl", "src ttl"],
  ["dttl", "dst ttl"],
  ["sload", "src load"],
  ["dload", "dst load"],
  ["sloss", "src loss"],
  ["dloss", "dst loss"],
  ["sinpkt", "src iat"],
  ["dinpkt", "dst iat"],
  ["sjit", "src jitter"],
  ["djit", "dst jitter"],
  ["tcprtt", "tcp rtt"],
  ["synack", "syn→ack"],
  ["ackdat", "ack→data"],
  ["smean", "src mean pkt"],
  ["dmean", "dst mean pkt"],
  ["trans_depth", "http depth"],
  ["response_body_len", "resp body"],
  ["ct_srv_src", "srv-src conns"],
  ["ct_state_ttl", "state-ttl conns"],
  ["ct_dst_ltm", "dst-ltm conns"],
  ["ct_src_dport_ltm", "src-dport conns"],
  ["ct_dst_sport_ltm", "dst-sport conns"],
  ["ct_dst_src_ltm", "dst-src conns"],
  ["ct_src_ltm", "src-ltm conns"],
  ["ct_srv_dst", "srv-dst conns"],
  ["is_sm_ips_ports", "same ip/port"],
];

export function EventDetailDialog({
  event,
  open,
  onOpenChange,
}: {
  event: ScoredEvent | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  if (!event) return null;
  const expl = event.explanation;
  const topPos = expl?.topPositive ?? [];
  const topNeg = expl?.topNegative ?? [];
  const scale = maxContribution([...topPos, ...topNeg]);

  const probs = Object.entries(event.categoryProbs ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] w-[95vw] max-w-4xl overflow-y-auto soc-scroll rounded-xl border-border/60 bg-card/95 p-0">
        <DialogHeader className="border-b border-border/50 px-5 py-4">
          <div className="flex flex-wrap items-center gap-2">
            <DialogTitle className="font-mono text-sm text-primary">{event.eventId}</DialogTitle>
            <SeverityBadge severity={event.severity} />
            <CategoryChip category={event.category} />
            <Badge variant="outline" className="border-border/60 font-mono text-[10px] text-muted-foreground">
              {event.binaryVerdict}
            </Badge>
            <WaveTag wave={event.wave} />
            <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
              t+{fmtMs(event.t)} <SimulatedTag />
            </span>
          </div>
          <DialogDescription className="sr-only">
            Flow detail and model explanation for event {event.eventId}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 px-5 py-4">
          {/* verdict block */}
          <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
            <RiskGauge value={event.riskScore} size={110} label="risk" />
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <VerdictCell label="Attack prob." value={fmtPct(event.attackProbability)} accent="#f43f5e" />
              <VerdictCell label="Confidence" value={fmtPct(event.attackConfidence, 0)} accent="#06b6d4" />
              <VerdictCell label="Anomaly" value={fmtRisk(event.anomalyScore)} accent="#eab308" />
              <VerdictCell label="Cluster" value={event.cluster === null ? "—" : `C${event.cluster}`} accent="#8b5cf6" />
              <VerdictCell label="Entity" value={event.entity} accent="#8b5cf6" small />
              <VerdictCell
                label="Ground truth"
                value={event.groundTruth ?? event.raw?.attack_cat ?? "—"}
                accent="#8b98a9"
                small
              />
            </div>
          </div>

          {/* explanation */}
          {expl ? (
            <section aria-label="model explanation" className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                  Why the model decided this
                </h3>
                <Badge
                  variant="outline"
                  className="border-primary/30 bg-primary/10 font-mono text-[9px] uppercase tracking-wider text-primary"
                >
                  {expl.method}
                </Badge>
                <span className="font-mono text-[10px] text-muted-foreground">
                  baseline {expl.baseline.toFixed(4)}
                </span>
              </div>
              <p className="rounded-lg border border-border/50 bg-background/40 px-3 py-2 text-xs leading-relaxed text-foreground/85">
                {expl.narrative}
              </p>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-border/50 bg-background/30 p-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-rose-400">
                    Increases risk
                  </p>
                  <AttributionList items={topPos} maxAbs={scale} emptyText="—" />
                </div>
                <div className="rounded-lg border border-border/50 bg-background/30 p-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-emerald-400">
                    Decreases risk
                  </p>
                  <AttributionList items={topNeg} maxAbs={scale} emptyText="—" />
                </div>
              </div>
            </section>
          ) : (
            <ChartSkeleton />
          )}

          {/* class probabilities */}
          {probs.length > 0 && (
            <section aria-label="predicted class probabilities" className="space-y-2">
              <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Predicted class probabilities
              </h3>
              <div className="space-y-1.5">
                {probs.map(([cat, p]) => (
                  <div key={cat} className="flex items-center gap-3">
                    <span className="w-28 shrink-0 truncate font-mono text-[11px] text-foreground/80">{cat}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                      <div
                        className="h-full rounded-full bg-primary/70"
                        style={{ width: `${Math.max(0.5, Math.min(100, p * 100))}%` }}
                      />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground">
                      {fmtPct(p, 2)}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* raw flow */}
          <section aria-label="raw flow record" className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
              Raw UNSW-NB15 flow fields
            </h3>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-border/50 bg-background/30 p-3 sm:grid-cols-3 lg:grid-cols-4">
              {FLOW_FIELDS.map(([key, label]) => (
                <div key={key} className="flex items-baseline justify-between gap-2 border-b border-border/30 pb-0.5">
                  <span className="text-[10px] text-muted-foreground">{label}</span>
                  <span className="font-mono text-[11px] tabular-nums text-foreground/85">
                    {typeof event.raw?.[key] === "number" ? fmtBig(event.raw[key] as number) : String(event.raw?.[key] ?? "—")}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-muted-foreground">
              Record #{event.rowIndex ?? "—"} from the official UNSW-NB15 test set · 45 CSV fields (engineered features
              are derived from these).
            </p>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function VerdictCell({
  label,
  value,
  accent,
  small = false,
}: {
  label: string;
  value: string;
  accent: string;
  small?: boolean;
}) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/40 px-2.5 py-1.5">
      <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p
        className={cn("truncate font-mono font-bold tabular-nums", small ? "text-xs" : "text-sm")}
        style={{ color: accent }}
        title={value}
      >
        {value}
      </p>
    </div>
  );
}
