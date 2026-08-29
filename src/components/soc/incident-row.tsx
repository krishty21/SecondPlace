"use client";

import { ChevronRight } from "lucide-react";
import type { Incident } from "@/lib/soc-api";
import { fmtMs, fmtPct, fmtRisk, riskColor } from "@/lib/soc-ui";
import { CategoryChip, SeverityBadge, SimulatedTag, StatusBadge } from "@/components/soc/primitives";
import { Sparkline } from "@/components/soc/risk-gauge";
import { cn } from "@/lib/utils";

export function IncidentRow({
  incident,
  onSelect,
  active = false,
  dense = false,
}: {
  incident: Incident;
  onSelect: (incident: Incident) => void;
  active?: boolean;
  dense?: boolean;
}) {
  const topCats = Object.entries(incident.categoryMix ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2);
  const contributorPreview = (incident.topContributors ?? [])
    .slice(0, 3)
    .map((c) => c.feature)
    .join(" · ");
  const trajectory = (incident.riskTrajectory ?? []).map((p) => p.risk);

  return (
    <button
      type="button"
      onClick={() => onSelect(incident)}
      aria-label={`Open incident ${incident.incidentId}: ${incident.title}`}
      className={cn(
        "group w-full rounded-lg border border-transparent px-3 py-2.5 text-left transition-colors duration-150",
        "hover:border-primary/30 hover:bg-accent/40",
        active && "border-primary/40 bg-accent/50",
        dense ? "py-2" : ""
      )}
    >
      <div className="flex items-start gap-3">
        {/* risk score block */}
        <div className="flex w-14 shrink-0 flex-col items-center gap-1 pt-0.5">
          <span
            className="rounded-md border px-1.5 py-0.5 font-mono text-[13px] font-bold tabular-nums"
            style={{
              color: riskColor(incident.riskScore),
              backgroundColor: `${riskColor(incident.riskScore)}12`,
              borderColor: `${riskColor(incident.riskScore)}35`,
            }}
          >
            {fmtRisk(incident.riskScore)}
          </span>
          {trajectory.length > 1 && (
            <Sparkline points={trajectory} width={52} height={16} />
          )}
        </div>

        {/* main info */}
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="font-mono text-xs font-semibold text-primary">{incident.incidentId}</span>
            <span className="truncate text-sm font-medium text-foreground/95">{incident.title}</span>
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <SeverityBadge severity={incident.severity} />
            <StatusBadge status={incident.status} />
            {topCats.map(([cat, n]) => (
              <CategoryChip key={cat} category={cat} count={n} />
            ))}
          </div>
          {contributorPreview && (
            <p className="truncate font-mono text-[10px] text-muted-foreground">
              evidence: {contributorPreview}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
            <span>
              <span className="font-mono font-semibold text-foreground/80">{incident.alertCount}</span> alerts
            </span>
            <span>
              conf <span className="font-mono font-semibold text-foreground/80">{fmtPct(incident.meanConfidence, 0)}</span>
            </span>
            <span>
              anomaly{" "}
              <span className="font-mono font-semibold text-foreground/80">{fmtRisk(incident.meanAnomaly)}</span>
            </span>
            <span className="inline-flex items-center gap-1">
              last seen <span className="font-mono">{fmtMs(incident.lastSeen)}</span>
              <SimulatedTag />
            </span>
          </div>
        </div>

        <ChevronRight
          className="mt-1 h-4 w-4 shrink-0 text-muted-foreground/50 transition-all duration-150 group-hover:translate-x-0.5 group-hover:text-primary"
          aria-hidden
        />
      </div>
    </button>
  );
}
