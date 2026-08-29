"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  BellRing,
  Crosshair,
  FilterX,
  Flame,
  Gauge,
  Info,
  ListFilter,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  Cell,
  ComposedChart,
  Bar,
  BarChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { socApi, type ScoredEvent } from "@/lib/soc-api";
import { CHART, fmtInt, fmtMs, fmtPct, fmtRisk, riskColor } from "@/lib/soc-ui";
import {
  ChartTooltip,
  ErrorState,
  SectionHeader,
  SeverityBadge,
  SimulatedTag,
  WaveTag,
} from "@/components/soc/primitives";
import { KpiCard } from "@/components/soc/kpi-card";
import { IncidentRow } from "@/components/soc/incident-row";
import { EventDetailDialog } from "@/components/soc/event-detail";
import { Skeleton } from "@/components/ui/skeleton";
import type { Incident } from "@/lib/soc-api";

export function CommandCenter({ onOpenIncident }: { onOpenIncident: (incident: Incident) => void }) {
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ["dashboard"],
    queryFn: ({ signal }) => socApi.dashboard(signal),
  });

  const [detailEvent, setDetailEvent] = useState<ScoredEvent | null>(null);

  const timeline = useMemo(
    () =>
      (data?.timeline ?? []).map((p) => ({
        ...p,
        label: fmtMs(p.t),
      })),
    [data?.timeline]
  );

  if (isPending) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </div>
        <div className="grid gap-4 lg:grid-cols-5">
          <Skeleton className="h-80 rounded-xl lg:col-span-3" />
          <Skeleton className="h-80 rounded-xl lg:col-span-2" />
        </div>
        <Skeleton className="h-96 rounded-xl" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <ErrorState
        title="Command Center unavailable"
        description="Could not load the boot-state dashboard from the inference engine (port 3010). If the engine is still training/loading artifacts, retry in a moment."
        onRetry={() => void refetch()}
        className="min-h-64"
      />
    );
  }

  const { kpis } = data;
  const topIncidents = [...data.topIncidents].sort((a, b) => b.riskScore - a.riskScore);

  return (
    <div className="space-y-6">
      {/* KPI row */}
      <section aria-label="key performance indicators" className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <KpiCard
          icon={<BellRing className="h-3.5 w-3.5" aria-hidden />}
          label="Total Alerts"
          value={kpis.totalAlerts}
          sublabel={`of ${fmtInt(kpis.totalEvents)} scored events`}
        />
        <KpiCard
          icon={<ShieldAlert className="h-3.5 w-3.5" aria-hidden />}
          label="Active Incidents"
          value={kpis.activeIncidents}
          sublabel={`${kpis.criticalIncidents} critical · correlated cases`}
          accent="#f97316"
        />
        <KpiCard
          icon={<Flame className="h-3.5 w-3.5" aria-hidden />}
          label="Critical Threats"
          value={kpis.criticalIncidents}
          sublabel="critical-severity incidents"
          accent="#f43f5e"
        />
        <KpiCard
          icon={<Crosshair className="h-3.5 w-3.5" aria-hidden />}
          label="Detection Rate"
          value={kpis.detectionRate * 100}
          format={(v) => `${v.toFixed(1)}%`}
          sublabel={`alerts / events · median risk ${fmtRisk(kpis.medianResponseRisk)}`}
          accent="#14b8a6"
        />
        <KpiCard
          icon={<FilterX className="h-3.5 w-3.5" aria-hidden />}
          label="False Positive Ind."
          value={kpis.falsePositiveIndicator * 100}
          format={(v) => `${v.toFixed(2)}%`}
          sublabel="alerts flagged Attack, ground-truth Normal"
          accent="#eab308"
        />
        <KpiCard
          icon={<TrendingUp className="h-3.5 w-3.5" aria-hidden />}
          label="High-Risk Trend"
          value={kpis.highRiskTrend}
          format={(v) => v.toFixed(1)}
          trend={kpis.highRiskTrend}
          sublabel="mean-risk slope · late vs early window"
          accent={kpis.highRiskTrend >= 0 ? "#f43f5e" : "#10b981"}
        />
      </section>

      {/* Row 2: timeline + category breakdown */}
      <section className="grid gap-4 lg:grid-cols-5" aria-label="traffic and attack charts">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <SectionHeader
            title="Threat Timeline"
            hint="boot-state window · simulated replay clock"
            icon={<Activity className="h-4 w-4" aria-hidden />}
            action={
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                <LegendDot color="#06b6d4" label="events" />
                <LegendDot color="#f43f5e" label="alerts" />
                <LegendDot color="#eab308" label="mean risk (right axis)" />
              </div>
            }
          />
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timeline} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="gEvents" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.28} />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gAlerts" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis
                  dataKey="label"
                  tickLine={false}
                  axisLine={{ stroke: CHART.axisLine }}
                  tickMargin={8}
                  minTickGap={48}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  yAxisId="counts"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                  width={48}
                />
                <YAxis
                  yAxisId="risk"
                  orientation="right"
                  domain={[0, 100]}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                  width={34}
                />
                <Tooltip
                  cursor={{ stroke: "rgba(255,255,255,0.15)" }}
                  content={({ active, payload, label }) => (
                    <ChartTooltip
                      active={active}
                      label={`t+${label}`}
                      entries={(payload ?? []).map((p) => ({
                        name:
                          p.dataKey === "meanRisk"
                            ? "mean risk"
                            : p.dataKey === "alerts"
                              ? "alerts"
                              : "events",
                        value: p.dataKey === "meanRisk" ? fmtRisk(p.value as number) : fmtInt(p.value as number),
                        color:
                          p.dataKey === "meanRisk" ? "#eab308" : p.dataKey === "alerts" ? "#f43f5e" : "#06b6d4",
                      }))}
                    />
                  )}
                />
                <Area
                  yAxisId="counts"
                  type="monotone"
                  dataKey="events"
                  stroke="#06b6d4"
                  strokeWidth={1}
                  fill="url(#gEvents)"
                  isAnimationActive={false}
                />
                <Area
                  yAxisId="counts"
                  type="monotone"
                  dataKey="alerts"
                  stroke="#f43f5e"
                  strokeWidth={1.4}
                  fill="url(#gAlerts)"
                  isAnimationActive={false}
                />
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="meanRisk"
                  stroke="#eab308"
                  strokeWidth={1.2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-2">
          <SectionHeader
            title="Attack Category Breakdown"
            hint="alerts by predicted category · bar color = mean risk"
            icon={<ListFilter className="h-4 w-4" aria-hidden />}
          />
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.categoryBreakdown}
                layout="vertical"
                margin={{ top: 0, right: 12, left: 8, bottom: 0 }}
                barCategoryGap={6}
              >
                <CartesianGrid stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="category"
                  tickLine={false}
                  axisLine={false}
                  width={96}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  cursor={{ fill: CHART.cursorFill }}
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as { category: string; count: number; meanRisk: number } | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={p.category}
                        entries={[
                          { name: "alerts", value: fmtInt(p.count), color: riskColor(p.meanRisk) },
                          { name: "mean risk", value: fmtRisk(p.meanRisk), color: "#eab308" },
                        ]}
                      />
                    );
                  }}
                />
                <Bar dataKey="count" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                  {data.categoryBreakdown.map((c) => (
                    <Cell key={c.category} fill={riskColor(c.meanRisk)} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      {/* Row 3: prioritized incidents */}
      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
        <SectionHeader
          title="What needs attention now?"
          hint="prioritized incidents · risk score descending"
          icon={<ShieldAlert className="h-4 w-4" aria-hidden />}
          action={
            <span className="font-mono text-[10px] text-muted-foreground">
              {topIncidents.length} open cases · click to investigate
            </span>
          }
        />
        <div className="soc-scroll max-h-96 divide-y divide-border/30 overflow-y-auto pr-1">
          {topIncidents.map((inc) => (
            <IncidentRow key={inc.incidentId} incident={inc} onSelect={onOpenIncident} />
          ))}
        </div>
      </Card>

      {/* Row 4: recent critical alerts + about data */}
      <section className="grid gap-4 lg:grid-cols-5" aria-label="recent alerts and data notes">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <SectionHeader
            title="Recent Critical Alerts"
            hint="high/critical severity · newest first"
            icon={<Flame className="h-4 w-4" aria-hidden />}
          />
          <div className="soc-scroll overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">
                  <th className="px-2 py-2 font-medium">Severity</th>
                  <th className="px-2 py-2 font-medium">Category</th>
                  <th className="px-2 py-2 text-right font-medium">Risk</th>
                  <th className="px-2 py-2 text-right font-medium">Anomaly</th>
                  <th className="px-2 py-2 text-right font-medium">Conf.</th>
                  <th className="px-2 py-2 font-medium">
                    Entity <span className="text-[#8b5cf6]">sim</span>
                  </th>
                  <th className="px-2 py-2 font-medium">Wave</th>
                  <th className="px-2 py-2 font-medium">Event</th>
                </tr>
              </thead>
              <tbody>
                {data.recentCritical.map((e) => (
                  <tr
                    key={e.eventId}
                    onClick={() => setDetailEvent(e)}
                    className="cursor-pointer border-b border-border/25 transition-colors hover:bg-accent/40"
                    tabIndex={0}
                    role="button"
                    aria-label={`Open event ${e.eventId}`}
                    onKeyDown={(ev) => {
                      if (ev.key === "Enter" || ev.key === " ") setDetailEvent(e);
                    }}
                  >
                    <td className="px-2 py-1.5">
                      <SeverityBadge severity={e.severity} />
                    </td>
                    <td className="px-2 py-1.5 text-foreground/90">{e.category}</td>
                    <td className="px-2 py-1.5 text-right font-mono tabular-nums" style={{ color: riskColor(e.riskScore) }}>
                      {fmtRisk(e.riskScore)}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono tabular-nums text-muted-foreground">
                      {fmtRisk(e.anomalyScore)}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono tabular-nums text-muted-foreground">
                      {fmtPct(e.attackConfidence, 0)}
                    </td>
                    <td className="px-2 py-1.5 font-mono text-muted-foreground">
                      {e.entity} <SimulatedTag />
                    </td>
                    <td className="px-2 py-1.5">
                      <WaveTag wave={e.wave} />
                    </td>
                    <td className="px-2 py-1.5 font-mono text-primary/80">{e.eventId}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-2">
          <SectionHeader
            title="About this data"
            icon={<Info className="h-4 w-4" aria-hidden />}
          />
          <div className="space-y-3">
            <p className="text-xs leading-relaxed text-foreground/80">{data.sampleDescription}</p>
            <div className="grid grid-cols-3 gap-2">
              <StatCell label="Boot scoring" value={`${(data.engineStats.bootScoringMs / 1000).toFixed(1)}s`} icon={<Gauge className="h-3 w-3" aria-hidden />} />
              <StatCell label="Events / sec" value={fmtInt(data.engineStats.eventsPerSec)} icon={<Activity className="h-3 w-3" aria-hidden />} />
              <StatCell label="Per-event" value={`${data.engineStats.singleEventLatencyMs.toFixed(2)}ms`} icon={<Crosshair className="h-3 w-3" aria-hidden />} />
            </div>
            <div className="rounded-lg border border-[#8b5cf6]/25 bg-[#8b5cf6]/8 px-3 py-2 text-[11px] leading-relaxed text-[#c4b5fd]">
              Timestamps, entity IDs and waves in this view are <strong>simulated replay metadata</strong> — the
              UNSW-NB15 CSV contains no capture times or host identifiers. Model outputs are research prototypes.
            </div>
            <p className="text-[10px] text-muted-foreground">
              Generated at {new Date(data.generatedAt).toLocaleTimeString("en-US")} · severity mix:{" "}
              {data.severityBreakdown.map((s) => `${s.severity} ${fmtInt(s.count)}`).join(" · ")}
            </p>
          </div>
        </Card>
      </section>

      <EventDetailDialog
        event={detailEvent}
        open={detailEvent !== null}
        onOpenChange={(open) => {
          if (!open) setDetailEvent(null);
        }}
      />
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: color }} aria-hidden />
      {label}
    </span>
  );
}

function StatCell({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/40 px-2.5 py-2">
      <p className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {icon}
        {label}
      </p>
      <p className="mt-0.5 font-mono text-sm font-bold tabular-nums text-foreground">{value}</p>
    </div>
  );
}
