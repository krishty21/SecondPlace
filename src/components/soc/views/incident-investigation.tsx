"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Layers,
  ListChecks,
  MessageSquare,
  Microscope,
  Search,
  Send,
  Shield,
  Sparkles,
  Users,
} from "lucide-react";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { socApi, type AnalystChatMessage, type Incident } from "@/lib/soc-api";
import { CHART, CLUSTER_PALETTE, clusterColor, fmtInt, fmtMs, fmtPct, fmtRisk, riskColor } from "@/lib/soc-ui";
import {
  CategoryChip,
  ChartTooltip,
  CopyableId,
  EmptyState,
  ErrorState,
  EpistemicsBadge,
  SectionHeader,
  SeverityBadge,
  SimulatedTag,
  StatusBadge,
  WaveTag,
} from "@/components/soc/primitives";
import { KpiChip } from "@/components/soc/kpi-card";
import { RiskGauge } from "@/components/soc/risk-gauge";
import { IncidentRow } from "@/components/soc/incident-row";
import { EventDetailDialog } from "@/components/soc/event-detail";
import { maxContribution } from "@/components/soc/contribution-bar";
import { ClusterCard } from "@/components/soc/cluster-card";
import { cn } from "@/lib/utils";

export interface IncidentFocus {
  incidentId: string;
  fallback?: Incident;
}

const SEVERITIES = ["All", "Critical", "High", "Medium", "Low"] as const;

export function IncidentInvestigation({ focus }: { focus: IncidentFocus | null }) {
  const [severity, setSeverity] = useState<(typeof SEVERITIES)[number]>("All");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedFallback, setSelectedFallback] = useState<Incident | null>(null);
  const [tab, setTab] = useState("overview");

  // absorb focus requests coming from other views (Command Center / Live Replay)
  // React-recommended "derive state from props" pattern (setState during render, not in an effect)
  const [appliedFocus, setAppliedFocus] = useState<IncidentFocus | null>(focus);
  if (focus !== appliedFocus) {
    setAppliedFocus(focus);
    if (focus) {
      setSelectedId(focus.incidentId);
      setSelectedFallback(focus.fallback ?? null);
      setTab("overview");
    }
  }

  const listQuery = useQuery({
    queryKey: ["incidents", "list"],
    queryFn: ({ signal }) => socApi.incidents({ limit: 100 }, signal),
  });

  const preferFallback = selectedFallback !== null && selectedFallback.incidentId === selectedId;
  const detailQuery = useQuery({
    queryKey: ["incident", selectedId],
    queryFn: ({ signal }) => socApi.incidentDetail(selectedId as string, signal),
    enabled: selectedId !== null && !preferFallback,
    retry: 1,
  });

  const incidents = listQuery.data?.incidents ?? [];
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return incidents.filter((i) => {
      if (severity !== "All" && i.severity !== severity) return false;
      if (!q) return true;
      return (
        i.incidentId.toLowerCase().includes(q) ||
        i.category.toLowerCase().includes(q) ||
        i.title.toLowerCase().includes(q)
      );
    });
  }, [incidents, severity, search]);

  const incident: Incident | null = preferFallback
    ? selectedFallback
    : (detailQuery.data?.incident ?? null);
  const groundTruthMix: Record<string, number> | undefined = preferFallback
    ? computeGroundTruth(selectedFallback)
    : detailQuery.data?.groundTruthMix;

  return (
    <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
      {/* ---------------------------------------------------------- list */}
      <aside aria-label="incident list" className="space-y-3">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4">
          <SectionHeader
            title="Incidents"
            hint={listQuery.data ? `${listQuery.data.total} tracked` : undefined}
            icon={<Layers className="h-4 w-4" aria-hidden />}
          />

          {/* mobile incident picker */}
          {incident && (
            <div className="mb-3 lg:hidden">
              <Select
                value={incident.incidentId}
                onValueChange={(v) => {
                  setSelectedId(v);
                  setSelectedFallback(null);
                }}
              >
                <SelectTrigger className="h-9 w-full border-border/60 bg-background/50 font-mono text-xs">
                  <SelectValue placeholder="Select incident" />
                </SelectTrigger>
                <SelectContent className="max-h-72 soc-scroll">
                  {incidents.map((i) => (
                    <SelectItem key={i.incidentId} value={i.incidentId} className="font-mono text-xs">
                      {i.incidentId} · {i.category} · {fmtRisk(i.riskScore)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="mb-3 flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search ID / category…"
                aria-label="Search incidents"
                className="h-9 border-border/60 bg-background/50 pl-8 text-xs"
              />
            </div>
          </div>
          <div className="mb-2 flex flex-wrap gap-1" role="group" aria-label="severity filter">
            {SEVERITIES.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSeverity(s)}
                aria-pressed={severity === s}
                className={cn(
                  "rounded-md border px-2 py-0.5 text-[10px] font-semibold transition-colors",
                  severity === s
                    ? "border-primary/50 bg-primary/15 text-primary"
                    : "border-border/50 text-muted-foreground hover:border-primary/30 hover:text-foreground/80"
                )}
              >
                {s}
              </button>
            ))}
          </div>

          <div className="soc-scroll max-h-[calc(100vh-22rem)] overflow-y-auto pr-1 lg:max-h-96">
            {listQuery.isPending && (
              <div className="space-y-2">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-20 rounded-lg" />
                ))}
              </div>
            )}
            {listQuery.isError && (
              <ErrorState
                title="Incident list unavailable"
                description="Could not reach the correlation engine."
                onRetry={() => void listQuery.refetch()}
              />
            )}
            {listQuery.isSuccess && filtered.length === 0 && (
              <EmptyState title="No incidents match" description="Adjust the severity filter or search query." />
            )}
            {filtered.map((inc) => (
              <IncidentRow
                key={inc.incidentId}
                incident={inc}
                active={inc.incidentId === selectedId}
                onSelect={(i) => {
                  setSelectedId(i.incidentId);
                  setSelectedFallback(null);
                }}
              />
            ))}
          </div>
        </Card>
      </aside>

      {/* -------------------------------------------------------- detail */}
      <section aria-label="incident detail" className="min-w-0">
        {!selectedId && (
          <EmptyState
            className="min-h-[60vh]"
            icon={<Shield className="h-7 w-7" aria-hidden />}
            title="Select an incident to investigate"
            description="Pick a case from the list to see the full attack story, evidence events, model explanations and the AI analyst copilot."
          />
        )}

        {selectedId && !incident && detailQuery.isPending && (
          <div className="space-y-4">
            <Skeleton className="h-28 rounded-xl" />
            <Skeleton className="h-96 rounded-xl" />
          </div>
        )}

        {selectedId && !incident && detailQuery.isError && !preferFallback && (
          <ErrorState
            title={`Incident ${selectedId} not found`}
            description="The engine has no incident with this identifier. It may be a session-scoped replay incident from an older replay session."
            onRetry={() => void detailQuery.refetch()}
          />
        )}

        {incident && (
          <div className="space-y-4">
            {/* header card */}
            <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
                <RiskGauge value={incident.riskScore} size={116} label="risk" className="shrink-0" />
                <div className="min-w-0 flex-1 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <CopyableId id={incident.incidentId} label="incident id" className="text-sm font-semibold" />
                    {preferFallback && (
                      <Badge variant="outline" className="border-[#8b5cf6]/30 bg-[#8b5cf6]/10 font-mono text-[9px] uppercase tracking-wider text-[#a78bfa]">
                        replay session
                      </Badge>
                    )}
                    <StatusBadge status={incident.status} />
                    <SeverityBadge severity={incident.severity} />
                  </div>
                  <h1 className="text-lg font-semibold leading-tight text-foreground">{incident.title}</h1>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                    <CategoryChip category={incident.category} />
                    <span className="inline-flex items-center gap-1">
                      <Clock className="h-3 w-3" aria-hidden />
                      first seen <span className="font-mono">{fmtMs(incident.firstSeen)}</span> · last seen{" "}
                      <span className="font-mono">{fmtMs(incident.lastSeen)}</span> <SimulatedTag />
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <Users className="h-3 w-3" aria-hidden />
                      {incident.entities.length} pseudo-entities <SimulatedTag />
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <Activity className="h-3 w-3" aria-hidden />
                      {fmtInt(incident.alertCount)} correlated alerts
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            <Tabs value={tab} onValueChange={setTab}>
              <TabsList className="soc-scroll flex h-10 w-full justify-start gap-1 overflow-x-auto rounded-xl border border-border/50 bg-card/40 p-1">
                {[
                  { value: "overview", label: "Overview", icon: <Layers className="h-3.5 w-3.5" aria-hidden /> },
                  { value: "timeline", label: "Timeline", icon: <Clock className="h-3.5 w-3.5" aria-hidden /> },
                  { value: "evidence", label: "Evidence", icon: <Microscope className="h-3.5 w-3.5" aria-hidden /> },
                  { value: "explain", label: "Explainability", icon: <Brain className="h-3.5 w-3.5" aria-hidden /> },
                  { value: "ai", label: "AI Analyst", icon: <Sparkles className="h-3.5 w-3.5" aria-hidden /> },
                  { value: "patterns", label: "Related Patterns", icon: <Activity className="h-3.5 w-3.5" aria-hidden /> },
                ].map((t) => (
                  <TabsTrigger
                    key={t.value}
                    value={t.value}
                    className="flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-3 text-xs data-[state=active]:bg-primary/15 data-[state=active]:text-primary"
                  >
                    {t.icon}
                    {t.label}
                  </TabsTrigger>
                ))}
              </TabsList>

              <TabsContent value="overview" className="mt-4">
                <OverviewTab incident={incident} groundTruthMix={groundTruthMix} />
              </TabsContent>
              <TabsContent value="timeline" className="mt-4">
                <TimelineTab incident={incident} />
              </TabsContent>
              <TabsContent value="evidence" className="mt-4">
                <EvidenceTab incident={incident} />
              </TabsContent>
              <TabsContent value="explain" className="mt-4">
                <ExplainTab incident={incident} />
              </TabsContent>
              <TabsContent value="ai" className="mt-4">
                <AiAnalystTab incident={incident} groundTruthMix={groundTruthMix} />
              </TabsContent>
              <TabsContent value="patterns" className="mt-4">
                <RelatedPatternsTab incident={incident} />
              </TabsContent>
            </Tabs>
          </div>
        )}
      </section>
    </div>
  );
}

// ------------------------------------------------------------------ overview

function OverviewTab({
  incident,
  groundTruthMix,
}: {
  incident: Incident;
  groundTruthMix?: Record<string, number>;
}) {
  const mix = Object.entries(incident.categoryMix ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, value], i) => ({ name, value, color: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length] }));
  const gt = Object.entries(groundTruthMix ?? {})
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        <KpiChip label="Alerts" value={fmtInt(incident.alertCount)} />
        <KpiChip label="Events" value={fmtInt(incident.eventCount)} />
        <KpiChip label="Mean conf." value={fmtPct(incident.meanConfidence, 0)} />
        <KpiChip label="Mean anomaly" value={fmtRisk(incident.meanAnomaly)} />
        <KpiChip label="Peak anomaly" value={fmtRisk(incident.peakAnomaly)} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader title="Category mix" hint="predicted attack categories in this incident" />
          <div className="flex flex-col items-center gap-2 sm:flex-row">
            <div className="h-52 w-full sm:w-1/2">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={mix}
                    dataKey="value"
                    nameKey="name"
                    innerRadius="58%"
                    outerRadius="85%"
                    paddingAngle={3}
                    stroke="none"
                    isAnimationActive={false}
                  >
                    {mix.map((m) => (
                      <Cell key={m.name} fill={m.color} fillOpacity={0.85} />
                    ))}
                  </Pie>
                  <Tooltip
                    content={({ active, payload }) => {
                      const p = payload?.[0];
                      if (!active || !p) return null;
                      return (
                        <ChartTooltip
                          active
                          label={String(p.name)}
                          entries={[{ name: "alerts", value: String(p.value), color: p.payload.fill as string }]}
                        />
                      );
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <ul className="w-full space-y-1.5 sm:w-1/2">
              {mix.map((m) => (
                <li key={m.name} className="flex items-center justify-between gap-2 text-xs">
                  <span className="flex items-center gap-1.5 text-foreground/85">
                    <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: m.color }} aria-hidden />
                    {m.name}
                  </span>
                  <span className="font-mono tabular-nums text-muted-foreground">
                    {m.value} · {((m.value / Math.max(1, incident.alertCount)) * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader title="Risk trajectory" hint="binned incident risk over the simulated window" />
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={incident.riskTrajectory}
                margin={{ top: 4, right: 8, left: -22, bottom: 0 }}
              >
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis
                  dataKey="t"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={(v: number) => fmtMs(v)}
                  tickLine={false}
                  axisLine={{ stroke: CHART.axisLine }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fontSize: 10 }} width={40} />
                <Tooltip
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as { t: number; risk: number; count: number } | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={`t+${fmtMs(p.t)}`}
                        entries={[
                          { name: "risk", value: fmtRisk(p.risk), color: riskColor(p.risk) },
                          { name: "alerts in bin", value: String(p.count), color: "#8b98a9" },
                        ]}
                      />
                    );
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="risk"
                  stroke="#06b6d4"
                  strokeWidth={1.6}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader
            title="Containment playbook"
            hint="deterministic, category-aware recommendations"
            icon={<ListChecks className="h-4 w-4" aria-hidden />}
          />
          <ol className="space-y-2">
            {(incident.containmentPlaybook ?? []).map((step, i) => (
              <li key={i} className="flex items-start gap-2.5 rounded-lg border border-border/40 bg-background/30 px-3 py-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-primary/15 font-mono text-[10px] font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-xs leading-relaxed text-foreground/85">{step}</span>
              </li>
            ))}
            {(incident.containmentPlaybook ?? []).length === 0 && (
              <EmptyState title="No playbook steps" description="The engine did not attach a containment playbook." />
            )}
          </ol>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader
            title="Ground truth mix"
            hint="demo transparency only — NOT seen by the model"
            icon={<CheckCircle2 className="h-4 w-4" aria-hidden />}
          />
          <div className="rounded-lg border border-[#8b5cf6]/25 bg-[#8b5cf6]/8 px-3 py-2.5">
            {gt.length > 0 ? (
              <div className="space-y-2">
                {gt.map(([cat, n]) => (
                  <div key={cat} className="flex items-center gap-3">
                    <span className="w-28 shrink-0 truncate text-xs text-foreground/85">{cat}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                      <div
                        className="h-full rounded-full bg-[#8b5cf6]/70"
                        style={{ width: `${(n / gt[0][1]) * 100}%` }}
                      />
                    </div>
                    <span className="w-8 shrink-0 text-right font-mono text-xs tabular-nums text-muted-foreground">
                      {n}
                    </span>
                  </div>
                ))}
                <p className="text-[10px] leading-relaxed text-[#c4b5fd]">
                  Actual attack_cat labels of the correlated flows, shown for evaluation transparency. The detection
                  pipeline never reads these labels at inference time.
                </p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Ground truth unavailable for this incident.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ timeline

function TimelineTab({ incident }: { incident: Incident }) {
  const stages = incident.story ?? [];
  if (stages.length === 0) {
    return (
      <EmptyState
        title="Attack story still forming"
        description="The correlation engine composes the attack story when the incident is finalized — check back after the replay completes."
      />
    );
  }
  return (
    <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
      <SectionHeader
        title="Attack story"
        hint="stages marked with epistemic status · evidence links to scored events"
        icon={<Clock className="h-4 w-4" aria-hidden />}
      />
      <ol className="relative ml-3 space-y-3 border-l border-border/60 pl-6">
        {stages.map((stage) => (
          <li key={stage.index} className="relative">
            <span
              className="absolute -left-[31px] flex h-6 w-6 items-center justify-center rounded-full border border-border/60 bg-card font-mono text-[10px] font-bold text-primary"
              aria-hidden
            >
              {stage.index}
            </span>
            <div className="rounded-xl border border-border/50 bg-background/30 p-3.5 transition-colors hover:border-primary/25">
              <div className="mb-1.5 flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-foreground/95">{stage.title}</h3>
                <EpistemicsBadge kind={stage.epistemics} />
                <span className="ml-auto inline-flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
                  t+{fmtMs(stage.timestamp)} <SimulatedTag />
                </span>
              </div>
              <p className="text-xs leading-relaxed text-foreground/80">{stage.detail}</p>
              {stage.evidenceEventIds?.length > 0 && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">evidence:</span>
                  {stage.evidenceEventIds.map((id) => (
                    <Badge
                      key={id}
                      variant="outline"
                      className="border-border/50 bg-background/60 font-mono text-[9px] text-muted-foreground"
                    >
                      {id}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

// ------------------------------------------------------------------ evidence

function EvidenceTab({ incident }: { incident: Incident }) {
  const [detail, setDetail] = useState<(typeof incident.events)[number] | null>(null);
  const events = incident.events ?? [];
  if (events.length === 0) {
    return <EmptyState title="No evidence events" description="This incident carries no scored events." />;
  }
  return (
    <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
      <SectionHeader
        title="Evidence events"
        hint={`${events.length} scored flows · click a row for the full flow record & explanation`}
        icon={<Microscope className="h-4 w-4" aria-hidden />}
      />
      <div className="soc-scroll max-h-96 overflow-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-card/95 backdrop-blur">
            <tr className="border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">
              <th className="px-2 py-2 font-medium">Sev</th>
              <th className="px-2 py-2 font-medium">Category</th>
              <th className="px-2 py-2 text-right font-medium">Risk</th>
              <th className="px-2 py-2 text-right font-medium">Anom</th>
              <th className="px-2 py-2 text-right font-medium">Conf</th>
              <th className="px-2 py-2 font-medium">Entity</th>
              <th className="px-2 py-2 font-medium">Proto/Svc/State</th>
              <th className="px-2 py-2 font-medium">Top feature</th>
              <th className="px-2 py-2 font-medium">Event</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e) => {
              const top = e.explanation?.topPositive?.[0];
              return (
                <tr
                  key={e.eventId}
                  onClick={() => setDetail(e)}
                  tabIndex={0}
                  role="button"
                  aria-label={`Open evidence event ${e.eventId}`}
                  onKeyDown={(ev) => {
                    if (ev.key === "Enter" || ev.key === " ") setDetail(e);
                  }}
                  className="cursor-pointer border-b border-border/25 transition-colors hover:bg-accent/40"
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
                  <td className="px-2 py-1.5 font-mono text-[10px] text-muted-foreground">{e.entity}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-muted-foreground">
                    {e.raw?.proto}/{e.raw?.service}/{e.raw?.state}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-foreground/75">
                    {top ? `${top.feature} ${top.contribution >= 0 ? "+" : ""}${top.contribution.toFixed(2)}` : "—"}
                  </td>
                  <td className="px-2 py-1.5">
                    <span className="inline-flex items-center gap-1 font-mono text-primary/80">
                      {e.eventId} <WaveTag wave={e.wave} />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <EventDetailDialog
        event={detail}
        open={detail !== null}
        onOpenChange={(open) => {
          if (!open) setDetail(null);
        }}
      />
    </Card>
  );
}

// -------------------------------------------------------------- explainability

function ExplainTab({ incident }: { incident: Incident }) {
  const contributors = incident.topContributors ?? [];
  const scale = maxContribution(contributors);
  const narrative = incident.events?.[0]?.explanation?.narrative;

  return (
    <div className="space-y-4">
      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
        <SectionHeader
          title="Aggregated incident evidence"
          hint="summed feature attributions across all correlated alerts"
          icon={<Brain className="h-4 w-4" aria-hidden />}
        />
        {contributors.length > 0 ? (
          <div className="grid gap-x-8 gap-y-2.5 md:grid-cols-2">
            {contributors.map((c) => (
              <ContributionRow key={c.feature} feature={c.feature} contribution={c.contribution} scale={scale} />
            ))}
          </div>
        ) : (
          <EmptyState title="No aggregated attributions" />
        )}
        <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">
          Positive (rose) contributions pushed the model toward an attack verdict; negative (emerald) pushed toward
          normal. Values are summed per-feature attributions across the incident&apos;s alerts.
        </p>
      </Card>

      {narrative && (
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader title="Narrative" hint="generated for the highest-risk event" />
          <p className="rounded-lg border border-border/50 bg-background/40 px-3 py-2.5 text-xs leading-relaxed text-foreground/85">
            {narrative}
          </p>
        </Card>
      )}

      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4">
        <SectionHeader title="Methodology" />
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Incident attributions aggregate local explanations computed by the TypeScript inference engine: exact
          TreeSHAP values where precomputed for the event, otherwise Saabas path attributions computed live (verified:
          baseline + Σ contributions equals the raw model score). Incident risk combines calibrated attack confidence,
          anomaly score, category severity, class rarity, prediction uncertainty and correlation boost.
        </p>
      </Card>
    </div>
  );
}

function ContributionRow({
  feature,
  contribution,
  scale,
}: {
  feature: string;
  contribution: number;
  scale: number;
}) {
  const width = scale > 0 ? Math.min(100, (Math.abs(contribution) / scale) * 100) : 0;
  const positive = contribution >= 0;
  const color = positive ? "#f43f5e" : "#10b981";
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate font-mono text-[11px] text-foreground/85">{feature}</span>
        <span className="font-mono text-[10px] tabular-nums" style={{ color }}>
          {contribution >= 0 ? "+" : ""}
          {contribution.toFixed(2)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full"
          style={{ width: `${width}%`, backgroundColor: color, boxShadow: `0 0 6px ${color}30` }}
        />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ AI analyst

function buildEvidencePayload(incident: Incident, groundTruthMix?: Record<string, number>) {
  return {
    incidentId: incident.incidentId,
    title: incident.title,
    category: incident.category,
    severity: incident.severity,
    status: incident.status,
    riskScore: incident.riskScore,
    alertCount: incident.alertCount,
    eventCount: incident.eventCount,
    entities: incident.entities,
    meanConfidence: incident.meanConfidence,
    meanAnomaly: incident.meanAnomaly,
    peakAnomaly: incident.peakAnomaly,
    categoryMix: incident.categoryMix,
    riskTrajectory: (incident.riskTrajectory ?? []).slice(-8),
    topContributors: incident.topContributors,
    story: incident.story,
    containmentPlaybook: incident.containmentPlaybook,
    sampleEvents: (incident.events ?? [])
      .slice()
      .sort((a, b) => b.riskScore - a.riskScore)
      .slice(0, 5)
      .map((e) => ({
        category: e.category,
        attackProbability: e.attackProbability,
        anomalyScore: e.anomalyScore,
        riskScore: e.riskScore,
        topPositive: (e.explanation?.topPositive ?? []).slice(0, 3).map((t) => ({
          feature: t.feature,
          contribution: t.contribution,
        })),
      })),
    groundTruthMix,
  };
}

function AiAnalystTab({
  incident,
  groundTruthMix,
}: {
  incident: Incident;
  groundTruthMix?: Record<string, number>;
}) {
  const evidence = useMemo(
    () => buildEvidencePayload(incident, groundTruthMix),
    [incident, groundTruthMix]
  );
  const summary = useMutation({
    mutationFn: () => socApi.aiIncidentSummary(evidence),
  });

  return (
    <div className="space-y-4">
      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
        <SectionHeader
          title="AI Incident Summary"
          hint="LLM analyst grounded in the structured incident evidence"
          icon={<Sparkles className="h-4 w-4" aria-hidden />}
          action={
            <Button
              size="sm"
              onClick={() => summary.mutate()}
              disabled={summary.isPending}
              className="h-8 gap-1.5 bg-primary/90 text-primary-foreground hover:bg-primary"
            >
              {summary.isPending ? (
                <>
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" aria-hidden />
                  Analyzing…
                </>
              ) : (
                <>
                  <Brain className="h-3.5 w-3.5" aria-hidden />
                  Generate AI Incident Summary
                </>
              )}
            </Button>
          }
        />

        {summary.isPending && (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 rounded-lg" />
            ))}
          </div>
        )}

        {summary.isError && (
          <ErrorState
            title="Summary generation failed"
            description="The AI endpoint did not respond. Retry — a deterministic fallback keeps the product functional."
            onRetry={() => summary.mutate()}
          />
        )}

        {summary.data && (
          <div className="space-y-3">
            <Badge
              variant="outline"
              className={
                summary.data.source === "llm"
                  ? "border-primary/30 bg-primary/10 font-mono text-[9px] uppercase tracking-wider text-primary"
                  : "border-amber-500/30 bg-amber-500/10 font-mono text-[9px] uppercase tracking-wider text-amber-400"
              }
            >
              source: {summary.data.source === "llm" ? "llm" : "deterministic fallback"}
            </Badge>
            <SummaryCard title="Executive Summary" body={summary.data.sections.executiveSummary} />
            <SummaryCard title="Technical Analysis" body={summary.data.sections.technicalAnalysis} />
            <SummaryCard
              title="Why This Matters"
              body={summary.data.sections.whyThisMatters}
              icon={<AlertTriangle className="h-3.5 w-3.5 text-amber-400" aria-hidden />}
            />
            {summary.data.sections.recommendedInvestigation.length > 0 && (
              <div className="rounded-xl border border-border/50 bg-background/30 p-4">
                <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                  <ListChecks className="h-3.5 w-3.5 text-primary" aria-hidden />
                  Recommended Investigation
                </h3>
                <ol className="space-y-1.5">
                  {summary.data.sections.recommendedInvestigation.map((step, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-xs leading-relaxed text-foreground/85">
                      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded bg-primary/15 font-mono text-[9px] font-bold text-primary">
                        {i + 1}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {summary.data.sections.suggestedContainment.length > 0 && (
              <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
                <h3 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-400">
                  <Shield className="h-3.5 w-3.5" aria-hidden />
                  Suggested Containment — recommended actions
                </h3>
                <ul className="space-y-1.5">
                  {summary.data.sections.suggestedContainment.map((step, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-xs leading-relaxed text-foreground/85">
                      <Shield className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400/70" aria-hidden />
                      {step}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <SummaryCard title="Confidence & Caveats" body={summary.data.sections.confidenceCaveats} muted />
          </div>
        )}

        {!summary.data && !summary.isPending && !summary.isError && (
          <EmptyState
            icon={<Brain className="h-6 w-6" aria-hidden />}
            title="No summary generated yet"
            description="Generate an AI summary grounded in this incident's evidence — executive view, technical analysis, investigation checklist and containment considerations."
          />
        )}
      </Card>

      <AnalystChat evidence={evidence} incidentId={incident.incidentId} />
    </div>
  );
}

function SummaryCard({
  title,
  body,
  icon,
  muted = false,
}: {
  title: string;
  body: string;
  icon?: React.ReactNode;
  muted?: boolean;
}) {
  return (
    <div className={cn("rounded-xl border p-4", muted ? "border-border/40 bg-background/20" : "border-border/50 bg-background/30")}>
      <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {icon}
        {title}
      </h3>
      <p className="text-xs leading-relaxed text-foreground/85">{body}</p>
    </div>
  );
}

function AnalystChat({
  evidence,
  incidentId,
}: {
  evidence: Record<string, unknown>;
  incidentId: string;
}) {
  const [messages, setMessages] = useState<AnalystChatMessage[]>([]);
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const chat = useMutation({
    mutationFn: (userMessage: string) => {
      const next: AnalystChatMessage[] = [...messages, { role: "user", content: userMessage }];
      return socApi.aiAnalystChat(next.slice(-10), evidence);
    },
    onSuccess: (res) => {
      setMessages((prev) => [...prev, { role: "assistant", content: res.reply }]);
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "The analyst copilot is temporarily unavailable. Try again — the system remains fully functional without it.",
        },
      ]);
    },
  });

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages, chat.isPending]);

  const send = () => {
    const text = input.trim();
    if (!text || chat.isPending) return;
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    chat.mutate(text);
  };

  return (
    <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
      <SectionHeader
        title="Analyst Copilot"
        hint={`chat grounded in ${incidentId} evidence · last 10 messages sent as context`}
        icon={<MessageSquare className="h-4 w-4" aria-hidden />}
      />
      <div
        ref={listRef}
        className="soc-scroll mb-3 max-h-72 space-y-2.5 overflow-y-auto rounded-lg border border-border/40 bg-background/30 p-3"
        role="log"
        aria-label="analyst chat messages"
      >
        {messages.length === 0 && !chat.isPending && (
          <p className="py-6 text-center text-xs text-muted-foreground">
            Ask about this incident — e.g. &quot;why is the risk score this high?&quot; or &quot;which features drove the
            detection?&quot;
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={cn(
              "max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed",
              m.role === "user"
                ? "ml-auto bg-primary/15 text-foreground/90"
                : "mr-auto border border-border/50 bg-card/70 text-foreground/85"
            )}
          >
            {m.content}
          </div>
        ))}
        {chat.isPending && (
          <div className="mr-auto flex items-center gap-1.5 rounded-lg border border-border/50 bg-card/70 px-3 py-2">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary/70"
                style={{ animationDelay: `${i * 150}ms` }}
                aria-hidden
              />
            ))}
            <span className="sr-only">Analyst copilot is typing</span>
          </div>
        )}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the analyst copilot…"
          aria-label="Message to analyst copilot"
          className="h-9 border-border/60 bg-background/50 text-xs"
          disabled={chat.isPending}
        />
        <Button
          type="submit"
          size="sm"
          disabled={!input.trim() || chat.isPending}
          className="h-9 w-9 shrink-0 gap-0 bg-primary/90 p-0 text-primary-foreground hover:bg-primary"
          aria-label="Send message"
        >
          <Send className="h-4 w-4" aria-hidden />
        </Button>
      </form>
    </Card>
  );
}

// ------------------------------------------------------------ related patterns

function RelatedPatternsTab({ incident }: { incident: Incident }) {
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ["patterns"],
    queryFn: ({ signal }) => socApi.patterns(signal),
  });
  const [selected, setSelected] = useState<number | null>(null);

  const related = (data?.clusters ?? []).filter((c) => c.dominant_category === incident.category);
  const scatterOf = (cluster: number) => (data?.scatter ?? []).filter((p) => p.cluster === cluster);

  if (isPending) {
    return (
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 2 }).map((_, i) => (
          <Skeleton key={i} className="h-64 rounded-xl" />
        ))}
      </div>
    );
  }
  if (isError) {
    return (
      <ErrorState
        title="Pattern data unavailable"
        description="Could not load behavior clusters from the engine."
        onRetry={() => void refetch()}
      />
    );
  }
  if (related.length === 0) {
    return (
      <EmptyState
        title={`No behavior cluster dominated by ${incident.category}`}
        description="Clusters are behavior groups — this incident's predicted category does not dominate any cluster. See the Pattern Explorer for the full cluster map."
      />
    );
  }

  const active = selected !== null ? related.find((c) => c.cluster === selected) : null;

  return (
    <div className="space-y-4">
      <p className="text-xs leading-relaxed text-muted-foreground">
        KMeans behavior clusters whose dominant training category is{" "}
        <span className="font-mono text-primary">{incident.category}</span>. These are traffic-behavior groups — NOT
        malware families.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {related.map((c) => (
          <ClusterCard
            key={c.cluster}
            profile={c}
            selected={selected === c.cluster}
            onSelect={setSelected}
          />
        ))}
      </div>
      {active && (
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4">
          <SectionHeader
            title={`Cluster C${active.cluster} members`}
            hint={`${scatterOf(active.cluster).length.toLocaleString("en-US")} sampled attack flows in PCA space`}
            action={
              <Button variant="ghost" size="sm" className="h-7 gap-1 text-[11px] text-primary" onClick={() => setSelected(null)}>
                <ChevronRight className="h-3.5 w-3.5 rotate-90" aria-hidden />
                Collapse
              </Button>
            }
          />
          <div className="soc-scroll max-h-64 overflow-y-auto pr-1">
            <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
              {scatterOf(active.cluster)
                .slice(0, 90)
                .map((p, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-md border border-border/40 bg-background/30 px-2 py-1 font-mono text-[10px]"
                  >
                    <span className="text-foreground/80">{p.category}</span>
                    <span className="text-muted-foreground">
                      pca ({p.x.toFixed(1)}, {p.y.toFixed(1)})
                    </span>
                  </div>
                ))}
            </div>
          </div>
          <p className="mt-2 text-[10px] text-muted-foreground">
            Cluster color <span className="font-mono" style={{ color: clusterColor(active.cluster) }}>■</span> — first
            90 sampled points shown. Full 2,500-point map in the Pattern Explorer view.
          </p>
        </Card>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ helpers

function computeGroundTruth(incident: Incident | null): Record<string, number> | undefined {
  if (!incident) return undefined;
  const gt = new Map<string, number>();
  for (const e of incident.events ?? []) {
    const g = e.groundTruth ?? (e.raw?.attack_cat as string | undefined);
    if (g) gt.set(g, (gt.get(g) ?? 0) + 1);
  }
  return gt.size ? Object.fromEntries(gt) : undefined;
}
