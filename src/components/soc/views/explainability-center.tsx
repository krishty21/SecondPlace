"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FlaskConical, Microscope, Scale, Sigma } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { socApi, UNSW_CATEGORIES, type ScoredEvent } from "@/lib/soc-api";
import { CHART, fmtPct, fmtRisk } from "@/lib/soc-ui";
import { ChartTooltip, ErrorState, SectionHeader, SeverityBadge } from "@/components/soc/primitives";
import { RiskGauge } from "@/components/soc/risk-gauge";
import { AttributionList, maxContribution } from "@/components/soc/contribution-bar";

export function ExplainabilityCenter() {
  const globalQuery = useQuery({
    queryKey: ["explain-global"],
    queryFn: ({ signal }) => socApi.explainGlobal(signal),
  });
  const modelQuery = useQuery({
    queryKey: ["model-info"],
    queryFn: ({ signal }) => socApi.modelInfo(signal),
  });

  if (globalQuery.isPending || modelQuery.isPending) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 lg:grid-cols-2">
          <Skeleton className="h-96 rounded-xl" />
          <Skeleton className="h-96 rounded-xl" />
        </div>
        <Skeleton className="h-96 rounded-xl" />
      </div>
    );
  }

  if (globalQuery.isError || !globalQuery.data) {
    return (
      <ErrorState
        title="Explainability data unavailable"
        description="Could not load global SHAP / calibration artifacts from the inference engine."
        onRetry={() => void globalQuery.refetch()}
        className="min-h-64"
      />
    );
  }

  return (
    <div className="space-y-6">
      <GlobalImportance data={globalQuery.data} />
      <Calibration data={globalQuery.data} />
      <LocalExplorer />
      <ModelComparison
        modelInfo={modelQuery.data}
        isError={modelQuery.isError}
        onRetry={() => void modelQuery.refetch()}
        isPending={modelQuery.isPending}
      />
    </div>
  );
}

type GlobalData = Awaited<ReturnType<typeof socApi.explainGlobal>>;

// ------------------------------------------------------------ section A

function GlobalImportance({ data }: { data: GlobalData }) {
  const top20 = useMemo(
    () =>
      [...(data.shapGlobal.features ?? [])]
        .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
        .slice(0, 20)
        .reverse(),
    [data.shapGlobal.features]
  );
  const mcTop = useMemo(
    () =>
      [...(data.multiclassGain?.features ?? [])]
        .sort((a, b) => b.gain - a.gain)
        .slice(0, 12)
        .reverse(),
    [data.multiclassGain]
  );

  return (
    <section aria-label="global feature importance">
      <SectionHeader
        title="Global Feature Importance"
        hint="what drives the binary attack detector across the training distribution"
        icon={<Sigma className="h-4 w-4" aria-hidden />}
      />
      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <p className="text-xs font-semibold text-foreground/90">Top 20 features · mean |SHAP|</p>
            <Badge variant="outline" className="border-primary/30 bg-primary/10 font-mono text-[9px] text-primary">
              {data.shapGlobal.method}
            </Badge>
          </div>
          <div className="h-[440px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top20} layout="vertical" margin={{ top: 0, right: 12, left: 8, bottom: 0 }} barCategoryGap={4}>
                <CartesianGrid stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tickLine={false}
                  axisLine={false}
                  width={130}
                  tick={{ fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                />
                <Tooltip
                  cursor={{ fill: CHART.cursorFill }}
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as { feature: string; mean_abs_shap: number } | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={p.feature}
                        entries={[{ name: "mean |SHAP|", value: p.mean_abs_shap.toFixed(3), color: "#06b6d4" }]}
                      />
                    );
                  }}
                />
                <Bar dataKey="mean_abs_shap" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                  {top20.map((f, i) => (
                    <Cell
                      key={f.feature}
                      fill={i >= top20.length - 5 ? "#06b6d4" : "#0e7490"}
                      fillOpacity={0.9 - (Math.abs(i - top20.length) / top20.length) * 0.45}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">
            Method: {data.shapGlobal.method} · expected model output (baseline) ={" "}
            <span className="font-mono">{data.shapGlobal.expected_value.toFixed(4)}</span>
          </p>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-2">
          <p className="mb-2 text-xs font-semibold text-foreground/90">Multiclass model · top gain features</p>
          <div className="h-[440px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mcTop} layout="vertical" margin={{ top: 0, right: 12, left: 8, bottom: 0 }} barCategoryGap={5}>
                <CartesianGrid stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tickLine={false}
                  axisLine={false}
                  width={130}
                  tick={{ fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                />
                <Tooltip
                  cursor={{ fill: CHART.cursorFill }}
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as { feature: string; gain: number } | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={p.feature}
                        entries={[{ name: "split gain", value: p.gain.toLocaleString("en-US"), color: "#14b8a6" }]}
                      />
                    );
                  }}
                />
                <Bar dataKey="gain" radius={[0, 3, 3, 0]} fill="#14b8a6" fillOpacity={0.8} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">
            Cumulative LightGBM split gain of the 10-class attack-category model.
          </p>
        </Card>
      </div>
    </section>
  );
}

// ------------------------------------------------------------ section B

function Calibration({ data }: { data: GlobalData }) {
  const cal = data.calibration;
  const reliability = (cal.oof_reliability ?? []).map((b) => ({
    predicted: b.mean_predicted,
    actual: b.fraction_positive,
    count: b.count,
  }));
  const diagonal = [
    { predicted: 0, actual: 0 },
    { predicted: 1, actual: 1 },
  ];
  const thresholdCurve = cal.threshold_curve ?? [];
  const improvement = cal.oof_brier_raw - cal.oof_brier_platt;

  return (
    <section aria-label="model calibration">
      <SectionHeader
        title="Model Calibration & Threshold"
        hint="5-fold out-of-fold predictions on the training set"
        icon={<Scale className="h-4 w-4" aria-hidden />}
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <p className="mb-2 text-xs font-semibold text-foreground/90">Reliability curve (OOF, after calibration)</p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reliability} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={CHART.grid} />
                <XAxis
                  type="number"
                  dataKey="predicted"
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  tickLine={false}
                  axisLine={{ stroke: CHART.axisLine }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                  width={44}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as { predicted: number; actual: number; count: number } | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={`predicted ${fmtPct(p.predicted)}`}
                        entries={[
                          { name: "observed attack rate", value: fmtPct(p.actual), color: "#06b6d4" },
                          { name: "bin count", value: String(p.count), color: "#8b98a9" },
                        ]}
                      />
                    );
                  }}
                />
                <Line
                  data={diagonal}
                  type="linear"
                  dataKey="actual"
                  stroke="rgba(255,255,255,0.25)"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  isAnimationActive={false}
                  name="perfect"
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#06b6d4"
                  strokeWidth={1.8}
                  dot={{ r: 2.5, fill: "#06b6d4" }}
                  isAnimationActive={false}
                  name="model"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2">
            <MetricCell label="Brier (raw)" value={cal.oof_brier_raw.toFixed(4)} />
            <MetricCell label="Brier (calibrated)" value={cal.oof_brier_platt.toFixed(4)} accent="#10b981" />
            <MetricCell label="Δ improvement" value={`−${improvement.toFixed(4)}`} accent="#10b981" />
          </div>
          <p className="mt-2 text-[10px] text-muted-foreground">
            Platt scaling a=<span className="font-mono">{cal.platt?.a?.toFixed(3)}</span>, b=
            <span className="font-mono">{cal.platt?.b?.toFixed(3)}</span> · temperature scaling T=
            <span className="font-mono">{cal.temperature?.toFixed(2)}</span>
          </p>
        </Card>

        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <p className="mb-2 text-xs font-semibold text-foreground/90">
            Precision / Recall / F1 vs decision threshold
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={thresholdCurve} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={CHART.grid} />
                <XAxis
                  dataKey="threshold"
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  tickLine={false}
                  axisLine={{ stroke: CHART.axisLine }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                  width={44}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    return (
                      <ChartTooltip
                        active
                        label={`threshold ${Number(label).toFixed(2)}`}
                        entries={payload.map((p) => ({
                          name: String(p.dataKey),
                          value: fmtPct(p.value as number, 1),
                          color: p.stroke as string,
                        }))}
                      />
                    );
                  }}
                />
                {cal.chosen_threshold !== undefined && (
                  <ReferenceLine
                    x={cal.chosen_threshold}
                    stroke="#f43f5e"
                    strokeWidth={1.2}
                    strokeDasharray="4 3"
                    label={{
                      value: `chosen ${cal.chosen_threshold.toFixed(2)}`,
                      position: "top",
                      fill: "#f43f5e",
                      fontSize: 10,
                    }}
                  />
                )}
                <Line type="monotone" dataKey="precision" stroke="#06b6d4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="recall" stroke="#f97316" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="f1" stroke="#10b981" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-[10px] text-muted-foreground">
            <Legend color="#06b6d4" label="precision" />
            <Legend color="#f97316" label="recall" />
            <Legend color="#10b981" label="F1" />
            <Legend color="#f43f5e" label={`operating threshold ${cal.chosen_threshold?.toFixed(2)}`} dashed />
          </div>
        </Card>
      </div>
    </section>
  );
}

function Legend({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-0.5 w-4"
        style={{ backgroundColor: dashed ? "transparent" : color, borderTop: dashed ? `2px dashed ${color}` : undefined }}
        aria-hidden
      />
      {label}
    </span>
  );
}

function MetricCell({ label, value, accent = "#06b6d4" }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/40 px-2.5 py-2">
      <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-0.5 font-mono text-sm font-bold tabular-nums" style={{ color: accent }}>
        {value}
      </p>
    </div>
  );
}

// ------------------------------------------------------------ section C

function LocalExplorer() {
  const [category, setCategory] = useState<string>("Exploits");
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  const eventsQuery = useQuery({
    queryKey: ["events", category],
    queryFn: ({ signal }) => socApi.events({ category, n: 25 }, signal),
  });

  const explainMutation = useMutation({
    mutationFn: (raw: Parameters<typeof socApi.explainEvent>[0]) => socApi.explainEvent(raw),
  });

  const events = eventsQuery.data?.events ?? [];
  const selected = events.find((e) => e.eventId === selectedEventId) ?? null;
  const explained = explainMutation.data ?? null;

  const runExplain = (eventId: string) => {
    setSelectedEventId(eventId);
    const ev = events.find((e) => e.eventId === eventId);
    if (ev) explainMutation.mutate(ev.raw);
  };

  return (
    <section aria-label="local explanation explorer">
      <SectionHeader
        title="Local Explanation Explorer"
        hint="pick a real test-set event and run it through the live engine"
        icon={<Microscope className="h-4 w-4" aria-hidden />}
      />
      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <Select value={category} onValueChange={(v) => { setCategory(v); setSelectedEventId(null); }}>
            <SelectTrigger className="h-9 w-56 border-border/60 bg-background/50 text-xs" aria-label="Select attack category">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="soc-scroll max-h-72">
              {UNSW_CATEGORIES.map((c) => (
                <SelectItem key={c} value={c} className="text-xs">
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="font-mono text-[10px] text-muted-foreground">
            GET /api/events?category={category}&n=25 — deterministic real test rows, scored live
          </span>
        </div>

        {eventsQuery.isPending && <Skeleton className="h-24 rounded-lg" />}
        {eventsQuery.isError && (
          <ErrorState
            title="Could not load events"
            description="The engine did not return test-set events for this category."
            onRetry={() => void eventsQuery.refetch()}
          />
        )}

        {eventsQuery.isSuccess && (
          <>
            <div className="soc-scroll mb-4 flex max-h-28 flex-wrap gap-1.5 overflow-y-auto pr-1">
              {events.map((e) => (
                <button
                  key={e.eventId}
                  type="button"
                  onClick={() => runExplain(e.eventId)}
                  aria-pressed={selectedEventId === e.eventId}
                  className={`rounded-md border px-2 py-1 font-mono text-[10px] transition-colors ${
                    selectedEventId === e.eventId
                      ? "border-primary/50 bg-primary/15 text-primary"
                      : "border-border/50 bg-background/40 text-muted-foreground hover:border-primary/30 hover:text-foreground/85"
                  }`}
                >
                  {e.eventId}
                  <span className="ml-1.5" style={{ color: e.severity === "Critical" ? "#f43f5e" : "#8b98a9" }}>
                    {fmtRisk(e.riskScore)}
                  </span>
                </button>
              ))}
              {events.length === 0 && (
                <p className="text-xs text-muted-foreground">No events found for this category.</p>
              )}
            </div>

            {!selected && !explainMutation.isPending && (
              <p className="py-6 text-center text-xs text-muted-foreground">
                Select an event chip above to score it live via POST /api/explain and get a full local explanation.
              </p>
            )}

            {(explainMutation.isPending || explained) && selected && (
              <div>
                {explainMutation.isPending && <Skeleton className="h-64 rounded-lg" />}
                {explained && <LocalExplanationResult event={explained} />}
              </div>
            )}
            {explainMutation.isError && selected && (
              <ErrorState
                title="Explanation failed"
                description="POST /api/explain did not succeed. Retry."
                onRetry={() => selected && explainMutation.mutate(selected.raw)}
              />
            )}
          </>
        )}
      </Card>
    </section>
  );
}

function LocalExplanationResult({ event }: { event: ScoredEvent }) {
  const expl = event.explanation;
  const scale = maxContribution([...expl.topPositive, ...expl.topNegative]);
  const probs = Object.entries(event.categoryProbs ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
        <div className="flex flex-col items-center gap-2">
          <RiskGauge value={event.riskScore} size={128} label="risk" />
          <SeverityBadge severity={event.severity} />
        </div>
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm font-semibold text-primary">{event.eventId}</span>
            <Badge variant="outline" className="border-primary/30 bg-primary/10 font-mono text-[9px] uppercase tracking-wider text-primary">
              {expl.method}
            </Badge>
            <span className="font-mono text-[10px] text-muted-foreground">
              baseline (expected logit) = <span className="font-mono text-foreground/80">{expl.baseline.toFixed(4)}</span>
            </span>
          </div>
          <p className="rounded-lg border border-border/50 bg-background/40 px-3 py-2 text-xs leading-relaxed text-foreground/85">
            {expl.narrative}
          </p>
          <div className="grid grid-cols-3 gap-2">
            <MetricCell label="Attack prob." value={fmtPct(event.attackProbability)} accent="#f43f5e" />
            <MetricCell label="Anomaly score" value={fmtRisk(event.anomalyScore)} accent="#eab308" />
            <MetricCell label="Verdict" value={event.binaryVerdict} accent="#06b6d4" />
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-3.5">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-rose-400">
            Increases risk — top positive contributions
          </p>
          <AttributionList items={expl.topPositive} maxAbs={scale} />
        </div>
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-3.5">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-emerald-400">
            Decreases risk — top negative contributions
          </p>
          <AttributionList items={expl.topNegative} maxAbs={scale} />
        </div>
      </div>

      <div className="rounded-xl border border-border/50 bg-background/30 p-3.5">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Predicted category probabilities (all 10 classes)
        </p>
        <div className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
          {probs.map(([cat, p], i) => (
            <div key={cat} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate font-mono text-[10px] text-foreground/80">{cat}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.max(0.5, Math.min(100, p * 100))}%`,
                    backgroundColor: i === 0 ? "#06b6d4" : "rgba(6,182,212,0.45)",
                  }}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono text-[9px] tabular-nums text-muted-foreground">
                {fmtPct(p, 2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ section D

function ModelComparison({
  modelInfo,
  isPending,
  isError,
  onRetry,
}: {
  modelInfo: Awaited<ReturnType<typeof socApi.modelInfo>> | undefined;
  isPending: boolean;
  isError: boolean;
  onRetry: () => void;
}) {
  if (isPending) return <Skeleton className="h-80 rounded-xl" />;
  if (isError || !modelInfo) {
    return (
      <ErrorState
        title="Model registry unavailable"
        description="Could not load model comparison data from the engine."
        onRetry={onRetry}
      />
    );
  }

  const binary = modelInfo.comparison?.binary ?? [];
  const multiclass = modelInfo.comparison?.multiclass ?? [];
  const abl = modelInfo.featureAblation ?? {};

  return (
    <section aria-label="model comparison">
      <SectionHeader
        title="Model Comparison & Feature Ablation"
        hint="validation split of the training set (train-only, no test leakage)"
        icon={<FlaskConical className="h-4 w-4" aria-hidden />}
      />
      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <p className="mb-2 text-xs font-semibold text-foreground/90">Binary attack detector candidates</p>
          <ComparisonTable
            rows={binary}
            columns={[
              { key: "model", label: "Model" },
              { key: "f1", label: "F1", format: (v: number) => v.toFixed(4), highlight: true },
              { key: "roc_auc", label: "AUC", format: (v: number) => v.toFixed(4), highlight: true },
              { key: "precision", label: "Precision", format: (v: number) => v.toFixed(3) },
              { key: "recall", label: "Recall", format: (v: number) => v.toFixed(3) },
              { key: "fit_seconds", label: "Fit (s)", format: (v: number) => v.toFixed(1) },
            ]}
            bestKey="f1"
          />
          <p className="mb-2 mt-5 text-xs font-semibold text-foreground/90">Attack-category (10-class) candidates</p>
          <ComparisonTable
            rows={multiclass}
            columns={[
              { key: "model", label: "Model" },
              { key: "macro_f1", label: "Macro F1", format: (v: number) => v.toFixed(4), highlight: true },
              { key: "accuracy", label: "Accuracy", format: (v: number) => v.toFixed(3) },
              { key: "weighted_f1", label: "Weighted F1", format: (v: number) => v.toFixed(3) },
              { key: "fit_seconds", label: "Fit (s)", format: (v: number) => v.toFixed(1) },
            ]}
            bestKey="macro_f1"
          />
        </Card>

        <div className="space-y-4 lg:col-span-2">
          <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
            <p className="mb-3 text-xs font-semibold text-foreground/90">Feature ablation</p>
            <div className="grid grid-cols-3 gap-2">
              <MetricCell label="Raw features F1" value={abl.raw_only_f1 !== undefined ? Number(abl.raw_only_f1).toFixed(4) : "—"} />
              <MetricCell label="Full pipeline F1" value={abl.full_pipeline_f1 !== undefined ? Number(abl.full_pipeline_f1).toFixed(4) : "—"} />
              <MetricCell
                label="Engineered Δ"
                value={abl.delta !== undefined ? `${abl.delta >= 0 ? "+" : ""}${Number(abl.delta).toFixed(4)}` : "—"}
                accent={abl.delta !== undefined && abl.delta >= 0 ? "#10b981" : "#f43f5e"}
              />
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">
              Engineering features (log1p transforms, categorical encoding, derived ratios) vs raw numeric features
              only, same model family.
            </p>
          </Card>

          <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4">
            <p className="mb-3 text-xs font-semibold text-foreground/90">Registry</p>
            <dl className="space-y-1.5 text-[11px]">
              <Row label="Model package" value={modelInfo.registry?.name ?? "—"} mono />
              <Row label="Version" value={`v${modelInfo.registry?.version ?? "—"}`} mono />
              <Row label="Trained at" value={modelInfo.registry?.trained_at ?? "—"} mono />
              <Row label="Train rows" value={(modelInfo.registry?.dataset?.train_rows ?? 0).toLocaleString("en-US")} mono />
              <Row label="Test rows" value={(modelInfo.registry?.dataset?.test_rows ?? 0).toLocaleString("en-US")} mono />
              <Row label="Features" value={String(modelInfo.registry?.feature_count ?? "—")} mono />
              <Row label="Decision threshold" value={String(modelInfo.registry?.threshold ?? "—")} mono />
            </dl>
          </Card>
        </div>
      </div>
    </section>
  );
}

interface ColumnDef {
  key: string;
  label: string;
  format?: (v: number) => string;
  highlight?: boolean;
}

function ComparisonTable({
  rows,
  columns,
  bestKey,
}: {
  rows: Record<string, unknown>[];
  columns: ColumnDef[];
  bestKey: string;
}) {
  const bestVal = Math.max(...rows.map((r) => Number(r[bestKey] ?? 0)));
  return (
    <div className="soc-scroll overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-border/50 text-[10px] uppercase tracking-wider text-muted-foreground">
            {columns.map((c) => (
              <th key={c.key} className={`px-2 py-2 font-medium ${c.key === "model" ? "" : "text-right"}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const isBest = Number(row[bestKey] ?? 0) === bestVal;
            return (
              <tr
                key={i}
                className={`border-b border-border/25 ${isBest ? "bg-primary/5" : ""}`}
              >
                {columns.map((c) => {
                  const v = row[c.key];
                  if (c.key === "model") {
                    return (
                      <td key={c.key} className="px-2 py-1.5">
                        <span className="inline-flex items-center gap-1.5 text-foreground/90">
                          {String(v)}
                          {isBest && (
                            <Badge variant="outline" className="border-primary/30 bg-primary/10 text-[8px] uppercase tracking-wider text-primary">
                              selected
                            </Badge>
                          )}
                        </span>
                      </td>
                    );
                  }
                  const num = Number(v);
                  return (
                    <td
                      key={c.key}
                      className={`px-2 py-1.5 text-right font-mono tabular-nums ${
                        c.highlight && isBest ? "font-bold text-primary" : "text-muted-foreground"
                      }`}
                    >
                      {Number.isFinite(num) && v !== undefined && v !== null ? c.format?.(num) ?? num.toFixed(3) : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-border/25 pb-1">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={`truncate text-foreground/85 ${mono ? "font-mono text-[10px]" : ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}
