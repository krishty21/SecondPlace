"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Boxes, Info, Radar } from "lucide-react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { socApi, type ScatterPoint } from "@/lib/soc-api";
import { CHART, CLUSTER_PALETTE, clusterColor } from "@/lib/soc-ui";
import { ChartTooltip, ErrorState, SectionHeader } from "@/components/soc/primitives";
import { ClusterCard } from "@/components/soc/cluster-card";

export function PatternExplorer() {
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ["patterns"],
    queryFn: ({ signal }) => socApi.patterns(signal),
  });
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const byCluster = useMemo(() => {
    const groups = new Map<number, ScatterPoint[]>();
    for (const p of data?.scatter ?? []) {
      const arr = groups.get(p.cluster) ?? [];
      arr.push(p);
      groups.set(p.cluster, arr);
    }
    return groups;
  }, [data?.scatter]);

  if (isPending) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-16 rounded-xl" />
        <div className="grid gap-4 lg:grid-cols-4">
          <Skeleton className="h-96 rounded-xl lg:col-span-3" />
          <Skeleton className="h-96 rounded-xl" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-64 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <ErrorState
        title="Pattern data unavailable"
        description="Could not load KMeans behavior clusters from the inference engine (port 3010)."
        onRetry={() => void refetch()}
        className="min-h-64"
      />
    );
  }

  const pca = data.pcaExplainedVariance ?? [];
  const pcaNote =
    pca.length >= 2
      ? `PCA components 1–2 explain ${(pca[0] * 100).toFixed(0)}% + ${(pca[1] * 100).toFixed(0)}% = ${((pca[0] + pca[1]) * 100).toFixed(0)}% of standardized behavioral variance.`
      : "PCA variance explained not available.";

  return (
    <div className="space-y-6">
      {/* header note */}
      <div
        role="note"
        className="flex items-start gap-2.5 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-xs leading-relaxed text-foreground/80"
      >
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden />
        <p>
          <span className="font-semibold text-primary">Behavior clusters</span> (KMeans k=8 on standardized behavioral
          features of training attacks). These are <span className="font-semibold">traffic-behavior groups — NOT malware
          families</span>. {pcaNote} Live alert counts reflect boot-state alerts scored by the real models.
        </p>
      </div>

      {/* scatter + sidebar */}
      <section className="grid gap-4 lg:grid-cols-4" aria-label="cluster scatter map">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <SectionHeader
            title="Behavioral cluster map"
            hint={`${data.scatter.length.toLocaleString("en-US")} training attacks · PCA space`
            }
            icon={<Radar className="h-4 w-4" aria-hidden />}
            action={
              highlighted !== null ? (
                <button
                  type="button"
                  onClick={() => setHighlighted(null)}
                  className="rounded-md border border-primary/40 bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary"
                >
                  C{highlighted} highlighted — clear
                </button>
              ) : (
                <span className="font-mono text-[10px] text-muted-foreground">click a cluster card to highlight</span>
              )
            }
          />
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                <CartesianGrid stroke={CHART.grid} />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PC1"
                  tickLine={false}
                  axisLine={{ stroke: CHART.axisLine }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PC2"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                />
                <ZAxis range={[12, 12]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.2)" }}
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as ScatterPoint | undefined;
                    if (!active || !p) return null;
                    return (
                      <ChartTooltip
                        active
                        label={`cluster C${p.cluster}`}
                        entries={[
                          { name: "category", value: p.category, color: clusterColor(p.cluster) },
                          { name: "PC1", value: p.x.toFixed(2), color: "#8b98a9" },
                          { name: "PC2", value: p.y.toFixed(2), color: "#8b98a9" },
                        ]}
                      />
                    );
                  }}
                />
                {[...byCluster.keys()]
                  .sort((a, b) => a - b)
                  .map((cluster) => (
                    <Scatter
                      key={cluster}
                      name={`C${cluster}`}
                      data={byCluster.get(cluster) ?? []}
                      fill={clusterColor(cluster)}
                      fillOpacity={highlighted === null || highlighted === cluster ? 0.55 : 0.08}
                      isAnimationActive={false}
                      shape="circle"
                    />
                  ))}
                {/* centroids as X marks */}
                <Scatter
                  name="centroids"
                  data={data.clusters.map((c) => ({
                    x: c.centroidPca.x,
                    y: c.centroidPca.y,
                    cluster: c.cluster,
                    dominant: c.dominant_category,
                  }))}
                  isAnimationActive={false}
                  shape={CentroidX}
                  legendType="none"
                />
                <ReferenceLine x={0} stroke="rgba(255,255,255,0.08)" />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
            {data.clusters.map((c) => (
              <button
                key={c.cluster}
                type="button"
                onClick={() => setHighlighted(highlighted === c.cluster ? null : c.cluster)}
                aria-pressed={highlighted === c.cluster}
                className="inline-flex items-center gap-1 rounded border border-border/40 px-1.5 py-0.5 font-mono text-[9px] transition-colors hover:border-primary/40"
                style={{ color: clusterColor(c.cluster) }}
              >
                <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: clusterColor(c.cluster) }} aria-hidden />
                C{c.cluster} · {c.dominant_category}
              </button>
            ))}
          </div>
        </Card>

        {/* sidebar: category distribution */}
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30">
          <SectionHeader title="Category distribution" hint="top 3 per cluster" icon={<Boxes className="h-4 w-4" aria-hidden />} />
          <div className="soc-scroll max-h-[420px] space-y-3 overflow-y-auto pr-1">
            {data.clusters.map((c) => {
              const top = Object.entries(c.category_distribution ?? {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3);
              const total = Object.values(c.category_distribution ?? {}).reduce((s, n) => s + n, 0) || 1;
              return (
                <div key={c.cluster} className="space-y-1 rounded-lg border border-border/40 bg-background/30 p-2.5">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px] font-bold" style={{ color: clusterColor(c.cluster) }}>
                      C{c.cluster}
                    </span>
                    <span className="font-mono text-[9px] text-muted-foreground">
                      {c.size.toLocaleString("en-US")} attacks
                    </span>
                  </div>
                  {top.map(([cat, n]) => (
                    <div key={cat} className="flex items-center gap-2">
                      <span className="w-24 shrink-0 truncate text-[10px] text-foreground/75">{cat}</span>
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/5">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${(n / total) * 100}%`,
                            backgroundColor: clusterColor(c.cluster),
                            opacity: 0.8,
                          }}
                        />
                      </div>
                      <span className="w-8 shrink-0 text-right font-mono text-[9px] tabular-nums text-muted-foreground">
                        {((n / total) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </Card>
      </section>

      {/* cluster cards grid */}
      <section aria-label="cluster profiles">
        <SectionHeader
          title="Cluster profiles"
          hint="signed z-scores vs the global attack mean · orange = above, cyan = below"
          icon={<Boxes className="h-4 w-4" aria-hidden />}
        />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {data.clusters.map((c) => (
            <ClusterCard
              key={c.cluster}
              profile={c}
              selected={highlighted === c.cluster}
              onSelect={(cluster) => setHighlighted(highlighted === cluster ? null : cluster)}
            />
          ))}
        </div>
        <p className="mt-3 text-[10px] text-muted-foreground">
          Features used for clustering: <span className="font-mono">{(data.featuresUsed ?? []).join(", ")}</span>. Palette
          fixed per cluster id ({CLUSTER_PALETTE.length} colors).
        </p>
      </section>
    </div>
  );
}

/** Custom X-mark shape for cluster centroids. */
function CentroidX(props: unknown) {
  const { cx, cy, payload } = props as { cx?: number; cy?: number; payload?: { cluster?: number } };
  const color = clusterColor(payload?.cluster ?? 0);
  const x = cx ?? 0;
  const y = cy ?? 0;
  const s = 7;
  return (
    <g>
      <line x1={x - s} y1={y - s} x2={x + s} y2={y + s} stroke={color} strokeWidth={2.2} strokeLinecap="round" />
      <line x1={x - s} y1={y + s} x2={x + s} y2={y - s} stroke={color} strokeWidth={2.2} strokeLinecap="round" />
      <circle cx={x} cy={y} r={9} fill="transparent" stroke={color} strokeWidth={1} strokeOpacity={0.35} />
    </g>
  );
}
