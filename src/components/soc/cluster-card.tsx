"use client";

import { Activity } from "lucide-react";
import type { ClusterProfile } from "@/lib/soc-api";
import { clusterColor } from "@/lib/soc-ui";
import { CategoryChip } from "@/components/soc/primitives";
import { cn } from "@/lib/utils";

/**
 * Card describing one KMeans behavior cluster: dominant category, size,
 * live alert count and signed z-score feature bars.
 */
export function ClusterCard({
  profile,
  selected = false,
  highlighted = false,
  onSelect,
}: {
  profile: ClusterProfile;
  selected?: boolean;
  highlighted?: boolean;
  onSelect?: (cluster: number) => void;
}) {
  const color = clusterColor(profile.cluster);
  const topCats = Object.entries(profile.category_distribution ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  const totalCats = topCats.reduce((s, [, n]) => s + n, 0) || 1;
  const maxZ = Math.max(...(profile.top_features ?? []).map((f) => Math.abs(f.z_score)), 0.01);

  return (
    <button
      type="button"
      onClick={onSelect ? () => onSelect(profile.cluster) : undefined}
      aria-label={`Cluster ${profile.cluster}: dominant category ${profile.dominant_category}`}
      className={cn(
        "w-full rounded-xl border bg-card/50 p-4 text-left transition-all duration-200 hover:border-primary/30",
        selected ? "border-primary/50 ring-1 ring-primary/25" : "border-border/50",
        highlighted && !selected && "border-primary/25"
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span
            className="flex h-6 w-8 items-center justify-center rounded-md font-mono text-[11px] font-bold"
            style={{ color, backgroundColor: `${color}16`, border: `1px solid ${color}38` }}
          >
            C{profile.cluster}
          </span>
          <CategoryChip category={profile.dominant_category} hex={color} />
        </div>
        <span className="inline-flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
          <Activity className="h-3 w-3 text-emerald-400" aria-hidden />
          {profile.liveAlertCount.toLocaleString("en-US")} live
        </span>
      </div>

      <div className="mb-2 flex items-center gap-3 text-[11px] text-muted-foreground">
        <span>
          <span className="font-mono font-semibold text-foreground/80">{profile.size.toLocaleString("en-US")}</span>{" "}
          train attacks
        </span>
        <span className="font-mono">
          pca ({profile.centroidPca.x.toFixed(1)}, {profile.centroidPca.y.toFixed(1)})
        </span>
      </div>

      {/* top features as signed z-score bars */}
      <div className="space-y-1.5">
        {(profile.top_features ?? []).slice(0, 5).map((f) => {
          const w = Math.min(100, (Math.abs(f.z_score) / maxZ) * 100);
          const pos = f.z_score >= 0;
          return (
            <div key={f.feature} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate font-mono text-[10px] text-foreground/75">{f.feature}</span>
              <div className="relative h-1 flex-1 rounded-full bg-white/5">
                <div className="absolute left-1/2 top-[-2px] h-[9px] w-px bg-white/15" aria-hidden />
                <div
                  className="absolute top-0 h-1 rounded-full"
                  style={{
                    width: `${w / 2}%`,
                    left: pos ? "50%" : `${50 - w / 2}%`,
                    backgroundColor: pos ? "#f97316" : "#06b6d4",
                  }}
                />
              </div>
              <span className="w-10 shrink-0 text-right font-mono text-[9px] tabular-nums text-muted-foreground">
                {f.z_score >= 0 ? "+" : ""}
                {f.z_score.toFixed(1)}σ
              </span>
            </div>
          );
        })}
      </div>

      {/* category distribution segmented bar */}
      <div className="mt-3">
        <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-white/5">
          {topCats.map(([cat, n], i) => (
            <div
              key={cat}
              title={`${cat}: ${n}`}
              style={{
                width: `${(n / totalCats) * 100}%`,
                backgroundColor: clusterColor(profile.cluster + i),
                opacity: 1 - i * 0.25,
              }}
              aria-hidden
            />
          ))}
        </div>
        <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5">
          {topCats.map(([cat, n]) => (
            <span key={cat} className="text-[9px] text-muted-foreground">
              {cat} <span className="font-mono">{((n / totalCats) * 100).toFixed(0)}%</span>
            </span>
          ))}
        </div>
      </div>
    </button>
  );
}
