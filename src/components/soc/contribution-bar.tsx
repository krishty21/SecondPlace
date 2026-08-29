"use client";

import type { FeatureAttribution } from "@/lib/soc-api";
import { fmtSigned } from "@/lib/soc-ui";
import { cn } from "@/lib/utils";

/**
 * Signed horizontal contribution bar.
 * Positive contributions (increase risk) render rose; negative render emerald.
 */
export function ContributionBar({
  feature,
  value,
  contribution,
  maxAbs,
  showValue = true,
  valueFormatter,
  className,
}: {
  feature: string;
  value?: number;
  contribution: number;
  maxAbs: number;
  showValue?: boolean;
  valueFormatter?: (v: number) => string;
  className?: string;
}) {
  const width = maxAbs > 0 ? Math.min(100, (Math.abs(contribution) / maxAbs) * 100) : 0;
  const positive = contribution >= 0;
  const color = positive ? "#f43f5e" : "#10b981";
  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate font-mono text-[11px] text-foreground/85">{feature}</span>
        <span className="flex shrink-0 items-baseline gap-2 font-mono text-[10px] tabular-nums">
          {showValue && value !== undefined && (
            <span className="text-muted-foreground">val {valueFormatter ? valueFormatter(value) : value.toFixed(2)}</span>
          )}
          <span style={{ color }}>{fmtSigned(contribution)}</span>
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5" role="presentation">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${width}%`, backgroundColor: color, boxShadow: `0 0 8px ${color}33` }}
        />
      </div>
    </div>
  );
}

/** Compute max |contribution| for a set of attributions. */
export function maxContribution(items: { contribution: number }[]): number {
  return items.reduce((m, i) => Math.max(m, Math.abs(i.contribution)), 0);
}

export function AttributionList({
  items,
  title,
  emptyText,
  maxAbs,
}: {
  items: FeatureAttribution[];
  title?: string;
  emptyText?: string;
  maxAbs?: number;
}) {
  if (!items.length) {
    return <p className="text-xs text-muted-foreground">{emptyText ?? "No attributions."}</p>;
  }
  const scale = maxAbs ?? maxContribution(items);
  return (
    <div className="space-y-2.5">
      {title && (
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{title}</p>
      )}
      {items.map((a) => (
        <ContributionBar
          key={a.feature}
          feature={a.feature}
          value={a.value}
          contribution={a.contribution}
          maxAbs={scale}
        />
      ))}
    </div>
  );
}
