"use client";

import type { ReactNode } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useCountUp } from "@/components/soc/risk-gauge";
import { cn } from "@/lib/utils";

export function KpiCard({
  icon,
  label,
  value,
  format,
  sublabel,
  trend,
  accent = "#06b6d4",
  className,
}: {
  icon: ReactNode;
  label: string;
  value: number;
  format?: (v: number) => string;
  sublabel?: string;
  /** trend hint value; sign controls arrow direction */
  trend?: number;
  trendFormat?: (v: number) => string;
  accent?: string;
  className?: string;
}) {
  const display = useCountUp(value);
  const trendUp = (trend ?? 0) >= 0;
  return (
    <Card
      className={cn(
        "group gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors duration-200 hover:border-primary/30",
        className
      )}
    >
      <CardContent className="space-y-1.5 p-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            {label}
          </span>
          <span
            className="flex h-6 w-6 items-center justify-center rounded-md"
            style={{ color: accent, backgroundColor: `${accent}14` }}
            aria-hidden
          >
            {icon}
          </span>
        </div>
        <div className="flex items-baseline gap-2">
          <span
            className="font-mono text-2xl font-bold tabular-nums leading-none text-foreground"
          >
            {format ? format(display) : Math.round(display).toLocaleString("en-US")}
          </span>
          {trend !== undefined && (
            <span
              className={cn(
                "inline-flex items-center gap-0.5 text-[11px] font-semibold",
                trendUp ? "text-rose-400" : "text-emerald-400"
              )}
            >
              {trendUp ? (
                <TrendingUp className="h-3 w-3" aria-hidden />
              ) : (
                <TrendingDown className="h-3 w-3" aria-hidden />
              )}
              {Math.abs(trend).toFixed(1)}
            </span>
          )}
        </div>
        {sublabel && <p className="truncate text-[11px] text-muted-foreground">{sublabel}</p>}
      </CardContent>
    </Card>
  );
}

export function KpiChip({
  label,
  value,
  icon,
  className,
}: {
  label: string;
  value: string;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-lg border border-border/50 bg-background/40 px-3 py-2",
        className
      )}
    >
      <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        {icon}
        {label}
      </span>
      <span className="font-mono text-sm font-bold tabular-nums text-foreground">{value}</span>
    </div>
  );
}
