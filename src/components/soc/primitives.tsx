"use client";

import { useState, type ReactNode } from "react";
import { AlertTriangle, Check, Copy, Inbox, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Epistemics, IncidentStatus, Severity } from "@/lib/soc-api";
import { EPISTEMICS_META, STATUS_META, severityMeta, waveColor, waveLabel } from "@/lib/soc-ui";
import { cn } from "@/lib/utils";

// ------------------------------------------------------------------ badges

export function SeverityBadge({
  severity,
  className,
  dot = false,
}: {
  severity: string;
  className?: string;
  dot?: boolean;
}) {
  const meta = severityMeta(severity);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold tracking-wide",
        className
      )}
      style={{
        color: meta.hex,
        backgroundColor: `${meta.hex}14`,
        borderColor: `${meta.hex}38`,
      }}
    >
      {dot && (
        <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.hex }} aria-hidden />
      )}
      {meta.label}
    </span>
  );
}

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const meta = STATUS_META[status as IncidentStatus] ?? { hex: "#8b98a9" };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold tracking-wide",
        className
      )}
      style={{
        color: meta.hex,
        backgroundColor: `${meta.hex}14`,
        borderColor: `${meta.hex}38`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.hex }} aria-hidden />
      {status}
    </span>
  );
}

export function EpistemicsBadge({ kind, className }: { kind: string; className?: string }) {
  const meta = EPISTEMICS_META[kind as Epistemics] ?? { hex: "#8b98a9", hint: "" };
  const badge = (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
        className
      )}
      style={{ color: meta.hex, backgroundColor: `${meta.hex}12`, borderColor: `${meta.hex}30` }}
    >
      {kind}
    </span>
  );
  if (!meta.hint) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent className="max-w-56">{meta.hint}</TooltipContent>
    </Tooltip>
  );
}

/** Small violet "SIM" tag marking simulated replay metadata. */
export function SimulatedTag({ className }: { className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-flex cursor-help items-center rounded border border-[#8b5cf6]/30 bg-[#8b5cf6]/10 px-1 py-px font-mono text-[9px] font-bold uppercase tracking-widest text-[#a78bfa]",
            className
          )}
        >
          sim
        </span>
      </TooltipTrigger>
      <TooltipContent>Simulated metadata — the UNSW-NB15 dataset has no timestamps/entities.</TooltipContent>
    </Tooltip>
  );
}

export function WaveTag({ wave, className }: { wave: string | undefined; className?: string }) {
  if (!wave) return null;
  const hex = waveColor(wave);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-px font-mono text-[9px] font-semibold uppercase tracking-wider",
        className
      )}
      style={{ color: hex, backgroundColor: `${hex}12`, borderColor: `${hex}30` }}
      title={`replay wave: ${wave}`}
    >
      {waveLabel(wave)}
    </span>
  );
}

export function CategoryChip({
  category,
  count,
  hex = "#06b6d4",
  className,
}: {
  category: string;
  count?: number;
  hex?: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium",
        className
      )}
      style={{ color: hex, backgroundColor: `${hex}10`, borderColor: `${hex}28` }}
    >
      {category}
      {count !== undefined && <span className="font-mono opacity-70">{count}</span>}
    </span>
  );
}

// ---------------------------------------------------------------- headers

export function SectionHeader({
  title,
  hint,
  icon,
  action,
  className,
}: {
  title: string;
  hint?: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-3 flex flex-wrap items-center justify-between gap-2", className)}>
      <div className="flex items-center gap-2">
        {icon && <span className="text-primary">{icon}</span>}
        <h2 className="text-sm font-semibold tracking-tight text-foreground">{title}</h2>
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </div>
      {action}
    </div>
  );
}

export function CopyableId({
  id,
  label,
  className,
}: {
  id: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable — silently ignore */
    }
  };
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={`Copy ${label ?? "identifier"} ${id}`}
      className={cn(
        "group inline-flex items-center gap-1 font-mono text-xs text-foreground/90 hover:text-primary",
        className
      )}
    >
      {id}
      {copied ? (
        <Check className="h-3 w-3 text-emerald-400" aria-hidden />
      ) : (
        <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-60" aria-hidden />
      )}
    </button>
  );
}

// ------------------------------------------------------------ state blocks

export function EmptyState({
  title,
  description,
  icon,
  action,
  className,
}: {
  title: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border/60 px-6 py-10 text-center",
        className
      )}
    >
      <div className="text-muted-foreground/60">{icon ?? <Inbox className="h-6 w-6" aria-hidden />}</div>
      <p className="text-sm font-medium text-foreground/90">{title}</p>
      {description && <p className="max-w-sm text-xs text-muted-foreground">{description}</p>}
      {action}
    </div>
  );
}

export function ErrorState({
  title = "Could not load data",
  description,
  onRetry,
  className,
}: {
  title?: string;
  description?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-xl border border-rose-500/20 bg-rose-500/5 px-6 py-8 text-center",
        className
      )}
    >
      <AlertTriangle className="h-5 w-5 text-rose-400" aria-hidden />
      <div>
        <p className="text-sm font-medium text-foreground/90">{title}</p>
        {description && <p className="mt-1 max-w-sm text-xs text-muted-foreground">{description}</p>}
      </div>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry} className="h-8 gap-1.5">
          <RefreshCw className="h-3.5 w-3.5" aria-hidden />
          Retry
        </Button>
      )}
    </div>
  );
}

export function ChartSkeleton({ className }: { className?: string }) {
  return <Skeleton className={cn("h-full min-h-48 w-full rounded-lg", className)} />;
}

// ------------------------------------------------------------- chart tooltip

export interface TooltipEntry {
  name?: string | number;
  value?: number | string;
  color?: string;
  dataKey?: string | number;
  payload?: Record<string, unknown>;
}

export function ChartTooltip({
  active,
  entries,
  label,
  renderEntry,
}: {
  active?: boolean;
  entries?: TooltipEntry[];
  label?: string;
  renderEntry?: (entry: TooltipEntry) => ReactNode;
}) {
  if (!active || !entries || entries.length === 0) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-[#0d121a]/95 px-3 py-2 text-xs shadow-xl backdrop-blur-sm">
      {label !== undefined && (
        <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      )}
      <div className="space-y-1">
        {entries.map((entry, i) => (
          <div key={i} className="flex items-center justify-between gap-4">
            {renderEntry ? (
              renderEntry(entry)
            ) : (
              <>
                <span className="flex items-center gap-1.5 text-muted-foreground">
                  <span
                    className="h-2 w-2 rounded-sm"
                    style={{ backgroundColor: entry.color ?? "#06b6d4" }}
                    aria-hidden
                  />
                  {String(entry.name ?? "")}
                </span>
                <span className="font-mono text-foreground">{String(entry.value ?? "—")}</span>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
