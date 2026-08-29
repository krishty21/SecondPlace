"use client";

import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { socApi } from "@/lib/soc-api";
import { cn } from "@/lib/utils";

/**
 * Engine health indicator — polls /api/health every 30s.
 * Shows a green pulsing dot + "ENGINE ONLINE" when ok.
 */
export function useEngineHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => socApi.health(signal),
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
    retry: 1,
    staleTime: 15_000,
  });
}

export function EngineStatus({ compact = false }: { compact?: boolean }) {
  const { data, isPending, isError } = useEngineHealth();
  const online = !isError && data?.status === "ok";

  if (isPending) {
    return (
      <Badge variant="outline" className="border-border/50 bg-card/40 font-mono text-[10px] text-muted-foreground">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" aria-hidden />
        ENGINE CONNECTING
      </Badge>
    );
  }

  if (!online) {
    return (
      <Badge variant="outline" className="border-amber-500/30 bg-amber-500/10 font-mono text-[10px] text-amber-400">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-400" aria-hidden />
        ENGINE OFFLINE
      </Badge>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Badge
        variant="outline"
        className="border-emerald-500/30 bg-emerald-500/10 font-mono text-[10px] text-emerald-400"
        aria-label={`Engine online, ${data.incidentsTracked} incidents tracked, uptime ${data.uptimeSec} seconds`}
      >
        <span className="relative flex h-1.5 w-1.5" aria-hidden>
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
        </span>
        ENGINE ONLINE
      </Badge>
      {!compact && (
        <Badge variant="outline" className="border-border/50 bg-card/40 font-mono text-[10px] text-muted-foreground">
          MODEL v{data.version}
        </Badge>
      )}
    </div>
  );
}

/** Banner shown at the top of a view while the engine is unreachable. */
export function EngineConnectingBanner({ className }: { className?: string }) {
  return (
    <div
      role="status"
      className={cn(
        "flex items-center gap-2.5 rounded-xl border border-amber-500/25 bg-amber-500/8 px-4 py-2.5 text-xs text-amber-200/90",
        className
      )}
    >
      <span className="relative flex h-2 w-2 shrink-0" aria-hidden>
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
      </span>
      <span>
        <span className="font-semibold">Engine connecting…</span> The inference engine on port 3010 is not responding
        yet (it may still be loading model artifacts). Data views will retry automatically.
      </span>
    </div>
  );
}
