"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useQueryClient } from "@tanstack/react-query";
import { LayoutDashboard, Microscope, Radar, Radio, Shield, ShieldAlert } from "lucide-react";
import type { Incident } from "@/lib/soc-api";
import { EngineStatus, useEngineHealth, EngineConnectingBanner } from "@/components/soc/engine-status";
import { CommandCenter } from "@/components/soc/views/command-center";
import { IncidentInvestigation, type IncidentFocus } from "@/components/soc/views/incident-investigation";
import { PatternExplorer } from "@/components/soc/views/pattern-explorer";
import { ExplainabilityCenter } from "@/components/soc/views/explainability-center";
import { LiveReplay } from "@/components/soc/views/live-replay";
import { cn } from "@/lib/utils";

type ViewId = "command" | "incidents" | "patterns" | "explain" | "replay";

const VIEWS: { id: ViewId; label: string; icon: React.ReactNode; short: string }[] = [
  { id: "command", label: "Command Center", short: "Command", icon: <LayoutDashboard className="h-3.5 w-3.5" aria-hidden /> },
  { id: "incidents", label: "Incidents", short: "Incidents", icon: <ShieldAlert className="h-3.5 w-3.5" aria-hidden /> },
  { id: "patterns", label: "Pattern Explorer", short: "Patterns", icon: <Radar className="h-3.5 w-3.5" aria-hidden /> },
  { id: "explain", label: "Explainability", short: "Explain", icon: <Microscope className="h-3.5 w-3.5" aria-hidden /> },
  { id: "replay", label: "Live Replay", short: "Replay", icon: <Radio className="h-3.5 w-3.5" aria-hidden /> },
];

export function SocApp() {
  const [view, setView] = useState<ViewId>("command");
  const [focusIncident, setFocusIncident] = useState<IncidentFocus | null>(null);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const queryClient = useQueryClient();
  const { isError, isSuccess } = useEngineHealth();
  const engineWasDown = useRef(false);

  // auto-recovery: when the engine comes back online, refetch everything
  useEffect(() => {
    if (isError) engineWasDown.current = true;
    if (engineWasDown.current && isSuccess) {
      engineWasDown.current = false;
      void queryClient.invalidateQueries();
    }
  }, [isError, isSuccess, queryClient]);

  // while the engine is down, poll health faster (15s) so recovery is quick
  useEffect(() => {
    if (!isError) return;
    const id = window.setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: ["health"] });
    }, 15_000);
    return () => window.clearInterval(id);
  }, [isError, queryClient]);

  const openIncident = useCallback((incident: Incident) => {
    // replay incidents are session-scoped: carry the live object as fallback data
    setFocusIncident({ incidentId: incident.incidentId, fallback: incident });
    setView("incidents");
  }, []);

  const openIncidentFromBoot = useCallback((incident: Incident) => {
    setFocusIncident({ incidentId: incident.incidentId });
    setView("incidents");
  }, []);

  const handleTabKeys = (e: React.KeyboardEvent, index: number) => {
    let next = index;
    if (e.key === "ArrowRight") next = (index + 1) % VIEWS.length;
    else if (e.key === "ArrowLeft") next = (index - 1 + VIEWS.length) % VIEWS.length;
    else return;
    e.preventDefault();
    const target = tabRefs.current[next];
    target?.focus();
    target?.click();
  };

  return (
    <div className="soc-bg flex min-h-screen flex-col text-foreground">
      {/* ---------------------------------------------------------- header */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-4 px-4 lg:px-6">
          {/* logo */}
          <div className="flex shrink-0 items-center gap-2.5">
            <span className="relative flex h-8 w-8 items-center justify-center rounded-lg border border-primary/30 bg-primary/10">
              <Shield className="h-4.5 w-4.5 text-primary" aria-hidden />
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-primary/80" aria-hidden />
            </span>
            <div className="hidden flex-col leading-none sm:flex">
              <span className="text-sm font-bold tracking-tight">CipherMind Sentinel</span>
              <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-primary/80">
                AI SOC Copilot
              </span>
            </div>
          </div>

          {/* view tabs */}
          <nav
            role="tablist"
            aria-label="Primary views"
            className="no-scrollbar flex flex-1 items-center gap-1 overflow-x-auto"
          >
            {VIEWS.map((v, i) => (
              <button
                key={v.id}
                ref={(el) => {
                  tabRefs.current[i] = el;
                }}
                type="button"
                role="tab"
                aria-selected={view === v.id}
                aria-label={v.label}
                aria-controls={`panel-${v.id}`}
                id={`tab-${v.id}`}
                onClick={() => setView(v.id)}
                onKeyDown={(e) => handleTabKeys(e, i)}
                className={cn(
                  "flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-xs font-medium transition-all duration-200",
                  view === v.id
                    ? "border-primary/40 bg-primary/12 text-primary"
                    : "border-transparent text-muted-foreground hover:border-border/60 hover:bg-accent/40 hover:text-foreground/90"
                )}
              >
                {v.icon}
                <span className="hidden md:inline">{v.label}</span>
                <span className="md:hidden">{v.short}</span>
              </button>
            ))}
          </nav>

          {/* engine status */}
          <div className="flex shrink-0 items-center gap-2">
            <EngineStatus />
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------------- main */}
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-4 py-6 lg:px-6">
        <AnimatePresence>
          {isError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="mb-5 overflow-hidden"
            >
              <EngineConnectingBanner />
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            role="tabpanel"
            aria-labelledby={`tab-${view}`}
            id={`panel-${view}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
          >
            {view === "command" && <CommandCenter onOpenIncident={openIncidentFromBoot} />}
            {view === "incidents" && <IncidentInvestigation focus={focusIncident} />}
            {view === "patterns" && <PatternExplorer />}
            {view === "explain" && <ExplainabilityCenter />}
            {view === "replay" && <LiveReplay onOpenIncident={openIncident} />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ---------------------------------------------------------- footer */}
      <footer className="mt-auto border-t border-border/60 bg-background/70">
        <div className="mx-auto flex max-w-[1600px] flex-col gap-1 px-4 py-4 text-[11px] leading-relaxed text-muted-foreground sm:flex-row sm:items-center sm:justify-between lg:px-6">
          <p>
            <span className="font-semibold text-foreground/70">CipherMind Sentinel</span> — CipherMind AI &apos;26 ·
            UNSW-NB15
          </p>
          <p className="max-w-2xl sm:text-right">
            Model outputs are research prototypes; timestamps &amp; entity labels are SIMULATED replay metadata.
          </p>
        </div>
      </footer>
    </div>
  );
}
