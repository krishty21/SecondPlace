"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { io, type Socket } from "socket.io-client";
import {
  ChevronDown,
  ChevronRight,
  CircleStop,
  FastForward,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  Signal,
  Zap,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { socApi, type Incident, type ReplayStats, type ReplayTick, type ScoredEvent } from "@/lib/soc-api";
import { fmtInt, fmtMs, fmtRisk, riskColor } from "@/lib/soc-ui";
import {
  CategoryChip,
  EmptyState,
  EpistemicsBadge,
  SectionHeader,
  SeverityBadge,
  SimulatedTag,
  StatusBadge,
} from "@/components/soc/primitives";
import { Sparkline } from "@/components/soc/risk-gauge";
import { EventDetailDialog, EventRow } from "@/components/soc/event-detail";
import { cn } from "@/lib/utils";

const SPEEDS = [1, 2, 4, 8] as const;
const MAX_FEED = 200;

export function LiveReplay({ onOpenIncident }: { onOpenIncident: (incident: Incident) => void }) {
  // session state
  const [replayId, setReplayId] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [starting, setStarting] = useState(false);
  const [mode, setMode] = useState<"idle" | "socket" | "rest">("idle");

  // stream state
  const [feed, setFeed] = useState<ScoredEvent[]>([]);
  const [incidentsMap, setIncidentsMap] = useState<Map<string, Incident>>(new Map());
  const [stats, setStats] = useState<ReplayStats | null>(null);
  const [virtualTime, setVirtualTime] = useState(0);
  const [done, setDone] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [expandedIncident, setExpandedIncident] = useState<string | null>(null);
  const [detailEvent, setDetailEvent] = useState<ScoredEvent | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  // refs
  const socketRef = useRef<Socket | null>(null);
  const replayIdRef = useRef<string | null>(null);
  const cursorRef = useRef(0);
  const receivedTickRef = useRef(false);
  const fallbackActiveRef = useRef(false);
  const doneRef = useRef(false);
  const failCheckTimer = useRef<number | null>(null);
  const pollTimer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (failCheckTimer.current) window.clearTimeout(failCheckTimer.current);
      if (pollTimer.current) window.clearInterval(pollTimer.current);
      socketRef.current?.disconnect();
      socketRef.current = null;
    };
  }, []);

  const stopRestFallback = useCallback(() => {
    if (pollTimer.current) {
      window.clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  const processTick = useCallback(
    (tick: ReplayTick) => {
      if (tick.replayId !== replayIdRef.current) return;
      setVirtualTime(tick.virtualTime);
      setStats(tick.stats);
      if (tick.done) {
        doneRef.current = true;
        setDone(true);
        setPlaying(false);
        stopRestFallback();
      }
      if (tick.events?.length) {
        setFeed((prev) => [...tick.events.slice().reverse(), ...prev].slice(0, MAX_FEED));
      }
      if (tick.incidents?.length) {
        setIncidentsMap((prev) => {
          const next = new Map(prev);
          for (const inc of tick.incidents) next.set(inc.incidentId, inc);
          return next;
        });
      }
    },
    [stopRestFallback]
  );

  const startRestFallback = useCallback(() => {
    if (fallbackActiveRef.current) return;
    fallbackActiveRef.current = true;
    setMode("rest");
    stopRestFallback();
    pollTimer.current = window.setInterval(async () => {
      const rid = replayIdRef.current;
      if (!rid || doneRef.current) {
        stopRestFallback();
        return;
      }
      try {
        const snap = await socApi.replayState(rid, cursorRef.current);
        receivedTickRef.current = true;
        cursorRef.current = snap.cursor ?? cursorRef.current;
        processTick(snap);
      } catch {
        /* transient poll failure — keep trying until interval cleared */
      }
    }, 1000);
  }, [processTick, stopRestFallback]);

  const ensureSocket = useCallback((): Socket => {
    if (socketRef.current) return socketRef.current;
    // Gateway mode (sandbox/preview): relative path + XTransformPort=3010.
    // Local mode: NEXT_PUBLIC_ENGINE_URL points straight at the engine
    // (the engine enables CORS for the socket.io handshake).
    const engineBase = (process.env.NEXT_PUBLIC_ENGINE_URL ?? "").replace(/\/+$/, "");
    const sock = io(engineBase ? engineBase : "/?XTransformPort=3010", {
      transports: ["websocket", "polling"],
      reconnectionAttempts: 5,
      timeout: 8000,
    });
    socketRef.current = sock;
    sock.on("connect", () => {
      const rid = replayIdRef.current;
      if (rid) sock.emit("replay:join", { replayId: rid });
      if (!fallbackActiveRef.current) setMode("socket");
    });
    sock.on("connect_error", () => {
      if (fallbackActiveRef.current) return;
      // handled by the 8s check timer below; if it already elapsed, switch now
      if (receivedTickRef.current === false && failCheckTimer.current === null) {
        startRestFallback();
      }
    });
    sock.on("disconnect", () => {
      // mid-replay socket loss: degrade to REST polling so the stream keeps flowing
      if (replayIdRef.current && !fallbackActiveRef.current && receivedTickRef.current && !doneRef.current) {
        startRestFallback();
      }
    });
    sock.on("replay:tick", (tick: ReplayTick) => {
      if (fallbackActiveRef.current) return;
      receivedTickRef.current = true;
      processTick(tick);
    });
    return sock;
  }, [processTick, startRestFallback]);

  const startReplay = async () => {
    if (starting || replayId) return;
    setStarting(true);
    setStartError(null);
    try {
      const res = await socApi.replayStart(speed);
      replayIdRef.current = res.replayId;
      cursorRef.current = 0;
      receivedTickRef.current = false;
      fallbackActiveRef.current = false;
      doneRef.current = false;
      setReplayId(res.replayId);
      setTotal(res.total);
      setDurationMs(res.durationMs);
      setFeed([]);
      setIncidentsMap(new Map());
      setStats(null);
      setVirtualTime(0);
      setDone(false);
      setExpandedIncident(null);

      const sock = ensureSocket();
      if (sock.connected) {
        sock.emit("replay:join", { replayId: res.replayId });
        setMode("socket");
      }
      // fallback detector: no tick within 8s -> REST polling
      if (failCheckTimer.current) window.clearTimeout(failCheckTimer.current);
      failCheckTimer.current = window.setTimeout(() => {
        failCheckTimer.current = null;
        if (!receivedTickRef.current) startRestFallback();
      }, 8000);

      // kick off playback
      setPlaying(true);
      if (sock.connected) {
        sock.emit("replay:control", { replayId: res.replayId, action: "play" });
      } else {
        void socApi.replayControl(res.replayId, "play");
      }
    } catch (err) {
      setStartError(
        err instanceof Error ? `Could not start replay: ${err.message}` : "Could not start replay — engine unreachable."
      );
      setStarting(false);
      return;
    }
    setStarting(false);
  };

  const dispatchControl = (action: "play" | "pause" | "speed" | "stop", value?: number) => {
    const rid = replayIdRef.current;
    if (!rid) return;
    if (mode === "socket" && !fallbackActiveRef.current && socketRef.current?.connected) {
      socketRef.current.emit("replay:control", { replayId: rid, action, value });
    } else {
      void socApi.replayControl(rid, action, value);
    }
  };

  const togglePlay = () => {
    if (done || !replayId) return;
    if (playing) {
      setPlaying(false);
      dispatchControl("pause");
    } else {
      setPlaying(true);
      dispatchControl("play");
    }
  };

  const changeSpeed = (s: number) => {
    setSpeed(s);
    dispatchControl("speed", s);
  };

  const resetSession = () => {
    if (pollTimer.current) window.clearInterval(pollTimer.current);
    if (failCheckTimer.current) window.clearTimeout(failCheckTimer.current);
    failCheckTimer.current = null;
    fallbackActiveRef.current = false;
    receivedTickRef.current = false;
    doneRef.current = false;
    cursorRef.current = 0;
    replayIdRef.current = null;
    setReplayId(null);
    setMode("idle");
    setFeed([]);
    setIncidentsMap(new Map());
    setStats(null);
    setVirtualTime(0);
    setDone(false);
    setPlaying(false);
    setExpandedIncident(null);
  };

  const incidents = useMemo(
    () => [...incidentsMap.values()].sort((a, b) => b.lastSeen - a.lastSeen || b.riskScore - a.riskScore),
    [incidentsMap]
  );
  const finalIncidents = useMemo(
    () => [...incidentsMap.values()].sort((a, b) => b.riskScore - a.riskScore),
    [incidentsMap]
  );

  const progress = stats && stats.total > 0 ? Math.min(100, (stats.processed / stats.total) * 100) : 0;

  return (
    <div className="space-y-5">
      {/* banner */}
      <div
        role="note"
        className="flex items-start gap-2.5 rounded-xl border border-rose-500/25 bg-rose-500/8 px-4 py-3 text-xs leading-relaxed text-foreground/85"
      >
        <Radio className="mt-0.5 h-4 w-4 shrink-0 animate-pulse text-rose-400" aria-hidden />
        <p>
          <span className="font-semibold text-rose-300">REAL-TIME DETECTION REPLAY</span> — replaying{" "}
          <span className="font-semibold">REAL UNSW-NB15 test events</span> through the trained models. Event
          timestamps &amp; entities are <span className="font-semibold text-[#a78bfa]">SIMULATED</span>. This is NOT a
          live production network feed.
        </p>
      </div>

      {/* control bar */}
      <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4">
        {startError && (
          <p role="alert" className="mb-3 rounded-lg border border-rose-500/25 bg-rose-500/8 px-3 py-2 text-xs text-rose-300">
            {startError}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-3">
          {!replayId ? (
            <Button
              onClick={() => void startReplay()}
              disabled={starting}
              className="h-10 gap-2 bg-primary/90 px-5 font-semibold text-primary-foreground hover:bg-primary"
            >
              {starting ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" aria-hidden />
                  Starting engine…
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" aria-hidden />
                  Start Replay
                </>
              )}
            </Button>
          ) : (
            <>
              <Button
                onClick={togglePlay}
                disabled={done}
                aria-label={playing ? "Pause replay" : "Resume replay"}
                className={cn(
                  "h-10 gap-2 font-semibold",
                  playing
                    ? "bg-amber-500/90 text-black hover:bg-amber-500"
                    : "bg-emerald-500/90 text-black hover:bg-emerald-500",
                  done && "opacity-50"
                )}
              >
                {playing ? <Pause className="h-4 w-4" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
                {playing ? "Pause" : done ? "Done" : "Resume"}
              </Button>
              <div className="flex items-center gap-1" role="group" aria-label="replay speed">
                <FastForward className="mr-1 h-3.5 w-3.5 text-muted-foreground" aria-hidden />
                {SPEEDS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => changeSpeed(s)}
                    aria-pressed={speed === s}
                    disabled={done}
                    className={cn(
                      "rounded-md border px-2 py-1 font-mono text-[11px] transition-colors",
                      speed === s
                        ? "border-primary/50 bg-primary/15 text-primary"
                        : "border-border/50 text-muted-foreground hover:border-primary/30 hover:text-foreground"
                    )}
                  >
                    {s}x
                  </button>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={resetSession}
                className="h-9 gap-1.5 border-border/60 text-xs text-muted-foreground"
              >
                <RotateCcw className="h-3.5 w-3.5" aria-hidden />
                New replay
              </Button>
            </>
          )}

          {replayId && (
            <div className="ml-auto flex flex-wrap items-center gap-3 font-mono text-[11px] text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <Signal className="h-3.5 w-3.5" aria-hidden />
                {mode === "socket" ? "socket.io live" : mode === "rest" ? "REST polling fallback" : "connecting…"}
              </span>
              <span>
                clock <span className="font-semibold text-primary">t+{fmtMs(virtualTime)}</span>
              </span>
              <span>
                throughput{" "}
                <span className="font-semibold text-foreground/85">{stats ? `${stats.throughput.toFixed(1)} ev/s` : "—"}</span>
              </span>
              <span className="text-muted-foreground/70">{replayId}</span>
            </div>
          )}
        </div>

        {/* progress */}
        {replayId && (
          <div className="mt-3 space-y-1" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} aria-label="replay progress">
            <div className="h-2 w-full overflow-hidden rounded-full bg-white/5">
              <div
                className={cn("h-full rounded-full transition-[width] duration-300", done ? "bg-emerald-500" : "bg-primary")}
                style={{ width: `${progress}%`, boxShadow: done ? "0 0 10px #10b98155" : "0 0 10px #06b6d444" }}
              />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>
                {stats ? fmtInt(stats.processed) : 0} / {fmtInt(total)} events
                {durationMs > 0 && ` · ~${fmtMs(durationMs)} virtual`}
              </span>
              <span>{progress.toFixed(0)}%</span>
            </div>
          </div>
        )}
      </Card>

      {/* done state */}
      <AnimatePresence>
        {done && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <Card className="gap-0 rounded-xl border-emerald-500/25 bg-emerald-500/5 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-400">
                    <Zap className="h-4 w-4" aria-hidden />
                  </span>
                  <div>
                    <p className="text-sm font-semibold text-foreground">Replay complete</p>
                    <p className="font-mono text-[11px] text-muted-foreground">
                      {stats
                        ? `${fmtInt(stats.processed)} events · ${fmtInt(stats.alerts)} alerts · ${stats.incidents} incidents (${stats.criticalIncidents} critical) · risk max ${fmtRisk(stats.riskMax)}`
                        : ""}
                    </p>
                  </div>
                </div>
                {finalIncidents[0] && (
                  <Button
                    size="sm"
                    onClick={() => onOpenIncident(finalIncidents[0])}
                    className="h-9 gap-1.5 bg-primary/90 text-primary-foreground hover:bg-primary"
                  >
                    <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                    Analyze final incidents
                  </Button>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* stats strip */}
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-8" aria-label="live replay statistics">
          <StatCell label="Processed" value={`${fmtInt(stats.processed)}/${fmtInt(stats.total)}`} />
          <StatCell label="Alerts" value={fmtInt(stats.alerts)} accent="#f43f5e" />
          <StatCell label="Normals" value={fmtInt(stats.normals)} accent="#10b981" />
          <StatCell label="Incidents" value={String(stats.incidents)} accent="#f97316" />
          <StatCell label="Critical" value={String(stats.criticalIncidents)} accent="#f43f5e" />
          <StatCell label="Risk max" value={fmtRisk(stats.riskMax)} accent={riskColor(stats.riskMax)} />
          <StatCell label="Throughput" value={`${stats.throughput.toFixed(1)} ev/s`} />
          <div className="rounded-xl border border-border/50 bg-card/50 px-3 py-2">
            <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">By category</p>
            <div className="mt-1 space-y-0.5">
              {Object.entries(stats.byCategory)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([cat, n]) => {
                  const max = Math.max(...Object.values(stats.byCategory));
                  return (
                    <div key={cat} className="flex items-center gap-1.5">
                      <span className="w-16 shrink-0 truncate text-[9px] text-muted-foreground">{cat}</span>
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/5">
                        <div className="h-full rounded-full bg-primary/70" style={{ width: `${(n / max) * 100}%` }} />
                      </div>
                      <span className="font-mono text-[9px] tabular-nums text-muted-foreground">{n}</span>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      )}

      {/* feed + incidents */}
      <div className="grid gap-4 lg:grid-cols-5">
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-3">
          <SectionHeader
            title="Live event stream"
            hint="newest first · normal traffic dimmed · alerts prominent"
            icon={<Radio className="h-4 w-4" aria-hidden />}
            action={
              replayId ? (
                <span className="font-mono text-[10px] text-muted-foreground">{feed.length} in view</span>
              ) : undefined
            }
          />
          {!replayId ? (
            <EmptyState
              className="min-h-72"
              icon={<CircleStop className="h-7 w-7" aria-hidden />}
              title="Replay not started"
              description="Start the replay to stream real UNSW-NB15 test events through the live detection engine — waves escalate from background noise to recon, exploit and DoS campaigns."
            />
          ) : (
            <>
              <div className="mb-1 flex items-center gap-2 px-2 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                <span className="w-[74px]">event</span>
                <span className="w-24">category</span>
                <span className="w-12 text-right">risk</span>
                <span className="w-12 text-right">anom</span>
                <span className="w-12 text-right">conf</span>
                <span className="w-20">entity</span>
                <span className="hidden sm:inline">wave</span>
              </div>
              <div className="soc-scroll max-h-[520px] overflow-y-auto pr-1" role="feed" aria-label="live event feed">
                <AnimatePresence initial={false}>
                  {feed.map((e) => (
                    <motion.div
                      key={e.eventId}
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.22 }}
                    >
                      <EventRow event={e} onClick={setDetailEvent} dimmed={e.binaryVerdict !== "Attack"} />
                    </motion.div>
                  ))}
                </AnimatePresence>
                {feed.length === 0 && !done && (
                  <p className="py-8 text-center font-mono text-[11px] text-muted-foreground">
                    waiting for first tick…
                  </p>
                )}
              </div>
            </>
          )}
        </Card>

        {/* live incidents */}
        <Card className="gap-0 rounded-xl border-border/50 bg-card/50 p-4 transition-colors hover:border-primary/30 lg:col-span-2">
          <SectionHeader
            title="Live incidents"
            hint="correlated in-session · click to expand"
            icon={<ShieldAlert className="h-4 w-4" aria-hidden />}
            action={
              incidents.length > 0 ? (
                <span className="font-mono text-[10px] text-muted-foreground">{incidents.length} tracked</span>
              ) : undefined
            }
          />
          <div className="soc-scroll max-h-[560px] space-y-2 overflow-y-auto pr-1">
            {incidents.length === 0 && (
              <p className="py-8 text-center text-xs text-muted-foreground">
                {replayId ? "No correlated incidents yet…" : "Start a replay to see incidents form live."}
              </p>
            )}
            {incidents.map((inc) => (
              <LiveIncidentCard
                key={inc.incidentId}
                incident={inc}
                expanded={expandedIncident === inc.incidentId}
                onToggle={() => setExpandedIncident(expandedIncident === inc.incidentId ? null : inc.incidentId)}
                onOpenFull={() => onOpenIncident(inc)}
              />
            ))}
          </div>
        </Card>
      </div>

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

function LiveIncidentCard({
  incident,
  expanded,
  onToggle,
  onOpenFull,
}: {
  incident: Incident;
  expanded: boolean;
  onToggle: () => void;
  onOpenFull: () => void;
}) {
  const trajectory = (incident.riskTrajectory ?? []).map((p) => p.risk);
  return (
    <div
      className={cn(
        "rounded-lg border bg-background/30 transition-colors",
        expanded ? "border-primary/40" : "border-border/40 hover:border-primary/25"
      )}
    >
      <button type="button" onClick={onToggle} className="w-full px-3 py-2.5 text-left" aria-expanded={expanded}>
        <div className="flex items-start gap-2">
          <span
            className="mt-0.5 rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-bold tabular-nums"
            style={{
              color: riskColor(incident.riskScore),
              backgroundColor: `${riskColor(incident.riskScore)}12`,
              borderColor: `${riskColor(incident.riskScore)}35`,
            }}
          >
            {fmtRisk(incident.riskScore)}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-[11px] font-semibold text-primary">{incident.incidentId}</span>
              <SeverityBadge severity={incident.severity} />
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[10px] text-muted-foreground">
              <CategoryChip category={incident.category} />
              <StatusBadge status={incident.status} />
            </div>
            <div className="mt-1 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
              <span>{incident.alertCount} alerts</span>
              <span>t+{fmtMs(incident.lastSeen)}</span>
              <SimulatedTag />
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            {trajectory.length > 1 && <Sparkline points={trajectory} width={64} height={20} />}
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
            )}
          </div>
        </div>
      </button>
      {expanded && (
        <div className="space-y-2 border-t border-border/40 px-3 py-2.5">
          {(incident.story ?? []).slice(0, 3).map((stage) => (
            <div key={stage.index} className="rounded-md border border-border/40 bg-background/40 px-2.5 py-1.5">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-[9px] font-bold text-primary">{stage.index}</span>
                <span className="text-[11px] font-medium text-foreground/90">{stage.title}</span>
                <EpistemicsBadge kind={stage.epistemics} />
              </div>
              <p className="mt-0.5 line-clamp-2 text-[10px] leading-relaxed text-muted-foreground">{stage.detail}</p>
            </div>
          ))}
          {(incident.story ?? []).length === 0 && (
            <p className="text-[10px] text-muted-foreground">
              Attack story composes when the incident finalizes — keep watching.
            </p>
          )}
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[10px] text-muted-foreground">
              top: {(incident.topContributors ?? []).slice(0, 2).map((c) => c.feature).join(", ") || "—"}
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={onOpenFull}
              className="h-7 gap-1 border-primary/30 text-[10px] text-primary hover:bg-primary/10"
            >
              Open full investigation
              <ChevronRight className="h-3 w-3" aria-hidden />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCell({ label, value, accent = "#06b6d4" }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-xl border border-border/50 bg-card/50 px-3 py-2">
      <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-0.5 font-mono text-sm font-bold tabular-nums" style={{ color: accent }}>
        {value}
      </p>
    </div>
  );
}
