"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { riskColor } from "@/lib/soc-ui";
import { cn } from "@/lib/utils";

/** Animated count-up helper (cubic ease-out). */
export function useCountUp(value: number, duration = 700): number {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  useEffect(() => {
    const from = fromRef.current;
    if (from === value) return;
    let raf = 0;
    const start = performance.now();
    const step = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay(from + (value - from) * eased);
      if (p < 1) {
        raf = requestAnimationFrame(step);
      } else {
        fromRef.current = value;
      }
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  // keep ref in sync when value settles
  useEffect(() => {
    fromRef.current = display;
  }, [display]);
  return display;
}

/**
 * Animated radial risk gauge (0-100), colored by severity band.
 */
export function RiskGauge({
  value,
  size = 120,
  thickness = 9,
  label = "risk",
  className,
  animate = true,
}: {
  value: number;
  size?: number;
  thickness?: number;
  label?: string;
  className?: string;
  animate?: boolean;
}) {
  const clamped = Math.max(0, Math.min(100, value));
  const display = useCountUp(clamped, animate ? 900 : 0);
  const color = riskColor(clamped);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - display / 100);

  return (
    <div
      className={cn("relative inline-flex items-center justify-center", className)}
      role="img"
      aria-label={`${label} score ${clamped.toFixed(1)} of 100`}
    >
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={thickness}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={thickness}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{ filter: `drop-shadow(0 0 6px ${color}55)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-bold tabular-nums" style={{ color }}>
          {clamped.toFixed(1)}
        </span>
        <span className="text-[9px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          {label}
        </span>
      </div>
    </div>
  );
}

/** Small inline risk score pill (used in rows). */
export function RiskPill({ value, className }: { value: number; className?: string }) {
  const color = riskColor(value);
  return (
    <span
      className={cn(
        "inline-flex min-w-14 items-center justify-center rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-bold tabular-nums",
        className
      )}
      style={{ color, backgroundColor: `${color}12`, borderColor: `${color}35` }}
    >
      {value.toFixed(1)}
    </span>
  );
}

/** Tiny SVG sparkline for risk trajectories (no recharts overhead). */
export function Sparkline({
  points,
  width = 96,
  height = 26,
  className,
}: {
  points: number[];
  width?: number;
  height?: number;
  className?: string;
}) {
  if (points.length < 2) {
    return <span className="text-[10px] text-muted-foreground">—</span>;
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const stepX = width / (points.length - 1);
  const coords = points.map((p, i) => {
    const x = i * stepX;
    const y = height - 3 - ((p - min) / span) * (height - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const color = riskColor(points[points.length - 1]);
  const last = coords[coords.length - 1].split(",");
  return (
    <svg width={width} height={height} className={className} aria-hidden>
      <polyline
        points={coords.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1.4}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.9}
      />
      <circle cx={last[0]} cy={last[1]} r={2} fill={color} />
    </svg>
  );
}

export function AnimatedNumber({
  value,
  format,
  className,
}: {
  value: number;
  format?: (v: number) => string;
  className?: string;
}) {
  const display = useCountUp(value);
  return (
    <span className={cn("font-mono tabular-nums", className)}>
      {format ? format(display) : Math.round(display).toLocaleString("en-US")}
    </span>
  );
}

export const MotionDiv = motion.div;
