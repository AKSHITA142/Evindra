"use client";

import { useEffect, useState, useMemo } from "react";
import { useReducedMotion } from "framer-motion";
import { ArrowRight, Layers3, Radio, ShieldAlert, Terminal, Waypoints } from "lucide-react";

interface TelemetryEvent {
  time: string;
  stage: string;
  event: string;
  detail: string;
  tone: string;
}

const events: TelemetryEvent[] = [
  {
    time: "14:02:01",
    stage: "UNDERSTAND",
    event: "profile.completed",
    detail: "18 features · 2 leakage ID cols removed",
    tone: "text-brand-300",
  },
  {
    time: "14:02:02",
    stage: "PLAN",
    event: "hypothesis.ranked",
    detail: "H-02 promoted: TargetEnc + CatBoost",
    tone: "text-info-400",
  },
  {
    time: "14:02:05",
    stage: "EXECUTE",
    event: "cv_fold.completed",
    detail: "5/5 folds: mean AUC 0.941 ± 0.011",
    tone: "text-warning-400",
  },
  {
    time: "14:02:06",
    stage: "EVALUATE",
    event: "scoring.composite",
    detail: "Rank #1 · 4-pillar score 0.924",
    tone: "text-success-400",
  },
  {
    time: "14:02:07",
    stage: "DIRECT",
    event: "state.knowledge_updated",
    detail: "Feature interaction promoted to memory",
    tone: "text-brand-300",
  },
  {
    time: "14:02:08",
    stage: "ROUTE",
    event: "router.decision",
    detail: "Delta +1.8% > 0.5% ➔ cycle 2 of 5 queued",
    tone: "text-info-400",
  },
  {
    time: "14:02:14",
    stage: "EXPORT",
    event: "pipeline.frozen",
    detail: "champion_model.joblib & report generated",
    tone: "text-success-400",
  },
];

const fallbackChain = [
  { name: "Gemini LLM", type: "Primary Reasoning" },
  { name: "OpenAI LLM", type: "High-Throughput Fallback" },
  { name: "OpenRouter Open-Source LLM", type: "Multi-Model Router" },
  { name: "Rule Engine", type: "Deterministic Fallback" },
];

export function TelemetryPanel() {
  const reducedMotion = useReducedMotion();
  const [visibleCount, setVisibleCount] = useState(reducedMotion ? events.length : 1);

  const wsEndpoint = useMemo(() => {
    const rawWsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000";
    const apiPrefix = process.env.NEXT_PUBLIC_API_PREFIX || "/api/v1";
    return `${rawWsUrl}${apiPrefix}/ws/research/stream`;
  }, []);

  useEffect(() => {
    if (reducedMotion) return;
    const timer = window.setInterval(() => {
      setVisibleCount((count) => (count >= events.length ? 1 : count + 1));
    }, 1600);
    return () => window.clearInterval(timer);
  }, [reducedMotion]);

  return (
    <section id="telemetry" className="scroll-mt-20 border-b border-border-subtle py-24 sm:py-32">
      <div className="mx-auto grid max-w-7xl gap-12 px-4 sm:px-6 lg:grid-cols-2 lg:items-center">
        
        {/* Left Side: Resiliency & Reasoning Description */}
        <div className="flex flex-col gap-6">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.24em] text-brand-300">
            <Radio className="size-3.5 animate-pulse-soft" /> Real-Time Telemetry &amp; Fault Tolerance
          </div>
          
          <h2 className="text-balance text-4xl font-semibold tracking-tight text-text sm:text-5xl">
            Live execution telemetry with automated multi-LLM fallback.
          </h2>
          
          <p className="text-pretty text-lg leading-relaxed text-text-secondary">
            Watch the agent formulate hypotheses, train folds, score evidence, and iterate in real-time over native WebSockets. If an upstream AI provider experiences rate limits or timeouts, Evidra automatically cascades down a zero-downtime reasoning chain.
          </p>

          {/* Fallback Chain Matrix */}
          <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface-2/60 p-4">
            <span className="font-mono text-xs font-semibold text-text">
              Resilient AI Reasoning Chain:
            </span>
            <div className="flex flex-wrap items-center gap-2 font-mono text-xs pt-1">
              {fallbackChain.map((provider, index) => (
                <div key={provider.name} className="flex items-center gap-2">
                  <div className="flex flex-col rounded-lg border border-border-subtle bg-surface-3 px-3 py-1.5 shadow-sm">
                    <span className="font-semibold text-text">{provider.name}</span>
                    <span className="text-[10px] text-text-muted">{provider.type}</span>
                  </div>
                  {index < fallbackChain.length - 1 && (
                    <ArrowRight className="size-3.5 text-brand-400 shrink-0" aria-hidden="true" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Quick Feature Badges */}
          <div className="grid gap-3 sm:grid-cols-3 pt-1">
            <div className="flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-2/70 px-3 py-2.5 text-xs text-text-secondary">
              <Waypoints className="size-4 text-brand-300 shrink-0" />
              <span>Native WebSockets</span>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-2/70 px-3 py-2.5 text-xs text-text-secondary">
              <Layers3 className="size-4 text-brand-300 shrink-0" />
              <span>Immutable State</span>
            </div>
            <div className="flex items-center gap-2 rounded-lg border border-border-subtle bg-surface-2/70 px-3 py-2.5 text-xs text-text-secondary">
              <ShieldAlert className="size-4 text-brand-300 shrink-0" />
              <span>Zero-Crash Circuit</span>
            </div>
          </div>
        </div>

        {/* Right Side: Live WebSocket Terminal Stream */}
        <div className="overflow-hidden rounded-2xl border border-border bg-surface-1 font-mono text-xs shadow-2xl">
          <div className="flex items-center justify-between border-b border-border-subtle bg-surface-2/80 px-4 py-3 text-[11px] text-text-muted">
            <div className="flex items-center gap-2 min-w-0">
              <Terminal className="size-3.5 text-brand-300 shrink-0" />
              <span className="text-text font-medium truncate">{wsEndpoint}</span>
            </div>
            <span className="flex items-center gap-1.5 text-success-400 font-semibold shrink-0">
              <span className="size-2 rounded-full bg-success-400 animate-pulse-soft" /> LIVE
            </span>
          </div>

          <div className="flex min-h-[380px] flex-col gap-2 p-4 sm:p-5 overflow-x-auto">
            {events.slice(0, visibleCount).map((item, index) => (
              <div
                key={`${item.event}-${index}`}
                className="grid grid-cols-[68px_86px_1fr] items-center gap-2.5 rounded-lg border border-border-subtle bg-surface-2/70 px-3 py-2.5 text-[11px]"
              >
                <span className="text-text-muted font-mono">{item.time}</span>
                <span className={`font-bold tracking-tight ${item.tone}`}>{item.stage}</span>
                <div className="flex items-center justify-between gap-2 truncate">
                  <span className="text-text font-medium">{item.event}</span>
                  <span className="text-text-muted truncate text-right">{item.detail}</span>
                </div>
              </div>
            ))}

            {!reducedMotion && (
              <div className="flex items-center gap-1.5 px-2 pt-2 text-text-muted text-[11px]">
                <span className="size-1.5 rounded-full bg-brand-400 animate-pulse" />
                <span>Listening for real-time LangGraph dispatch events...</span>
              </div>
            )}
          </div>

          <div className="border-t border-border-subtle bg-surface-2/40 px-4 py-2.5 text-[11px] text-text-muted flex items-center justify-between">
            <span>Protocol: JSON-RPC 2.0 via WebSocket</span>
            <span className="text-brand-300 font-semibold">Buffered Events: {visibleCount} / {events.length}</span>
          </div>
        </div>

      </div>
    </section>
  );
}
