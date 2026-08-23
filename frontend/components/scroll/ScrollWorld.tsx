"use client";

import { useEffect, useRef } from "react";
import { mountLetsScroll, type ScrollConfig } from "./scrubEngine";

const config: ScrollConfig = {
  brand: { name: "Evidra", href: "/" },
  cta: { label: "Launch workspace", href: "/overview" },
  diveScroll: 2.4,
  crossfade: 0.2,
  hint: "scroll to fly through the pipeline",
  nav: true,
  atmosphere: true,
  connectors: [], // no connectors — the engine crossfades between scenes
  sections: [
    {
      id: "rawdata",
      label: "Raw Data",
      still: "/world/rawdata.webp",
      clip: "/world/rawdata.mp4",
      accent: "#76FF03",
      scroll: 1.6, // this clip is ~5s, so it needs less scroll distance than the 10s clips
      eyebrow: "From messy CSV",
      title: "It starts with raw data.",
      body: "Upload any tabular dataset — Evidra takes it from there.",
      tags: ["Any CSV", "Zero config"],
    },
    {
      id: "profiling",
      label: "Profiling",
      still: "/world/profiling.webp",
      clip: "/world/profiling.mp4",
      accent: "#76FF03",
      eyebrow: "Understand",
      title: "It reads your data.",
      body: "A semantic profile of every column, with leakage flagged before training.",
      tags: ["Semantic profile", "Leakage-safe"],
    },
    {
      id: "agentloop",
      label: "Agent Loop",
      still: "/world/agentloop.webp",
      clip: "/world/agentloop.mp4",
      accent: "#76FF03",
      scroll: 2.8,
      linger: 0.4,
      eyebrow: "Reason",
      title: "A LangGraph agent, on a loop.",
      body: "It forms hypotheses and tests them across a bounded research budget.",
      tags: ["LangGraph", "Multi-LLM"],
    },
    {
      id: "crossval",
      label: "Cross-Validation",
      still: "/world/crossval.webp",
      clip: "/world/crossval.mp4",
      accent: "#76FF03",
      eyebrow: "Validate",
      title: "Five-fold, no leakage.",
      body: "Every idea is proven with rigorous cross-validation — not a single lucky split.",
      tags: ["5-fold CV", "XGBoost · LightGBM · CatBoost"],
    },
    {
      id: "decision",
      label: "Decision Gate",
      still: "/world/decision.webp",
      clip: "/world/decision.mp4",
      accent: "#76FF03",
      eyebrow: "Decide",
      title: "Only real gains pass.",
      body: "A 0.5% improvement threshold keeps the noise out of your pipeline.",
      tags: ["0.5% gate", "Evidence-driven"],
    },
    {
      id: "champion",
      label: "Champion",
      still: "/world/champion.webp",
      clip: "/world/champion.mp4",
      accent: "#76FF03",
      scroll: 3.0,
      linger: 0.45,
      eyebrow: "Ship",
      title: "An evidence-backed pipeline.",
      body: "A champion model with an auditable report — ready for the research workspace.",
      tags: ["Champion pipeline", "Audit report"],
      cta: {
        primary: { label: "Launch workspace", href: "/overview" },
        secondary: { label: "View on GitHub", href: "https://github.com/AKSHITA142/DataPilot-AI" },
      },
    },
  ],
};

export default function ScrollWorld() {
  const ref = useRef<HTMLDivElement>(null);
  const mounted = useRef(false);

  useEffect(() => {
    if (mounted.current || !ref.current) return;
    mounted.current = true;
    mountLetsScroll(ref.current, config);
  }, []);

  // Auto-scroll intentionally removed: scroll-scrubbed video can't auto-play both
  // fast and smoothly (seeking is decode-bound). Manual scroll is the intended,
  // smooth UX; an animated "scroll" hint (see .sw-hint overrides in globals.css)
  // signals that the page is interactive.

  return (
    <div
      ref={ref}
      style={
        {
          "--sw-bg": "#080C0E",
          "--sw-ink": "#E6E7E9",
          "--sw-ink-soft": "#A1A4AA",
          "--sw-accent": "#76FF03",
          "--sw-font-display": "var(--font-inter), ui-sans-serif, system-ui, sans-serif",
          "--sw-font-body": "var(--font-inter), ui-sans-serif, system-ui, sans-serif",
        } as React.CSSProperties
      }
    />
  );
}
