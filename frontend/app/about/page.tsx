import Link from "next/link";
import Image from "next/image";
import { ArrowRight, Brain, Cpu, GitBranch, Network, Sparkles, Zap } from "lucide-react";
import { FeatureGrid } from "@/components/landing/FeatureGrid";
import { Hero } from "@/components/landing/Hero";
import { TelemetryPanel } from "@/components/landing/TelemetryPanel";
import { WorkflowDiagram } from "@/components/landing/WorkflowDiagram";

interface TechLayer {
  layerName: string;
  icon: React.ElementType;
  description: string;
  technologies: { name: string; role: string }[];
}

const techLayers: TechLayer[] = [
  {
    layerName: "01 / ML RUNTIME & MODEL GRID",
    icon: Cpu,
    description: "Leakage-free pipelines, cross-validation, and multi-model training.",
    technologies: [
      { name: "Scikit-Learn", role: "Pipelines, encoders, & scalers" },
      { name: "XGBoost", role: "Gradient boosted decision trees" },
      { name: "LightGBM", role: "Fast histogram-based learning" },
      { name: "CatBoost", role: "Categorical feature handling" },
    ],
  },
  {
    layerName: "02 / AGENTIC ORCHESTRATION",
    icon: GitBranch,
    description: "Cyclic state machine managing the 5-iteration research budget.",
    technologies: [
      { name: "LangGraph", role: "Cyclic multi-agent state graph" },
      { name: "Pydantic v2", role: "Strict typed state validation" },
      { name: "Decision Engine", role: "0.5% gain threshold router" },
      { name: "State Memory", role: "Immutable research history" },
    ],
  },
  {
    layerName: "03 / MULTI-LLM REASONING CHAIN",
    icon: Brain,
    description: "High-reasoning intelligence with automated zero-downtime fallback.",
    technologies: [
      { name: "Gemini LLM", role: "Primary hypothesis reasoning" },
      { name: "OpenAI LLM", role: "High-throughput fallback" },
      { name: "OpenRouter Open-Source LLM", role: "Multi-provider resilience" },
      { name: "Rule Engine", role: "Deterministic fallback" },
    ],
  },
  {
    layerName: "04 / REAL-TIME WORKSPACE & API",
    icon: Network,
    description: "Sub-second streaming telemetry and reactive research workspace.",
    technologies: [
      { name: "FastAPI & Uvicorn", role: "Async Python 3.12+ backend" },
      { name: "WebSockets", role: "Real-time bi-directional telemetry" },
      { name: "Next.js 16", role: "Turbopack App Router interface" },
      { name: "Tailwind CSS v4", role: "Design tokens & glassmorphism" },
    ],
  },
];

export default function AboutPage() {
  return (
    <div className="overflow-x-hidden bg-bg text-text">
      <Hero />
      <WorkflowDiagram />
      <FeatureGrid />
      <TelemetryPanel />

      {/* Tech Architecture Section */}
      <section id="stack" className="scroll-mt-20 border-b border-border-subtle py-24 sm:py-32">
        <div className="mx-auto flex max-w-7xl flex-col gap-14 px-4 sm:px-6">

          {/* Section Header */}
          <div className="flex flex-col gap-4 max-w-3xl">
            <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.24em] text-brand-300">
              <Sparkles className="size-3.5" /> Technical Foundation
            </div>
            <h2 className="text-balance text-4xl font-semibold tracking-tight text-text sm:text-5xl">
              Engineered on a modern, inspectable open architecture.
            </h2>
            <p className="text-pretty text-lg leading-relaxed text-text-secondary">
              Evidra couples battle-tested machine learning frameworks with stateful LangGraph orchestration and multi-LLM reasoning to create a durable, production-ready research platform.
            </p>
          </div>

          {/* 4-Layer Architectural Grid */}
          <div className="grid gap-6 md:grid-cols-2">
            {techLayers.map((layer) => {
              const LayerIcon = layer.icon;
              return (
                <div
                  key={layer.layerName}
                  className="flex flex-col justify-between rounded-2xl border border-border-subtle bg-surface-2/80 p-6 sm:p-8 transition-all hover:border-border hover:bg-surface-2 shadow-sm"
                >
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-brand-300 tracking-wider">
                        {layer.layerName}
                      </span>
                      <span className="flex size-9 items-center justify-center rounded-lg border border-border bg-surface-3 text-brand-300">
                        <LayerIcon className="size-4.5" />
                      </span>
                    </div>

                    <p className="text-sm text-text-secondary leading-relaxed">
                      {layer.description}
                    </p>
                  </div>

                  {/* Technologies Inside Layer */}
                  <div className="mt-6 grid grid-cols-2 gap-2.5 pt-5 border-t border-border-subtle">
                    {layer.technologies.map((tech) => (
                      <div
                        key={tech.name}
                        className="flex flex-col rounded-lg border border-border-subtle bg-surface-3/80 p-2.5"
                      >
                        <span className="font-mono text-xs font-bold text-text truncate">
                          {tech.name}
                        </span>
                        <span className="font-mono text-[10px] text-text-muted truncate mt-0.5">
                          {tech.role}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

        </div>
      </section>

      {/* Call To Action Section */}
      <section className="py-24 sm:py-32">
        <div className="mx-auto max-w-5xl px-4 text-center sm:px-6">
          <div className="flex flex-col items-center gap-7 rounded-3xl border border-brand-500/30 bg-gradient-to-b from-brand-500/[0.08] to-transparent px-6 py-16 sm:px-14 sm:py-20 shadow-2xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-brand-500/40 bg-brand-500/10 px-3.5 py-1.5 font-mono text-xs text-brand-300">
              <Zap className="size-3.5" /> Zero Configuration Required
            </div>

            <h2 className="max-w-3xl text-balance text-4xl font-bold tracking-tight text-text sm:text-6xl">
              Stop babysitting model runs. Start directing AI research.
            </h2>

            <p className="max-w-2xl text-pretty text-lg leading-relaxed text-text-secondary">
              Upload any tabular CSV dataset. Evidra builds the semantic profile, formulates hypotheses, executes 5-fold cross-validation, and returns an evidence-backed champion pipeline.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 pt-2">
              <Link
                href="/overview"
                prefetch={true}
                className="inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-brand-500 px-8 font-semibold text-bg transition-all hover:bg-brand-400 shadow-lg shadow-brand-500/20 cursor-pointer"
              >
                Launch Research Workspace <ArrowRight className="size-4" aria-hidden="true" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border-subtle py-10 bg-surface-1/50">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 sm:px-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <Image
              src="/evidra-second-logo.png"
              alt="Evidra"
              width={947}
              height={380}
              className="h-11 w-auto object-contain"
            />
            <span className="text-text-muted text-xs border-l border-border pl-4">
              Autonomous Evidence-Driven AI for Data Science
            </span>
          </div>

          <div className="flex items-center gap-6 text-sm text-text-muted font-mono text-xs">
            <a
              className="transition-colors hover:text-brand-300"
              href="https://github.com/AKSHITA142/DataPilot-AI"
              target="_blank"
              rel="noreferrer"
            >
              GitHub Repository
            </a>
            <span>•</span>
            <span>MIT License</span>
            <span>•</span>
            <span>v2.0 Production</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
