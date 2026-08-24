"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  Brain,
  Cpu,
  Layers,
  LineChart,
  ShieldCheck,
  PackageCheck,
  CheckCircle2,
  Sparkles,
  BarChart3,
  Gauge,
  Scale,
  Briefcase,
} from "lucide-react";

interface FeatureItem {
  icon: React.ElementType;
  tag: string;
  title: string;
  body: string;
  highlights: string[];
}

const features: FeatureItem[] = [
  {
    icon: Brain,
    tag: "PROFILING ENGINE",
    title: "Semantic Profiling & Leakage Shield",
    body: "Diagnoses target distribution, cardinality, and missingness. Automatically strips identifier columns and timestamps masquerading as features to prevent catastrophic test leakage.",
    highlights: ["Target Imbalance Detection", "Automated ID Isolation", "Datetime Cyclical Encoding"],
  },
  {
    icon: Layers,
    tag: "PLANNING AGENT",
    title: "Hypothesis-Driven Strategy Search",
    body: "Rather than brute-force grid searching, the LLM reasoning agent formulates structured, falsifiable research hypotheses—pairing feature encoders and model families based on data geometry.",
    highlights: ["Target & Frequency Encoding", "Power & Quantile Transforms", "Bounded Parameter Tuning"],
  },
  {
    icon: Cpu,
    tag: "ML RUNTIME",
    title: "Leakage-Safe Multi-Model Execution",
    body: "Constructs scikit-learn pipelines with fit-transforms isolated strictly inside training folds. Trains XGBoost, LightGBM, CatBoost, Random Forest, and regularized linear models.",
    highlights: ["5-Fold Stratified / K-Fold CV", "Out-of-Fold Error Tracking", "Reproducible Random Seeds"],
  },
  {
    icon: LineChart,
    tag: "EVALUATION ENGINE",
    title: "4-Pillar Multi-Dimensional Ranking",
    body: "Eliminates leaderboard chasing by evaluating candidate models across generalization power, fold stability, inference latency, and business objective satisfaction.",
    highlights: ["Cross-Fold Variance Penalty", "Inference Latency Profiling", "Overfitting Gap Audit"],
  },
  {
    icon: ShieldCheck,
    tag: "STATE GRAPH",
    title: "Cyclic Knowledge Memory & Routing",
    body: "Retains an immutable state graph storing winning hypotheses and feature importances across a 5-iteration budget. If gain exceeds 0.5%, it routes back to Planner with fresh context.",
    highlights: ["LangGraph State Machine", "Iterative Context Injection", "0.5% Gain Stop Condition"],
  },
  {
    icon: PackageCheck,
    tag: "EXPORT & AUDIT",
    title: "Production Artifacts & Executive Reports",
    body: "Synthesizes final discoveries into an executive research report, and serializes the winning pipeline into self-contained .joblib weights ready for immediate deployment.",
    highlights: ["Single-file .joblib Export", "Feature Importance Graphs", "Executive Summary Generator"],
  },
];

const rankingDimensions = [
  {
    label: "Generalization",
    weight: "35%",
    desc: "Primary test metric (ROC-AUC, F1, or RMSE)",
    icon: BarChart3,
    color: "bg-brand-400",
  },
  {
    label: "Cross-Fold Stability",
    weight: "25%",
    desc: "Inverse variance across all 5 cross-validation folds",
    icon: Scale,
    color: "bg-info-400",
  },
  {
    label: "Inference Efficiency",
    weight: "20%",
    desc: "Prediction throughput and memory footprint",
    icon: Gauge,
    color: "bg-warning-400",
  },
  {
    label: "Business Constraints",
    weight: "20%",
    desc: "False positive penalties and model interpretability",
    icon: Briefcase,
    color: "bg-success-400",
  },
];

export function FeatureGrid() {
  const reducedMotion = useReducedMotion();

  return (
    <section id="capabilities" className="scroll-mt-20 border-b border-border-subtle py-24 sm:py-32">
      <div className="mx-auto flex max-w-7xl flex-col gap-14 px-4 sm:px-6">
        
        {/* Header */}
        <div className="flex flex-col gap-4 max-w-3xl">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.24em] text-brand-300">
            <Sparkles className="size-3.5" /> Engine Architecture
          </div>
          <h2 className="text-balance text-4xl font-semibold tracking-tight text-text sm:text-5xl">
            Engineered for genuine machine learning research.
          </h2>
          <p className="text-pretty text-lg leading-relaxed text-text-secondary">
            Every module in Evidra is designed to replicate the workflow of a senior data scientist: identifying data quirks, formulating disciplined hypotheses, ensuring zero data leakage, and scoring evidence with statistical rigor.
          </p>
        </div>

        {/* 6 Feature Cards Grid */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.article
                key={feature.title}
                initial={reducedMotion ? false : { opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: (index % 3) * 0.08 }}
                className="flex flex-col justify-between rounded-2xl border border-border-subtle bg-surface-2/70 p-6 sm:p-7 transition-all hover:border-border hover:bg-surface-2 shadow-sm"
              >
                <div className="flex flex-col gap-4">
                  <div className="flex items-center justify-between">
                    <span className="flex size-11 items-center justify-center rounded-xl border border-border bg-surface-3 text-brand-300 shadow-sm">
                      <Icon className="size-5" aria-hidden="true" />
                    </span>
                    <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                      {feature.tag}
                    </span>
                  </div>

                  <div className="flex flex-col gap-2">
                    <h3 className="text-lg font-bold text-text tracking-tight">
                      {feature.title}
                    </h3>
                    <p className="text-sm leading-relaxed text-text-secondary">
                      {feature.body}
                    </p>
                  </div>
                </div>

                {/* Micro-tags */}
                <div className="mt-6 flex flex-wrap gap-1.5 pt-4 border-t border-border-subtle">
                  {feature.highlights.map((h) => (
                    <span
                      key={h}
                      className="inline-flex items-center rounded-md border border-border-subtle bg-surface-3 px-2 py-0.5 font-mono text-[11px] text-text-muted"
                    >
                      {h}
                    </span>
                  ))}
                </div>
              </motion.article>
            );
          })}
        </div>

        {/* Highlight Banner: 4-Pillar Evaluation Formula */}
        <motion.div
          initial={reducedMotion ? false : { opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="rounded-2xl border border-brand-500/25 bg-brand-500/[0.04] p-6 sm:p-10"
        >
          <div className="grid gap-8 lg:grid-cols-[1.1fr_.9fr] lg:items-center">
            
            {/* Left Description */}
            <div className="flex flex-col gap-4">
              <span className="font-mono text-xs uppercase tracking-[0.24em] text-brand-300 font-semibold">
                Multi-Dimensional Scoring Model
              </span>
              <h3 className="text-2xl sm:text-3xl font-bold tracking-tight text-text">
                Never pick a fragile model based on a single score.
              </h3>
              <p className="text-sm sm:text-base leading-relaxed text-text-secondary">
                Standard AutoML tools overfit by picking the model with the highest single-split validation metric. Evidra calculates a composite confidence score by balancing out-of-fold generalization with cross-fold stability, inference latency, and task constraints.
              </p>
              <div className="flex items-center gap-3 pt-2 text-xs font-mono text-text-muted">
                <span className="flex items-center gap-1 text-brand-300">
                  <CheckCircle2 className="size-3.5" /> 5-Fold Stratified CV
                </span>
                <span>•</span>
                <span className="flex items-center gap-1 text-brand-300">
                  <CheckCircle2 className="size-3.5" /> Stability Penalty
                </span>
                <span>•</span>
                <span className="flex items-center gap-1 text-brand-300">
                  <CheckCircle2 className="size-3.5" /> Latency Budget
                </span>
              </div>
            </div>

            {/* Right Pillar Weight Visualizer */}
            <div className="flex flex-col gap-3 rounded-xl border border-border bg-surface-1/90 p-5 shadow-xl">
              <span className="font-mono text-xs font-semibold text-text border-b border-border-subtle pb-2.5">
                Evaluation Weights Formula
              </span>
              {rankingDimensions.map((dim) => {
                const DimIcon = dim.icon;
                return (
                  <div key={dim.label} className="flex flex-col gap-1.5 py-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-2 font-medium text-text">
                        <DimIcon className="size-3.5 text-text-muted" />
                        {dim.label}
                      </span>
                      <span className="font-mono font-bold text-brand-300">{dim.weight}</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
                      <div className={`h-full rounded-full ${dim.color}`} style={{ width: dim.weight }} />
                    </div>
                    <span className="text-[11px] text-text-muted leading-tight">{dim.desc}</span>
                  </div>
                );
              })}
            </div>

          </div>
        </motion.div>

      </div>
    </section>
  );
}

