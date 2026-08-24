"use client";

import { use, useState } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Trophy,
  Download,
  CheckCircle2,
  Lightbulb,
  Wrench,
  ChevronLeft,
  ArrowRight,
  ShieldAlert,
  Sparkles,
  BarChart3,
  Layers,
  FlaskConical,
  BookOpen,
  Sliders,
  Copy,
  Check,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/buttons/Button";
import { MetricCard } from "@/components/cards/MetricCard";


import { SkeletonCard } from "@/components/loading/Loading";
import { HorizontalBarChart } from "@/components/charts/Charts";
import { EmptyState } from "@/components/feedback/EmptyState";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useReport, useExperiments, useDataset } from "@/hooks/useResearch";
import { downloadReport } from "@/services/apiClient";
import { formatMetric } from "@/utils/formatters";
import { useRouter } from "next/navigation";
import type { QualityWarning, SemanticProfile } from "@/types/api";


export default function RecommendationPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);
  const router = useRouter();
  const { data: report, isLoading, isError, refetch } = useReport(jobId);
  const { data: experiments } = useExperiments(jobId);
  const { data: dataset } = useDataset(report?.dataset_id ?? null);

  const [activeCodeTab, setActiveCodeTab] = useState<"python" | "json">("python");
  const [copiedCode, setCopiedCode] = useState(false);

  const rec = report?.recommendation;

  if (isError) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <ErrorState
          title="Failed to Load Research Report"
          description={`Unable to fetch the recommendation report for job "${jobId}". Please check if the backend service is running.`}
          onRetry={() => refetch()}
          action={
            <Button variant="primary" size="sm" onClick={() => router.push(`/timeline/${jobId}`)}>
              View Timeline
            </Button>
          }
        />
      </div>
    );
  }

  // Feature importance for winner experiment
  const winnerExp = experiments?.find(
    (e) => e.experiment_id === rec?.experiment_id
  );
  const featureData = winnerExp?.feature_importance
    ? Object.entries(winnerExp.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([name, value]) => ({ name, value }))
    : [];

  const rawParams = rec?.hyperparameters || {};
  const paramEntries = Object.entries(rawParams);

  const pythonParamsStr = paramEntries.length > 0
    ? paramEntries
        .map(([k, v]) => `    ${k}=${typeof v === "string" ? `"${v}"` : typeof v === "boolean" ? (v ? "True" : "False") : v}`)
        .join(",\n")
    : "    random_state=42";

  const pythonSnippet = `# ==============================================================================
# DataPilot-AI Production Pipeline & Hyperparameter Configuration
# Model: ${rec?.recommended_model ?? "MLModel"}
# Target Metric: ${rec?.primary_metric_name ?? "Score"} (${(rec?.primary_metric_value ?? 0).toFixed(4)})
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load your dataset
df = pd.read_csv("your_dataset.csv")

# 2. Split Features & Target (adjust column names as appropriate)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Instantiate model with DataPilot-AI recommended hyperparameters
from sklearn.ensemble import ${rec?.recommended_model ?? "HistGradientBoostingClassifier"}

model = ${rec?.recommended_model ?? "HistGradientBoostingClassifier"}(
${pythonParamsStr}
)

# 4. Train model & evaluate
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Optimal Model Test Score: {test_score:.4f}")
`;

  const jsonSnippet = JSON.stringify(
    {
      model_name: rec?.recommended_model,
      hyperparameters: rawParams,
      pipeline_steps: rec?.recommended_pipeline,
      validation_score: rec?.primary_metric_value,
      confidence_score: rec?.confidence_score,
    },
    null,
    2
  );

  const handleCopyCode = () => {
    const textToCopy = activeCodeTab === "python" ? pythonSnippet : jsonSnippet;
    navigator.clipboard.writeText(textToCopy);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2500);
  };

  const profileObj = dataset?.profile as SemanticProfile | undefined;
  const qualityWarnings: QualityWarning[] = Array.isArray(profileObj?.quality_warnings)
    ? profileObj.quality_warnings
    : [];

  const handleDownload = async (format: "html" | "markdown") => {
    if (!report?.report_id) return;
    try {
      const blob = await downloadReport(report.report_id, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `datapilot-report.${format === "html" ? "html" : "md"}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // ignore download errors
    }
  };

  const handleDownloadBusinessDataset = () => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    const downloadUrl = `${backendUrl}/api/v1/reports/${jobId}/download-business-dataset`;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = `business_action_${jobId}.csv`;
    a.click();
  };

  const handleDownloadMLMatrix = () => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    const downloadUrl = `${backendUrl}/api/v1/reports/${jobId}/download-ml-feature-matrix`;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = `ml_ready_matrix_${jobId}.csv`;
    a.click();
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 md:py-8">
      {/* ── Page Header / Action Bar ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            icon={<ChevronLeft className="w-3.5 h-3.5" />}
            onClick={() => router.push(`/experiments/${jobId}`)}
          >
            Leaderboard
          </Button>
          <span className="text-text-muted text-xs">•</span>
          <span className="text-xs font-mono text-text-muted">Job {jobId.slice(0, 8)}</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            icon={<BookOpen className="w-3.5 h-3.5" />}
            onClick={() => router.push(`/knowledge/${jobId}`)}
          >
            Knowledge Base
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<Download className="w-3.5 h-3.5" />}
            onClick={() => handleDownloadBusinessDataset()}
          >
            Business Action CSV
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<Download className="w-3.5 h-3.5" />}
            onClick={() => handleDownloadMLMatrix()}
          >
            ML-Ready Feature Matrix
          </Button>
          <Button
            variant="primary"
            size="sm"
            icon={<Download className="w-3.5 h-3.5" />}
            onClick={() => handleDownload("markdown")}
          >
            Export Report
          </Button>
        </div>
      </div>


      {isLoading ? (
        <div className="space-y-6">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : !rec ? (
        <EmptyState
          icon={FlaskConical}
          title="Recommendation Report Not Ready"
          description="The AI research engine is currently evaluating pipeline experiments for this job. Check the live timeline for execution progress."
          action={
            <Button
              variant="primary"
              size="sm"
              onClick={() => router.push(`/timeline/${jobId}`)}
            >
              View Timeline Progress
            </Button>
          }
        />
      ) : (

        <>
          {/* ── 1. PRIMARY RESULT HERO BANNER ── */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="relative card p-6 sm:p-8 mb-8 overflow-hidden border border-success-500/30 bg-surface-2"
          >
            {/* Background subtle radial glow */}
            <div
              className="absolute -top-24 -right-24 w-80 h-80 rounded-full pointer-events-none"
              style={{
                background:
                  "radial-gradient(circle, rgba(34,197,94,0.12) 0%, transparent 70%)",
              }}
            />

            <div className="relative z-10 flex flex-col lg:flex-row items-start justify-between gap-6">
              <div className="flex-1 min-w-0">
                {/* Winner Pill */}
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-success-500/10 border border-success-500/25 text-success-400 text-xs font-semibold mb-3">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Top Recommended Pipeline
                </div>

                <h1 className="text-2xl sm:text-3xl font-bold text-text mb-3 tracking-tight">
                  {rec.recommended_model}
                </h1>

                {/* Connected Pipeline Steps Flow */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-xs text-text-muted font-medium mr-1 flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-brand-400" /> Pipeline:
                  </span>
                  {rec.recommended_pipeline.map((step, idx) => (
                    <div key={step} className="flex items-center gap-1.5">
                      <span className="text-xs px-2.5 py-1 rounded-md bg-surface-3 border border-border text-text font-mono">
                        {step}
                      </span>
                      {idx < rec.recommended_pipeline.length - 1 && (
                        <ArrowRight className="w-3 h-3 text-text-muted" />
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Confidence Score Gauge Card (Tremor Style) */}
              <div className="flex items-center gap-4 bg-surface-3 p-4 rounded-xl border border-border shrink-0 self-stretch sm:self-auto justify-center">
                <div className="relative w-16 h-16 shrink-0">
                  <svg viewBox="0 0 80 80" className="w-16 h-16 -rotate-90">
                    <circle cx="40" cy="40" r="32" fill="none" stroke="var(--border)" strokeWidth="6" />
                    <circle
                      cx="40" cy="40" r="32" fill="none"
                      stroke="var(--success)" strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 32}`}
                      strokeDashoffset={`${2 * Math.PI * 32 * (1 - rec.confidence_score)}`}
                      style={{ transition: "stroke-dashoffset 1s ease" }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-base font-bold text-success-400 font-mono">
                      {(rec.confidence_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-text-muted uppercase tracking-wider font-semibold">
                    Confidence
                  </p>
                  <p className="text-sm font-bold text-text mt-0.5">High Certainty</p>
                  <p className="text-[10px] text-text-muted">Based on cross-validation</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* ── 2. SUPPORTING METRICS GRID ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-8">
            <MetricCard
              label={rec.primary_metric_name}
              value={formatMetric(rec.primary_metric_value)}
              icon={<Trophy className="w-5 h-5" />}
              subtext="Primary validation score"
              accent="success"
            />
            <MetricCard
              label="Composite Score"
              value={formatMetric(rec.composite_score)}
              icon={<Sparkles className="w-5 h-5" />}
              subtext="Multi-objective score"
              accent="brand"
            />
            <MetricCard
              label="Experiments Run"
              value={report?.experiment_count ?? "—"}
              icon={<FlaskConical className="w-5 h-5" />}
              subtext="Pipelines evaluated"
              accent="neutral"
            />
            <MetricCard
              label="Findings Extracted"
              value={report?.knowledge_findings_count ?? "—"}
              icon={<Lightbulb className="w-5 h-5" />}
              subtext="Knowledge base insights"
              accent="neutral"
            />
          </div>

          {/* ── 3. REASONING & STRATEGIC EXPLANATION ── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mb-8"
          >
            <div className="card p-6 border-l-4 border-l-brand-500 bg-surface-2">
              <p className="text-xs text-brand-400 uppercase tracking-widest font-semibold mb-2.5 flex items-center gap-2">
                <Brain className="w-4 h-4" />
                AI Scientist Reasoning & Strategy
              </p>
              <p className="text-sm text-text-secondary leading-relaxed font-normal">
                {rec.reasoning}
              </p>
            </div>
          </motion.div>

          {/* ── 3.5 OPTIMAL HYPERPARAMETERS & PRODUCTION TRAINING CODE ── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18 }}
            className="mb-8"
          >
            <div className="card p-6 bg-surface-2 border border-border">
              {/* Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 pb-4 border-b border-border-subtle">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/25 flex items-center justify-center">
                    <Sliders className="w-4 h-4 text-brand-400" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-text">
                      Recommended Model Hyperparameters & Training Code
                    </h2>
                    <p className="text-xs text-text-muted">
                      Exact parameters verified and tuned for {rec.recommended_model}
                    </p>
                  </div>
                </div>

                {/* Tab Switcher & Copy Action */}
                <div className="flex items-center gap-2 self-start sm:self-auto">
                  <div className="flex rounded-lg bg-surface-3 p-0.5 border border-border-subtle text-xs">
                    <button
                      type="button"
                      onClick={() => setActiveCodeTab("python")}
                      className={`px-3 py-1 rounded-md font-medium transition-colors ${
                        activeCodeTab === "python"
                          ? "bg-brand-500 text-[#052620] font-bold shadow-sm"
                          : "text-text-muted hover:text-text"
                      }`}
                    >
                      Python Script
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveCodeTab("json")}
                      className={`px-3 py-1 rounded-md font-medium transition-colors ${
                        activeCodeTab === "json"
                          ? "bg-brand-500 text-[#052620] font-bold shadow-sm"
                          : "text-text-muted hover:text-text"
                      }`}
                    >
                      JSON Config
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={handleCopyCode}
                    className="
                      flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                      bg-surface-3 hover:bg-surface-4 text-xs font-semibold text-text
                      border border-border hover:border-brand-500/40 transition-all cursor-pointer
                    "
                  >
                    {copiedCode ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-success-400" />
                        <span className="text-success-400 font-bold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-text-muted" />
                        <span>Copy Code</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Hyperparameter Badges Grid */}
              <div className="mb-4">
                <p className="text-[11px] text-text-muted uppercase tracking-wider font-semibold mb-2.5">
                  Optimal Hyperparameter Values ({paramEntries.length}):
                </p>
                <div className="flex flex-wrap gap-2">
                  {paramEntries.length > 0 ? (
                    paramEntries.map(([key, val]) => (
                      <div
                        key={key}
                        className="
                          flex items-center gap-2 px-3 py-1.5 rounded-lg
                          bg-surface-3 border border-border-subtle text-xs
                        "
                      >
                        <span className="font-mono text-text-muted font-medium">{key}:</span>
                        <span className="font-mono font-bold text-brand-400">
                          {typeof val === "boolean" ? (val ? "True" : "False") : String(val)}
                        </span>
                      </div>
                    ))
                  ) : (
                    <span className="text-xs text-text-muted italic">
                      Standard tuned defaults applied for {rec.recommended_model}
                    </span>
                  )}
                </div>
              </div>

              {/* Code Snippet Box */}
              <div className="relative rounded-xl overflow-hidden border border-border-subtle bg-[#0c0c0e]">
                <div className="flex items-center justify-between px-4 py-2 bg-surface-3/50 border-b border-border-subtle text-[11px] text-text-muted font-mono">
                  <span className="flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5 text-brand-400" />
                    {activeCodeTab === "python" ? "train_optimal_model.py" : "hyperparameters.json"}
                  </span>
                  <span className="text-[10px] text-text-muted/60">Ready to execute</span>
                </div>
                <pre className="p-4 text-xs font-mono text-text-secondary overflow-x-auto leading-relaxed max-h-72 select-text">
                  <code>{activeCodeTab === "python" ? pythonSnippet : jsonSnippet}</code>
                </pre>
              </div>
            </div>
          </motion.div>

          {/* ── 4. DATA QUALITY WARNINGS (IF GENUINE WARNINGS EXIST) ── */}
          {qualityWarnings.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8"
            >
              <div className="card p-6 border-l-4 border-l-warning-500 bg-surface-2">
                <p className="text-xs text-warning-400 uppercase tracking-widest font-semibold mb-3 flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4" />
                  Dataset Quality Alerts ({qualityWarnings.length})
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {qualityWarnings.map((w: { column?: string; severity?: string; message?: string }, idx: number) => (
                    <div key={idx} className="p-3 rounded-lg bg-surface-3 border border-border-subtle text-xs">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-text font-mono">{w.column ?? "Dataset"}</span>
                        <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded ${
                          w.severity === "high" ? "bg-error-500/20 text-error-400" : "bg-warning-500/20 text-warning-400"
                        }`}>
                          {w.severity}
                        </span>
                      </div>
                      <p className="text-text-muted">{w.message}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── 5. KEY FINDINGS + IMPLEMENTATION GUIDE (SIDE-BY-SIDE) ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* Key Findings */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
            >
              <div className="card p-6 h-full flex flex-col">
                <p className="text-xs text-text-muted uppercase tracking-widest font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-success-400" />
                  Key Findings & Insights
                </p>
                <ul className="space-y-3 flex-1">
                  {rec.key_findings.map((f, i) => (
                    <li key={i} className="flex items-start gap-3 text-xs leading-relaxed text-text-secondary">
                      <span className="w-5 h-5 rounded-full bg-success-500/15 text-success-400 font-semibold flex items-center justify-center shrink-0 mt-0.5 text-[10px]">
                        ✓
                      </span>
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>

            {/* Implementation Tips */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <div className="card p-6 h-full flex flex-col">
                <p className="text-xs text-text-muted uppercase tracking-widest font-semibold mb-4 flex items-center gap-2">
                  <Wrench className="w-4 h-4 text-brand-400" />
                  Deployment & Implementation Steps
                </p>
                <ol className="space-y-3 flex-1">
                  {rec.implementation_tips.map((tip, i) => (
                    <li key={i} className="flex items-start gap-3 text-xs leading-relaxed text-text-secondary">
                      <span className="w-5 h-5 rounded-full bg-brand-500/15 text-brand-400 font-bold font-mono flex items-center justify-center shrink-0 mt-0.5 text-[10px]">
                        {i + 1}
                      </span>
                      <span>{tip}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </motion.div>
          </div>

          {/* ── 6. FEATURE IMPORTANCE VISUALIZER ── */}
          {featureData.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="mb-8"
            >
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-xs text-text-muted uppercase tracking-widest font-semibold flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-brand-400" />
                    Feature Importance — Top Predictors
                  </p>
                  <span className="text-[10px] text-text-muted font-mono">{rec.recommended_model}</span>
                </div>
                <HorizontalBarChart data={featureData} height={240} />
              </div>
            </motion.div>
          )}

          {/* ── 7. LIVE HTML AUDIT REPORT IFRAME ── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mb-8"
          >
            <div className="card p-6 border border-border">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs text-brand-400 uppercase tracking-widest font-semibold flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  Technical Audit & Validation Report (HTML Output)
                </p>
                <span className="text-[10px] text-text-muted font-mono">Real-Time Generated Audit Report</span>
              </div>
              <div className="rounded-xl overflow-hidden border border-border bg-bg shadow-inner h-[600px] w-full">
                <iframe
                  src={`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/api/v1/reports/${jobId}/html`}
                  className="w-full h-full border-0"
                  title="DataPilot-AI HTML Research Report"
                />
              </div>
            </div>
          </motion.div>

          {/* ── 8. DOWNLOAD & EXPORT ACTIONS ── */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 }}
            className="card p-6 flex flex-col sm:flex-row items-center justify-between gap-4"
          >
            <div>
              <p className="text-sm font-bold text-text mb-0.5">Export Production Report & Dataset</p>
              <p className="text-xs text-text-muted">
                Download full standalone HTML/Markdown research summary or preprocessed CSV artifact.
              </p>
            </div>
            <div className="flex flex-wrap gap-2.5 shrink-0">
              <Button
                variant="primary"
                size="sm"
                icon={<Download className="w-3.5 h-3.5" />}
                onClick={() => handleDownload("html")}
              >
                HTML Report
              </Button>
              <Button
                variant="secondary"
                size="sm"
                icon={<Download className="w-3.5 h-3.5" />}
                onClick={() => handleDownload("markdown")}
              >
                Markdown Report
              </Button>
              <Button
                variant="secondary"
                size="sm"
                icon={<Download className="w-3.5 h-3.5" />}
                onClick={handleDownloadBusinessDataset}
              >
                Business Action CSV
              </Button>
              <Button
                variant="secondary"
                size="sm"
                icon={<Download className="w-3.5 h-3.5" />}
                onClick={handleDownloadMLMatrix}
              >
                ML-Ready Matrix
              </Button>
            </div>
          </motion.div>
        </>
      )}
    </div>
  );
}
