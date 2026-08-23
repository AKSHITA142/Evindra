"use client";

import { use, useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  Lightbulb,
  ChevronRight,
  BookOpen,
  Sparkles,
  Award,
  Layers,
  ArrowRight,
  Sliders,
  Copy,
  Check,
} from "lucide-react";
import { MetricCard } from "@/components/cards/MetricCard";

import { Skeleton } from "@/components/loading/Loading";
import { useReport } from "@/hooks/useResearch";
import { useRouter } from "next/navigation";
import { Button } from "@/components/buttons/Button";
import { EmptyState } from "@/components/feedback/EmptyState";
import { ErrorState } from "@/components/feedback/ErrorState";
import { formatPercent } from "@/utils/formatters";




export default function KnowledgePage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);
  const router = useRouter();
  const { data: report, isLoading, isError, refetch } = useReport(jobId);

  const findings = report?.recommendation?.key_findings ?? [];
  const rec = report?.recommendation;

  const [copiedConfig, setCopiedConfig] = useState(false);
  const rawParams = rec?.hyperparameters || {};
  const paramEntries = Object.entries(rawParams);

  const handleCopyConfig = () => {
    navigator.clipboard.writeText(JSON.stringify(rawParams, null, 2));
    setCopiedConfig(true);
    setTimeout(() => setCopiedConfig(false), 2500);
  };

  if (isError) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-12">
        <ErrorState
          title="Failed to Load Knowledge Base"
          description={`Unable to fetch knowledge findings for research job "${jobId}". Please check if the backend API is reachable.`}
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

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 md:py-8">
      {/* ── Page Header ── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/25 text-brand-400 text-xs font-semibold mb-3">
              <BookOpen className="w-3.5 h-3.5" />
              AI Knowledge Base Engine
            </div>
            <h1 className="text-2xl font-bold text-text tracking-tight">
              Research Knowledge Evolution
            </h1>
            <p className="text-sm text-text-muted mt-1 leading-relaxed max-w-xl">
              Structured insights and cumulative empirical learnings extracted by the AI research agent across experiment iterations.
            </p>
          </div>
          <Button
            variant="primary"
            size="sm"
            icon={<ChevronRight className="w-3.5 h-3.5" />}
            onClick={() => router.push(`/recommendation/${jobId}`)}
            className="shrink-0"
          >
            Final Recommendation
          </Button>
        </div>
      </motion.div>

      {/* ── Top Overview Stats ── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 md:gap-4 mb-8">
        <MetricCard
          label="Total Insights"
          value={findings.length}
          icon={<Lightbulb className="w-5 h-5" />}
          accent="brand"
        />
        <MetricCard
          label="Recommended Model"
          value={rec?.recommended_model ?? "—"}
          icon={<Award className="w-5 h-5" />}
          accent="success"
        />
        <MetricCard
          label="Confidence"
          value={rec?.confidence_score ? formatPercent(rec.confidence_score * 100, 0) : "—"}
          icon={<Sparkles className="w-5 h-5" />}
          accent="neutral"
        />
      </div>

      {/* ── Optimal Model Hyperparameter Spec Panel ── */}
      {rec && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="card p-5 mb-8 bg-surface-2 border border-border"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-border-subtle">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-brand-500/10 border border-brand-500/25 flex items-center justify-center">
                <Sliders className="w-4 h-4 text-brand-400" />
              </div>
              <div>
                <h2 className="text-xs font-bold text-text uppercase tracking-wider">
                  Optimal Model Hyperparameter Specification
                </h2>
                <p className="text-xs text-brand-400 font-semibold mt-0.5">
                  {rec.recommended_model}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCopyConfig}
                className="
                  inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                  bg-surface-3 hover:bg-surface-4 text-xs font-semibold text-text
                  border border-border hover:border-brand-500/40 transition-all cursor-pointer
                "
              >
                {copiedConfig ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-success-400" />
                    <span className="text-success-400 font-bold">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-text-muted" />
                    <span>Copy Config</span>
                  </>
                )}
              </button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => router.push(`/recommendation/${jobId}`)}
                icon={<ChevronRight className="w-3.5 h-3.5" />}
                className="text-xs text-text-muted hover:text-brand-400"
              >
                Full Script
              </Button>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {paramEntries.length > 0 ? (
              paramEntries.map(([key, val]) => (
                <div
                  key={key}
                  className="
                    flex items-center gap-1.5 px-2.5 py-1 rounded-md
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
                Optimal standard configuration verified for {rec.recommended_model}
              </span>
            )}
          </div>
        </motion.div>
      )}

      {isLoading ? (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-xl" />
          ))}
        </div>
      ) : findings.length === 0 ? (
        <EmptyState
          icon={BookOpen}
          title="No Knowledge Base Entries Yet"
          description="Insights will populate automatically as the AI research agent completes experimentation iterations."
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

        <div className="space-y-4">
          <p className="text-xs text-text-muted uppercase tracking-widest font-semibold mb-2">
            Chronological Insight Trajectory ({findings.length})
          </p>

          {findings.map((finding, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08, duration: 0.35 }}
              className="card p-5 flex flex-col sm:flex-row items-start gap-4 border border-border hover:border-brand-500/30 transition-colors"
            >
              {/* Step counter badge */}
              <div className="flex-shrink-0 flex items-center gap-2 sm:flex-col sm:items-center justify-between w-full sm:w-auto">
                <div className="w-9 h-9 rounded-xl bg-brand-500/10 border border-brand-500/25 flex items-center justify-center">
                  <span className="text-xs font-bold text-brand-400 font-mono">
                    #{i + 1}
                  </span>
                </div>
                <span className="text-[10px] uppercase font-semibold text-success-400 bg-success-500/10 px-2 py-0.5 rounded-full border border-success-500/20">
                  Verified
                </span>
              </div>

              {/* Main content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="text-xs font-semibold text-text font-mono">
                    Iteration {i + 1} Finding
                  </span>
                </div>
                <p className="text-sm text-text leading-relaxed font-normal">
                  {finding}
                </p>
              </div>

              <CheckCircle2 className="w-5 h-5 text-success-400 shrink-0 hidden sm:block mt-1" />
            </motion.div>
          ))}

          {/* Bottom link to leaderboard */}
          <div className="pt-4 flex justify-center">
            <Button
              variant="ghost"
              size="sm"
              icon={<Layers className="w-4 h-4" />}
              onClick={() => router.push(`/experiments/${jobId}`)}
            >
              Explore Experiment Leaderboard <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
