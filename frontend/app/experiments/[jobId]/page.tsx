"use client";

import { use, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Filter,
  GitCompare,
  ChevronDown,
  ChevronUp,
  Trophy,
  ArrowRight,
  FlaskConical,
} from "lucide-react";

import { Button } from "@/components/buttons/Button";
import { Badge } from "@/components/badges/Badge";
import { GlassCard } from "@/components/cards/GlassCard";
import { SkeletonTable } from "@/components/loading/Loading";
import { HorizontalBarChart, AppScatterChart } from "@/components/charts/Charts";
import { Modal } from "@/components/modals/Modal";
import { EmptyState } from "@/components/feedback/EmptyState";
import { ErrorState } from "@/components/feedback/ErrorState";
import { useExperiments } from "@/hooks/useResearch";
import { useExperimentStore } from "@/store/experimentStore";
import { formatMetric, formatDuration, snakeToTitle } from "@/utils/formatters";
import type { ExperimentResult } from "@/types/api";
import { useRouter } from "next/navigation";


function ExperimentTableRow({
  exp,
  rank,
  selected,
  onToggleSelect,
}: {
  exp: ExperimentResult;
  rank: number;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const featureData = exp.feature_importance
    ? Object.entries(exp.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8)
        .map(([name, value]) => ({ name, value }))
    : [];

  return (
    <>
      {/* ── Desktop Table Row ── */}
      <tr
        className={`
          border-b border-border-subtle
          hover:bg-surface-3 transition-colors cursor-pointer select-none
          ${selected ? "bg-brand-500/10 border-l-2 border-l-brand-500" : ""}
        `}
        onClick={() => setExpanded((p) => !p)}
      >
        {/* Rank */}
        <td className="px-4 py-3.5 text-center">
          {rank === 1 ? (
            <Trophy className="w-4 h-4 text-warning-400 mx-auto" />
          ) : (
            <span className="text-xs font-mono text-text-muted">#{rank}</span>
          )}
        </td>

        {/* Select */}
        <td className="px-3 py-3.5" onClick={(e) => { e.stopPropagation(); onToggleSelect(); }}>
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-3.5 h-3.5 accent-brand-500 rounded cursor-pointer"
          />
        </td>

        {/* Model */}
        <td className="px-4 py-3.5">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-text break-words leading-tight">{exp.model_name}</p>
            <p className="text-xs text-text-muted mt-0.5">{snakeToTitle(exp.model_type)}</p>
          </div>
        </td>

        {/* Pipeline */}
        <td className="px-4 py-3.5">
          <div className="flex flex-wrap gap-1">
            {exp.pipeline_steps?.slice(0, 3).map((step) => (
              <span
                key={step}
                className="text-[10px] px-1.5 py-0.5 rounded bg-surface-4 text-text-secondary font-mono border border-border-subtle"
              >
                {step}
              </span>
            ))}
            {(exp.pipeline_steps?.length ?? 0) > 3 && (
              <span className="text-[10px] text-text-muted self-center">
                +{(exp.pipeline_steps?.length ?? 0) - 3}
              </span>
            )}
          </div>
        </td>

        {/* Primary Metric */}
        <td className="px-4 py-3.5 text-right">
          {exp.primary_metric_value !== undefined ? (
            <span className="text-sm font-semibold text-brand-400 tabular-nums font-mono">
              {formatMetric(exp.primary_metric_value)}
            </span>
          ) : (
            <span className="text-text-muted">—</span>
          )}
          {exp.primary_metric_name && (
            <p className="text-[10px] text-text-muted uppercase tracking-wider">{exp.primary_metric_name}</p>
          )}
        </td>

        {/* Composite Score */}
        <td className="px-4 py-3.5 text-right">
          <span className="text-sm font-bold text-text tabular-nums font-mono">
            {exp.composite_score !== undefined
              ? formatMetric(exp.composite_score)
              : "—"}
          </span>
        </td>

        {/* Runtime */}
        <td className="px-4 py-3.5 text-right text-xs text-text-muted tabular-nums font-mono">
          {exp.runtime_seconds ? formatDuration(exp.runtime_seconds) : "—"}
        </td>

        {/* Status */}
        <td className="px-4 py-3.5">
          <Badge
            variant={exp.status === "pending" ? "queued" : (exp.status as "running" | "completed" | "failed")}
            label={exp.status}
          />
        </td>

        {/* Expand toggle */}
        <td className="px-3 py-3.5 text-text-muted text-right">
          <div className="w-6 h-6 rounded-md hover:bg-surface-4 flex items-center justify-center ml-auto">
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-text-secondary" />
            ) : (
              <ChevronDown className="w-4 h-4 text-text-muted" />
            )}
          </div>
        </td>
      </tr>

      {/* ── Desktop Expanded Details Drawer ── */}
      <AnimatePresence>
        {expanded && (
          <tr>
            <td colSpan={9} className="bg-surface-1 border-b border-border-subtle">
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="px-6 py-5 grid grid-cols-1 md:grid-cols-2 gap-6"
              >
                {/* Metrics */}
                <div>
                  <p className="text-xs text-text-muted uppercase tracking-wider mb-2 font-semibold">
                    All Metrics
                  </p>
                  {exp.primary_metric_rationale && (
                    <div className="mb-3 p-2.5 rounded bg-brand-500/10 border border-brand-500/20 text-xs text-brand-300">
                      <span className="font-semibold block text-brand-400 mb-0.5">
                        🎯 Primary Metric Selection Rationale ({exp.primary_metric_name})
                      </span>
                      {exp.primary_metric_rationale}
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      ["Accuracy", exp.accuracy],
                      ["F1 Score", exp.f1_score],
                      ["Precision", exp.precision],
                      ["Recall", exp.recall],
                      ["ROC-AUC", exp.roc_auc],
                      ["RMSE", exp.rmse],
                      ["MAE", exp.mae],
                      ["R²", exp.r2],
                    ]
                      .filter(([, v]) => v !== undefined)
                      .map(([label, value]) => (
                        <div
                          key={label as string}
                          className="flex justify-between items-center px-3 py-2 rounded-md bg-surface-2 border border-border-subtle"
                        >
                          <span className="text-xs text-text-muted">{label as string}</span>
                          <span className="text-xs font-semibold text-text font-mono">
                            {formatMetric(value as number)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Feature Importance */}
                {featureData.length > 0 && (
                  <div>
                    <p className="text-xs text-text-muted uppercase tracking-wider mb-3 font-semibold">
                      Feature Importance
                    </p>
                    <HorizontalBarChart data={featureData} height={200} />
                  </div>
                )}
              </motion.div>
            </td>
          </tr>
        )}
      </AnimatePresence>
    </>
  );
}

function ExperimentCardItem({
  exp,
  rank,
  selected,
  onToggleSelect,
}: {
  exp: ExperimentResult;
  rank: number;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const featureData = exp.feature_importance
    ? Object.entries(exp.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8)
        .map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div
      className={`
        p-4 border-b border-border-subtle last:border-0 flex flex-col gap-3
        ${selected ? "bg-brand-500/5 border-l-2 border-l-brand-500" : ""}
      `}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-4 h-4 accent-brand-500 rounded cursor-pointer shrink-0"
          />
          {rank === 1 ? (
            <Trophy className="w-4 h-4 text-warning-400 shrink-0" />
          ) : (
            <span className="text-xs font-mono text-text-muted w-5">#{rank}</span>
          )}
          <div className="min-w-0">
            <p className="text-sm font-semibold text-text break-words leading-tight">{exp.model_name}</p>
            <p className="text-xs text-text-muted">{snakeToTitle(exp.model_type)}</p>
          </div>
        </div>
        <Badge
          variant={exp.status === "pending" ? "queued" : (exp.status as "running" | "completed" | "failed")}
          label={exp.status}
        />
      </div>

      {/* Metrics summary card */}
      <div className="grid grid-cols-3 gap-2 py-2 px-3 rounded-md bg-surface-3 text-xs border border-border-subtle">
        <div>
          <span className="text-[10px] text-text-muted block uppercase">Score</span>
          <span className="font-bold text-text tabular-nums font-mono">
            {exp.composite_score !== undefined ? formatMetric(exp.composite_score) : "—"}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-text-muted block uppercase">Primary</span>
          <span className="font-semibold text-brand-400 tabular-nums font-mono">
            {exp.primary_metric_value !== undefined ? formatMetric(exp.primary_metric_value) : "—"}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-text-muted block uppercase">Runtime</span>
          <span className="text-text-secondary tabular-nums font-mono">
            {exp.runtime_seconds ? formatDuration(exp.runtime_seconds) : "—"}
          </span>
        </div>
      </div>

      {/* Pipeline steps & expand button */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1">
          {exp.pipeline_steps?.slice(0, 3).map((step) => (
            <span
              key={step}
              className="text-[10px] px-1.5 py-0.5 rounded bg-surface-4 text-text-secondary font-mono border border-border-subtle"
            >
              {step}
            </span>
          ))}
        </div>
        <button
          onClick={() => setExpanded((p) => !p)}
          className="text-xs text-brand-400 hover:text-brand-300 font-medium flex items-center gap-1 ml-auto"
        >
          {expanded ? "Hide details" : "Details"}
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Mobile Expanded Section */}
      {expanded && (
        <div className="pt-3 border-t border-border-subtle flex flex-col gap-4">
          <div>
            <p className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-semibold">
              Metrics Breakdown
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                ["Accuracy", exp.accuracy],
                ["F1 Score", exp.f1_score],
                ["Precision", exp.precision],
                ["Recall", exp.recall],
                ["ROC-AUC", exp.roc_auc],
                ["RMSE", exp.rmse],
                ["MAE", exp.mae],
                ["R²", exp.r2],
              ]
                .filter(([, v]) => v !== undefined)
                .map(([label, value]) => (
                  <div
                    key={label as string}
                    className="flex justify-between items-center px-2.5 py-1.5 rounded bg-surface-3 border border-border-subtle text-xs"
                  >
                    <span className="text-text-muted">{label as string}</span>
                    <span className="font-semibold text-text font-mono">
                      {formatMetric(value as number)}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {featureData.length > 0 && (
            <div>
              <p className="text-[10px] text-text-muted uppercase tracking-wider mb-2 font-semibold">
                Feature Importance
              </p>
              <HorizontalBarChart data={featureData} height={180} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ExperimentsPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);
  const router = useRouter();
  const { data: experiments, isLoading, isError, refetch } = useExperiments(jobId);

  const {
    filterModelType,
    filterStatus,
    selectedIds,
    compareModalOpen,
    setFilterModelType,
    setFilterStatus,
    toggleSelectExperiment,
    clearSelection,
    setCompareModalOpen,
  } = useExperimentStore();

  if (isError) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-12">
        <ErrorState
          title="Failed to Load Experiments"
          description={`Unable to fetch experiment results for job "${jobId}". The job may still be initializing or the backend server is unreachable.`}
          onRetry={() => refetch()}
          action={
            <Button
              variant="primary"
              size="sm"
              onClick={() => router.push(`/timeline/${jobId}`)}
            >
              View Timeline
            </Button>
          }
        />
      </div>
    );
  }


  // Filter experiments
  const filtered = (experiments ?? [])
    .filter((e) => !filterModelType || e.model_type === filterModelType)
    .filter((e) => !filterStatus || e.status === filterStatus)
    .sort(
      (a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0)
    );

  // Scatter data: runtime vs composite score
  const scatterData = filtered
    .filter((e) => e.runtime_seconds && e.composite_score !== undefined)
    .map((e) => ({
      x: e.runtime_seconds!,
      y: e.composite_score!,
      name: e.model_name,
    }));

  const modelTypes = [...new Set((experiments ?? []).map((e) => e.model_type))];
  const selectedExps = (experiments ?? []).filter((e) =>
    selectedIds.has(e.experiment_id)
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 md:py-8">
      {/* ── Page header ── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-text mb-1">
              Experiment Leaderboard
            </h1>
            <p className="text-sm text-text-muted">
              {filtered.length} experiments ranked by composite score
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {selectedIds.size >= 2 && (
              <Button
                variant="secondary"
                size="sm"
                icon={<GitCompare className="w-3.5 h-3.5" />}
                onClick={() => setCompareModalOpen(true)}
              >
                Compare ({selectedIds.size})
              </Button>
            )}
            <Button
              variant="primary"
              size="sm"
              icon={<ArrowRight className="w-3.5 h-3.5" />}
              onClick={() => router.push(`/recommendation/${jobId}`)}
            >
              Final Report
            </Button>
          </div>
        </div>
      </motion.div>

      {/* ── Scatter chart ── */}
      {scatterData.length > 0 && (
        <GlassCard className="mb-6" padding="md">
          <p className="text-xs text-text-muted uppercase tracking-wider font-semibold mb-4">
            Runtime vs. Score Trade-off
          </p>
          <AppScatterChart
            data={scatterData}
            xLabel="Runtime (s)"
            yLabel="Composite Score"
            height={220}
          />
        </GlassCard>
      )}

      {/* ── Controls & Filters ── */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="flex items-center gap-1.5 text-xs text-text-muted font-medium">
          <Filter className="w-3.5 h-3.5" />
          Filters:
        </div>
        <select
          value={filterModelType ?? ""}
          onChange={(e) => setFilterModelType(e.target.value || null)}
          className="text-xs px-3 py-1.5 rounded-md bg-surface-2 border border-border text-text focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          <option value="">All Model Types</option>
          {modelTypes.map((t) => (
            <option key={t} value={t}>{snakeToTitle(t)}</option>
          ))}
        </select>
        <select
          value={filterStatus ?? ""}
          onChange={(e) => setFilterStatus(e.target.value || null)}
          className="text-xs px-3 py-1.5 rounded-md bg-surface-2 border border-border text-text focus:outline-none focus:ring-1 focus:ring-brand-500"
        >
          <option value="">All Statuses</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="running">Running</option>
        </select>
        {(filterModelType || filterStatus) && (
          <button
            onClick={() => { setFilterModelType(null); setFilterStatus(null); }}
            className="text-xs text-brand-400 hover:text-brand-300 transition-colors font-medium"
          >
            Clear filters
          </button>
        )}
        {selectedIds.size > 0 && (
          <button
            onClick={clearSelection}
            className="text-xs text-text-muted hover:text-text-secondary transition-colors ml-auto"
          >
            Clear selection ({selectedIds.size})
          </button>
        )}
      </div>

      {/* ── Leaderboard Table / Card List ── */}
      <div className="card overflow-hidden">
        {isLoading ? (
          <div className="p-6">
            <SkeletonTable rows={6} />
          </div>
        ) : filtered.length === 0 ? (
          <EmptyState
            icon={FlaskConical}
            title="No experiments found"
            description={
              filterModelType || filterStatus
                ? "No experiment runs match your selected filters. Try clearing active filters to view all runs."
                : "The research job is still initializing or has not completed any experiment pipelines yet."
            }
            action={
              filterModelType || filterStatus ? (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setFilterModelType(null);
                    setFilterStatus(null);
                  }}
                >
                  Clear Filters
                </Button>
              ) : (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => router.push(`/timeline/${jobId}`)}
                >
                  View Timeline Progress
                </Button>
              )
            }
          />
        ) : (

          <div>
            {/* Desktop Table View */}
            <table className="hidden md:table w-full text-left">
              <thead>
                <tr className="border-b border-border-subtle bg-surface-1">
                  {["#", "✓", "Model", "Pipeline", "Primary Metric", "Score", "Runtime", "Status", ""].map((h) => (
                    <th key={h} className="px-4 py-3 text-[10px] font-semibold text-text-muted uppercase tracking-wider whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((exp, i) => (
                  <ExperimentTableRow
                    key={exp.experiment_id}
                    exp={exp}
                    rank={i + 1}
                    selected={selectedIds.has(exp.experiment_id)}
                    onToggleSelect={() => toggleSelectExperiment(exp.experiment_id)}
                  />
                ))}
              </tbody>
            </table>

            {/* Mobile Card List View */}
            <div className="md:hidden divide-y divide-border-subtle">
              {filtered.map((exp, i) => (
                <ExperimentCardItem
                  key={exp.experiment_id}
                  exp={exp}
                  rank={i + 1}
                  selected={selectedIds.has(exp.experiment_id)}
                  onToggleSelect={() => toggleSelectExperiment(exp.experiment_id)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Compare Modal ── */}
      <Modal
        open={compareModalOpen}
        onClose={() => setCompareModalOpen(false)}
        title={`Comparing ${selectedExps.length} Experiments`}
        size="xl"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left text-xs font-semibold text-text-muted py-2 pr-4">Metric</th>
                {selectedExps.map((e) => (
                  <th key={e.experiment_id} className="text-center text-xs font-semibold text-text py-2 px-4">
                    {e.model_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {[
                ["Composite Score", "composite_score"],
                ["Accuracy", "accuracy"],
                ["F1 Score", "f1_score"],
                ["ROC-AUC", "roc_auc"],
                ["Runtime (s)", "runtime_seconds"],
              ].map(([label, key]) => (
                <tr key={key} className="hover:bg-surface-4/40 transition-colors">
                  <td className="text-xs text-text-muted py-2.5 pr-4 font-medium">{label}</td>
                  {selectedExps.map((e) => {
                    const val = e[key as keyof ExperimentResult] as number | undefined;
                    return (
                      <td key={e.experiment_id} className="text-center text-xs font-semibold text-text py-2.5 px-4 font-mono">
                        {val !== undefined ? formatMetric(val) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Modal>
    </div>
  );
}
