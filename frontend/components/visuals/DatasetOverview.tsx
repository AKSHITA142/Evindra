"use client";

import React from "react";
import { Badge } from "@/components/badges/Badge";
import { GlassCard } from "@/components/cards/GlassCard";
import { FileText, Database, Target, AlertTriangle, Layers, BarChart2 } from "lucide-react";
import type { Dataset, ColumnProfile, QualityWarning } from "@/types/api";
import { formatBytes } from "@/utils/formatters";

interface DatasetOverviewProps {
  dataset: Dataset;
  className?: string;
}

interface ColumnData extends Omit<Partial<ColumnProfile>, "distinct_count"> {
  name: string;
  type?: string;
  dtype?: string;
  missing_pct?: number;
  missing_percent?: number;
  missing_count?: number;
  distinct_count?: number | string;
  unique_count?: number;
  sample_values?: (string | number | null)[];
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  skewness?: number | null;
  encoding_recommendation?: string;
  scaling_recommendation?: string;
}

interface QualityIssueItem extends Partial<QualityWarning> {
  problem?: string;
  warning_type?: string;
  severity?: "low" | "medium" | "high";
  message?: string;
  description?: string;
  affected_columns?: string[];
}

export function DatasetOverview({ dataset, className = "" }: DatasetOverviewProps) {
  const profile = (dataset.profile || {}) as Record<string, unknown>;
  const datasetSummary = (profile.dataset_summary || {}) as Record<string, unknown>;
  const columnProfiles: ColumnData[] = (profile.column_profiles || []) as ColumnData[];
  const qualityIssues: QualityIssueItem[] = (profile.quality_issues || profile.quality_warnings || []) as QualityIssueItem[];

  const targetInfo = (datasetSummary.target || {}) as Record<string, unknown>;
  const targetCol = String(targetInfo.target_column || profile.detected_target_column || "N/A");
  const taskType = String(targetInfo.task_type || profile.detected_task_type || "N/A");

  const rows = dataset.row_count || (datasetSummary.rows as number) || (profile.row_count as number) || 0;
  const cols = dataset.column_count || (datasetSummary.columns as number) || (profile.column_count as number) || 0;
  const fileSize = dataset.file_size_bytes || (datasetSummary.file_size_bytes as number) || (profile.file_size_bytes as number) || 0;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* 1. Mission Brief Banner */}
      {dataset.mission_brief && (
        <GlassCard className="p-5 border-l-2 border-brand-500 bg-brand-500/[0.06]">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-brand-500/15 text-brand-500">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wider text-brand-500 mb-1">
                Dataset Mission & Objective
              </h4>
              <p className="text-sm text-text font-medium">
                {dataset.mission_brief}
              </p>
            </div>
          </div>
        </GlassCard>
      )}

      {/* 2. Top Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <GlassCard className="p-4 flex items-center gap-3">
          <div className="p-3 rounded-lg bg-info-500/10 text-info-400">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-text-secondary">Total Rows</div>
            <div className="text-lg font-bold text-text">{rows.toLocaleString()}</div>
          </div>
        </GlassCard>

        <GlassCard className="p-4 flex items-center gap-3">
          <div className="p-3 rounded-lg bg-success-500/10 text-success-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-text-secondary">Total Columns</div>
            <div className="text-lg font-bold text-text">{cols.toLocaleString()}</div>
          </div>
        </GlassCard>

        <GlassCard className="p-4 flex items-center gap-3">
          <div className="p-3 rounded-lg bg-surface-4 text-text-secondary">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-text-secondary">File Size</div>
            <div className="text-lg font-bold text-text">{formatBytes(fileSize)}</div>
          </div>
        </GlassCard>

        <GlassCard className="p-4 flex items-center gap-3 border-l-2 border-brand-500">
          <div className="p-3 rounded-lg bg-warning-500/10 text-warning-400 shrink-0">
            <BarChart2 className="w-5 h-5" />
          </div>
          <div className="overflow-hidden">
            <div className="text-xs text-text-secondary flex items-center gap-1.5">
              <span>Target & Task</span>
              <span className="px-1.5 py-0.2 text-[9px] font-mono bg-brand-500/15 text-brand-500 rounded border border-brand-500/30">
                AUTO-DETECTED
              </span>
            </div>
            <div className="text-sm font-semibold text-text truncate flex items-center gap-1.5 mt-0.5">
              <span className="text-brand-500 font-mono">{targetCol}</span>
              <span className="text-xs text-text-secondary font-normal">({taskType})</span>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* 3. Detailed Column Profiles Table */}
      <GlassCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-text flex items-center gap-2">
            <Database className="w-4 h-4 text-brand-500" />
            Column Schema & Statistical Profiles ({columnProfiles.length})
          </h3>
        </div>

        {columnProfiles.length === 0 ? (
          <div className="text-center py-8 text-text-secondary text-sm">
            No column profiles available for this dataset.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border text-text-secondary uppercase tracking-wider font-semibold">
                  <th className="py-3 px-3">Column Name</th>
                  <th className="py-3 px-3">Type</th>
                  <th className="py-3 px-3">Missing</th>
                  <th className="py-3 px-3">Unique</th>
                  <th className="py-3 px-3">Distribution Stats</th>
                  <th className="py-3 px-3">Recommendations</th>
                  <th className="py-3 px-3">Sample Values</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle text-text">
                {columnProfiles.map((col: ColumnData) => {
                  const missingPct = col.missing_pct ?? col.missing_percent ?? 0;
                  const missingCount = col.missing_count ?? 0;
                  const distinctCount = col.distinct_count ?? col.unique_count ?? "N/A";
                  const colType = col.type || col.dtype || "unknown";

                  return (
                    <tr key={col.name} className="hover:bg-white/[0.025] transition-colors">
                      <td className="py-3 px-3 font-medium text-text">
                        {col.name}
                        {col.name === targetCol && (
                          <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-brand-500/15 text-brand-500 rounded border border-brand-500/30 font-mono">
                            TARGET
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <Badge
                          label={String(colType).replace("_", " ")}
                          variant={String(colType).includes("high") ? "warning" : String(colType) === "numeric" ? "info" : "neutral"}
                        />
                      </td>
                      <td className="py-3 px-3">
                        <span className={missingPct > 0 ? "text-warning-400 font-semibold" : "text-text-secondary"}>
                          {missingCount} ({missingPct}%)
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono text-text-secondary">
                        {typeof distinctCount === "number" ? distinctCount.toLocaleString() : distinctCount}
                      </td>
                      <td className="py-3 px-3 text-text-secondary font-mono">
                        {col.mean !== undefined && col.mean !== null ? (
                          <div>
                            <div>mean: {col.mean} | std: {col.std}</div>
                            <div>min: {col.min} | max: {col.max}</div>
                            {col.skewness !== null && col.skewness !== undefined && (
                              <div className="text-text-muted text-[10px]">skew: {col.skewness}</div>
                            )}
                          </div>
                        ) : (
                          <span className="text-text-muted">—</span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <div className="flex flex-col gap-1">
                          {col.encoding_recommendation && (
                            <span className="text-[10px] px-1.5 py-0.5 bg-brand-500/10 text-brand-300 border border-brand-500/20 rounded">
                              Enc: {col.encoding_recommendation}
                            </span>
                          )}
                          {col.scaling_recommendation && (
                            <span className="text-[10px] px-1.5 py-0.5 bg-success-500/10 text-success-400 border border-success-500/20 rounded">
                              Scale: {col.scaling_recommendation}
                            </span>
                          )}
                          {!col.encoding_recommendation && !col.scaling_recommendation && (
                            <span className="text-text-muted text-[10px]">—</span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-3 max-w-[200px] truncate text-text-secondary font-mono">
                        {col.sample_values && col.sample_values.length > 0
                          ? col.sample_values.join(", ")
                          : "N/A"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>

      {/* 4. Quality Issues & Warnings */}
      {qualityIssues.length > 0 && (
        <GlassCard className="p-6 border-l-2 border-warning-500">
          <h3 className="text-base font-semibold text-warning-400 flex items-center gap-2 mb-3">
            <AlertTriangle className="w-5 h-5" />
            Detected Data Quality Issues ({qualityIssues.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {qualityIssues.map((issue: QualityIssueItem, idx: number) => (
              <div key={idx} className="p-3 bg-surface-1 rounded-lg border border-warning-500/20 text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-text">
                    {issue.problem || issue.warning_type || "Quality Issue"}
                  </span>
                  <Badge
                    label={issue.severity || "medium"}
                    variant={issue.severity === "high" ? "cancelled" : "warning"}
                  />
                </div>
                <p className="text-text-secondary mb-1">{issue.description || issue.message}</p>
                {issue.affected_columns && issue.affected_columns.length > 0 && (
                  <div className="text-[10px] text-warning-400/80">
                    Columns: {issue.affected_columns.join(", ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        </GlassCard>
      )}
    </div>
  );
}
