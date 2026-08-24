/* ─────────────────────────────────────────────
   DataPilot-AI — Shared TypeScript API Types
   Mirrors backend Pydantic models exactly
   ───────────────────────────────────────────── */

// ── Job Status ────────────────────────────────
export type JobStatus =
  | "queued"
  | "running"
  | "profiling"
  | "understanding"
  | "planning"
  | "executing"
  | "evaluating"
  | "directing"
  | "reporting"
  | "completed"
  | "failed"
  | "cancelled";

// ── Pipeline Stages ───────────────────────────
export type PipelineStage =
  | "profiling"
  | "understanding"
  | "planning"
  | "executing"
  | "evaluating"
  | "decision"
  | "reporting";

// ── Dataset ───────────────────────────────────
export interface Dataset {
  dataset_id: string;
  filename: string;
  file_size_bytes: number;
  row_count: number;
  column_count: number;
  upload_timestamp: string;
  status: "uploaded" | "profiled" | "processing";
  mission_brief?: string;
  profile?: SemanticProfile | Record<string, unknown>;
}

export interface ColumnProfile {
  name: string;
  dtype?: string;
  type?: string;
  missing_count?: number;
  missing_percent?: number;
  missing_pct?: number;
  unique_count?: number;
  distinct_count?: number;
  sample_values?: (string | number | null)[];
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  distribution?: string;
  is_target?: boolean;
}

export interface QualityIssue {
  problem?: string;
  warning_type?: string;
  severity: "low" | "medium" | "high";
  description?: string;
  message?: string;
  affected_columns?: string[];
  column?: string;
}

export type QualityWarning = QualityIssue;

export interface DatasetSummary {
  rows?: number;
  columns?: number;
  memory_mb?: number;
  filename?: string;
  file_size_bytes?: number;
  target?: {
    target_column?: string;
    task_type?: "classification" | "regression" | "general";
  };
}

export interface SemanticProfile {
  dataset_id?: string;
  row_count?: number;
  column_count?: number;
  file_size_bytes?: number;
  missing_cells_total?: number;
  missing_percent_overall?: number;
  numeric_columns?: number;
  categorical_columns?: number;
  datetime_columns?: number;
  boolean_columns?: number;
  detected_target_column?: string;
  detected_task_type?: "classification" | "regression" | "unknown" | "general";
  user_task_type?: string;
  dataset_summary?: DatasetSummary;
  column_profiles?: ColumnProfile[];
  quality_issues?: QualityIssue[];
  quality_warnings?: QualityWarning[];
  memory_usage_mb?: number;
}

// ── Job ───────────────────────────────────────
export interface Job {
  job_id: string;
  dataset_id: string;
  mission: string;
  status: JobStatus;
  current_stage?: PipelineStage;
  progress_percent: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface JobLogEntry {
  id: string;
  timestamp: string;
  level: "info" | "warning" | "error" | "success";
  message: string;
  stage?: PipelineStage | string;
}

// ── Experiment ────────────────────────────────
export interface ExperimentResult {
  experiment_id: string;
  job_id: string;
  pipeline_name: string;
  model_name: string;
  model_type: string;
  status: "pending" | "running" | "completed" | "failed";
  primary_metric_name?: string;
  primary_metric_value?: number;
  primary_metric_rationale?: string;
  composite_score?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  rmse?: number;
  mae?: number;
  r2?: number;
  runtime_seconds?: number;
  feature_importance?: Record<string, number>;
  pipeline_steps?: string[];
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

// ── Knowledge Finding ─────────────────────────
export interface KnowledgeFinding {
  finding_id: string;
  job_id: string;
  iteration: number;
  category: string;
  finding: string;
  confidence: number;
  accepted: boolean;
  evidence?: string;
  created_at: string;
}

// ── Report / Final Recommendation ─────────────
export interface FinalRecommendation {
  recommended_model: string;
  recommended_pipeline: string[];
  hyperparameters?: Record<string, string | number | boolean | null | undefined | Record<string, unknown> | unknown[]>;
  confidence_score: number;
  composite_score: number;
  primary_metric_name: string;
  primary_metric_value: number;
  reasoning: string;
  key_findings: string[];
  implementation_tips: string[];
  experiment_id: string;
}

export interface Report {
  report_id: string;
  job_id: string;
  dataset_id: string;
  status: "generating" | "completed" | "failed";
  recommendation?: FinalRecommendation;
  experiment_count: number;
  knowledge_findings_count: number;
  markdown_report?: string;
  created_at: string;
  completed_at?: string;
}

// ── WebSocket Events ──────────────────────────
export type WSEventType =
  | "job.status_changed"
  | "job.progress"
  | "job.stage_update"
  | "job.completed"
  | "job.failed"
  | "experiment.started"
  | "experiment.completed"
  | "knowledge.updated"
  | "log.message";

export interface WSEvent {
  event: WSEventType;
  job_id: string;
  timestamp: string;
  data: WSEventData;
}

export interface WSEventData {
  status?: JobStatus;
  stage?: PipelineStage;
  progress_percent?: number;
  message?: string;
  experiment_id?: string;
  level?: "info" | "warning" | "error" | "success";
  finding?: KnowledgeFinding;
  report?: Report;
}

// ── API Response Wrappers ─────────────────────
export interface UploadResponse {
  dataset_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  file_size_bytes: number;
  message: string;
}

export interface StartJobResponse {
  job_id: string;
  dataset_id: string;
  status: JobStatus;
  message: string;
}

export interface DashboardData {
  total_jobs: number;
  completed_jobs: number;
  total_experiments: number;
  status_counts?: Record<string, number>;
  recent_jobs: Job[];
}
