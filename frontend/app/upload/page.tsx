"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileSpreadsheet,
  X,
  AlertCircle,
  Sparkles,
  Zap,
  Target,
  ArrowRight,
  Layers,
  TrendingUp,
  Sliders,
  CheckCircle2,
} from "lucide-react";

import { Button } from "@/components/buttons/Button";
import { ProgressBar } from "@/components/loading/Loading";
import { uploadDataset, startJob } from "@/services/apiClient";
import { formatBytes } from "@/utils/formatters";

const MAX_UPLOAD_SIZE_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB) || 150;
const MAX_FILE_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
const MISSION_MAX = 500;

const PRESET_MISSIONS = [
  "Predict customer churn with high recall and explain top risk drivers.",
  "Build a regression model to estimate target value with minimum RMSE.",
  "Classify high-risk financial transactions while maintaining 95%+ precision.",
];

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [mission, setMission] = useState("");
  const [taskType, setTaskType] = useState<"classification" | "regression" | "general">("general");
  const [status, setStatus] = useState<"idle" | "uploading" | "starting" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    if (f.size > MAX_FILE_SIZE) {
      setErrorMsg(`File size must be under ${MAX_UPLOAD_SIZE_MB} MB.`);
      return;
    }
    setFile(f);
    setErrorMsg("");
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
    disabled: status === "uploading" || status === "starting",
  });

  const handleSubmit = async () => {
    if (!file || !mission.trim()) return;
    setStatus("uploading");
    setUploadProgress(0);
    setErrorMsg("");

    try {
      // Simulate progress during upload
      const progressInterval = setInterval(() => {
        setUploadProgress((p) => Math.min(p + 8, 85));
      }, 200);

      const uploadResult = await uploadDataset(file, mission.trim(), taskType);
      clearInterval(progressInterval);
      setUploadProgress(90);

      setStatus("starting");
      const jobResult = await startJob(uploadResult.dataset_id, mission.trim(), taskType);
      setUploadProgress(100);

      // Small delay for UX
      await new Promise((r) => setTimeout(r, 600));
      router.push(`/timeline/${jobResult.job_id}`);
    } catch (err) {
      setStatus("error");
      setErrorMsg(
        err instanceof Error ? err.message : "Upload failed. Please try again."
      );
      setUploadProgress(0);
    }
  };

  const isSubmittable =
    file !== null &&
    mission.trim().length >= 10 &&
    status === "idle";

  const step1Done = file !== null;
  const step2Done = mission.trim().length >= 10;

  return (
    <div className="relative min-h-full">
      {/* Background radial highlight */}
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] pointer-events-none -z-0"
        style={{
          background:
            "radial-gradient(ellipse, rgba(118,255,3,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Page content */}
      <div className="relative z-10 flex flex-col items-center px-4 sm:px-6 pt-6 pb-16 max-w-2xl mx-auto">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="text-center mb-6"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/25 text-brand-400 text-xs font-semibold mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            Autonomous AI Research Launcher
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-text tracking-tight mb-2">
            Upload Dataset & Launch Mission
          </h1>
          <p className="text-text-muted text-sm leading-relaxed max-w-md mx-auto">
            Provide your raw CSV and research objective. Evidra handles profiling, preprocessing, pipeline execution, and model optimization.
          </p>
        </motion.div>

        {/* ── Visual Stepper Header ── */}
        <div className="w-full card p-3.5 mb-6 flex items-center justify-between text-xs bg-surface-2 border border-border">
          <div className={`flex items-center gap-2 ${step1Done ? "text-success-400 font-semibold" : "text-text font-medium"}`}>
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono ${
              step1Done ? "bg-success-500/20 text-success-400" : "bg-brand-500/20 text-brand-400"
            }`}>
              {step1Done ? "✓" : "1"}
            </span>
            <span>Dataset</span>
          </div>
          <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
          <div className={`flex items-center gap-2 ${step2Done ? "text-success-400 font-semibold" : step1Done ? "text-text font-medium" : "text-text-muted"}`}>
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono ${
              step2Done ? "bg-success-500/20 text-success-400" : step1Done ? "bg-brand-500/20 text-brand-400" : "bg-surface-4 text-text-muted"
            }`}>
              {step2Done ? "✓" : "2"}
            </span>
            <span>Mission</span>
          </div>
          <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
          <div className={`flex items-center gap-2 ${isSubmittable ? "text-brand-400 font-semibold" : "text-text-muted"}`}>
            <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono ${
              isSubmittable ? "bg-brand-500/20 text-brand-400 animate-pulse-soft" : "bg-surface-4 text-text-muted"
            }`}>
              3
            </span>
            <span>Launch</span>
          </div>
        </div>

        {/* Drop Zone Card */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.4 }}
          className="w-full mb-6"
        >
          <div
            {...getRootProps()}
            className={`
              relative w-full rounded-xl border-2 border-dashed p-8 sm:p-10 text-center
              cursor-pointer transition-all duration-200 select-none
              ${
                isDragActive
                  ? "border-brand-400 bg-brand-500/10 shadow-md"
                  : file
                  ? "border-success-500/50 bg-success-500/5"
                  : "border-border hover:border-brand-400/60 hover:bg-surface-3 bg-surface-2"
              }
            `}
          >
            <input {...getInputProps()} />

            <AnimatePresence mode="wait">
              {file ? (
                <motion.div
                  key="file-selected"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div className="w-12 h-12 rounded-xl bg-success-500/15 border border-success-500/30 flex items-center justify-center">
                    <FileSpreadsheet className="w-6 h-6 text-success-400" />
                  </div>
                  <div>
                    <p className="font-semibold text-success-400 text-sm">
                      {file.name}
                    </p>
                    <p className="text-xs text-text-muted mt-0.5 font-mono">
                      {formatBytes(file.size)} · CSV Dataset Ready
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                    }}
                    className="flex items-center gap-1.5 text-xs text-text-muted hover:text-error-400 transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                    Change dataset
                  </button>
                </motion.div>
              ) : (
                <motion.div
                  key="drop-prompt"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-3"
                >
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center border transition-colors duration-200 ${
                      isDragActive
                        ? "bg-brand-500/20 border-brand-400/50"
                        : "bg-surface-3 border-border"
                    }`}
                  >
                    <Upload
                      className={`w-6 h-6 transition-colors ${
                        isDragActive ? "text-brand-400" : "text-text-muted"
                      }`}
                    />
                  </div>
                  <div>
                    <p className="text-text font-semibold text-sm">
                      {isDragActive
                        ? "Drop your CSV file here"
                        : "Click to browse or drag & drop CSV file"}
                    </p>
                    <p className="text-text-muted text-xs mt-1">
                      Supports tabular CSV datasets up to {MAX_UPLOAD_SIZE_MB} MB
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Mission Input Form */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="w-full mb-6"
        >
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="mission-goal-input" className="text-sm font-semibold text-text flex items-center gap-1.5 cursor-pointer">
              <Target className="w-4 h-4 text-brand-400" />
              Research Goal & Mission
              <span className="text-brand-400">*</span>
            </label>
            <span className="text-xs text-text-muted font-mono">
              {mission.length}/{MISSION_MAX}
            </span>
          </div>

          <textarea
            id="mission-goal-input"
            value={mission}
            onChange={(e) => setMission(e.target.value.slice(0, MISSION_MAX))}

            placeholder="e.g., Predict customer churn with highest possible accuracy. Focus on recall for the positive class. Explain key factors driving churn."
            rows={4}
            disabled={status === "uploading" || status === "starting"}
            className="
              w-full px-4 py-3 rounded-lg text-sm
              bg-surface-2 border border-border
              text-text placeholder:text-text-muted
              focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500
              transition-all duration-200 resize-none
              disabled:opacity-50
            "
          />

          {/* Preset Mission Pills */}
          <div className="mt-3">
            <p className="text-[11px] text-text-muted font-medium mb-2 uppercase tracking-wider">
              Quick Mission Presets:
            </p>
            <div className="flex flex-col gap-1.5">
              {PRESET_MISSIONS.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setMission(preset)}
                  className="
                    text-left text-xs px-3 py-2 rounded-md
                    bg-surface-2 hover:bg-surface-3 border border-border-subtle hover:border-brand-500/30
                    text-text-secondary hover:text-text transition-colors flex items-center justify-between group
                  "
                >
                  <span className="truncate">{preset}</span>
                  <span className="text-[10px] text-brand-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0 ml-2">
                    Use Preset →
                  </span>
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Task Type Selector — Modern Interactive Checkbox Cards */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, duration: 0.4 }}
          className="w-full mb-6"
        >
          <div className="flex items-center justify-between mb-2.5">
            <label className="text-sm font-semibold text-text flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-brand-400" />
              Machine Learning Problem Type
            </label>
            <span className="text-[11px] text-text-muted">Select task objective</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                id: "general" as const,
                title: "General / Auto",
                desc: "Auto-detects target & metrics",
                badge: "Smart Auto",
                icon: Sparkles,
                activeColor: "border-brand-400 bg-brand-500/10 text-brand-400 shadow-[0_0_15px_rgba(118,255,3,0.15)]",
                iconBg: "bg-brand-500/15 text-brand-400 border-brand-500/30",
              },
              {
                id: "classification" as const,
                title: "Classification",
                desc: "Categories, labels & churn",
                badge: "Precision / F1",
                icon: Layers,
                activeColor: "border-info-400 bg-info-500/10 text-info-400 shadow-[0_0_15px_rgba(56,146,246,0.15)]",
                iconBg: "bg-info-500/15 text-info-400 border-info-500/30",
              },
              {
                id: "regression" as const,
                title: "Regression",
                desc: "Numeric value estimation",
                badge: "R² / RMSE",
                icon: TrendingUp,
                activeColor: "border-success-400 bg-success-500/10 text-success-400 shadow-[0_0_15px_rgba(0,230,118,0.15)]",
                iconBg: "bg-success-500/15 text-success-400 border-success-500/30",
              },
            ].map((option) => {
              const selected = taskType === option.id;
              const Icon = option.icon;

              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setTaskType(option.id)}
                  disabled={status === "uploading" || status === "starting"}
                  className={`
                    relative p-4 rounded-xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none
                    ${
                      selected
                        ? `${option.activeColor} ring-1 ring-current`
                        : "bg-surface-2 hover:bg-surface-3 border-border text-text-muted hover:text-text hover:border-border-subtle"
                    }
                    disabled:opacity-50 disabled:cursor-not-allowed
                  `}
                >
                  {/* Top Row: Icon + Custom Checkbox */}
                  <div className="flex items-center justify-between w-full mb-2.5">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${option.iconBg}`}>
                      <Icon className="w-4 h-4" />
                    </div>

                    {/* Checkbox indicator */}
                    <div
                      className={`w-5 h-5 rounded-md border flex items-center justify-center transition-all ${
                        selected
                          ? "bg-brand-500 border-brand-400 text-[#052620] shadow-sm"
                          : "border-border bg-surface-3 text-transparent"
                      }`}
                    >
                      <CheckCircle2 className={`w-3.5 h-3.5 ${selected ? "opacity-100" : "opacity-0"}`} />
                    </div>
                  </div>

                  {/* Text Details */}
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <p className={`text-xs font-bold ${selected ? "text-text" : "text-text"}`}>
                        {option.title}
                      </p>
                    </div>
                    <p className="text-[11px] text-text-muted leading-tight mb-2">
                      {option.desc}
                    </p>
                    <span className="inline-block text-[9px] font-mono font-semibold uppercase px-1.5 py-0.5 rounded bg-surface-4 text-text-secondary border border-border-subtle">
                      {option.badge}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </motion.div>

        {/* Error Notification Banner */}
        <AnimatePresence>
          {errorMsg && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="w-full mb-4 px-4 py-3 rounded-lg bg-error-500/10 border border-error-500/25 flex items-start gap-2.5"
            >
              <AlertCircle className="w-4 h-4 text-error-400 shrink-0 mt-0.5" />
              <p className="text-sm text-error-400">{errorMsg}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Upload / Start Progress Bar */}
        <AnimatePresence>
          {(status === "uploading" || status === "starting") && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="w-full mb-4"
            >
              <ProgressBar
                value={uploadProgress}
                label={
                  status === "starting"
                    ? "Initializing research engine job…"
                    : "Uploading CSV dataset…"
                }
                color="brand"
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Launch Button */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="w-full"
        >
          <Button
            variant="primary"
            size="lg"
            className="w-full text-base py-3"
            loading={status === "uploading" || status === "starting"}
            disabled={!isSubmittable}
            onClick={handleSubmit}
            icon={<Zap className="w-4 h-4" />}
          >
            {status === "uploading"
              ? "Uploading Dataset…"
              : status === "starting"
              ? "Starting Research Job…"
              : "Launch Research Engine"}
          </Button>
          {!file && (
            <p className="text-xs text-text-muted text-center mt-3">
              Upload a CSV file and enter your goal to enable research execution
            </p>
          )}
        </motion.div>

      </div>
    </div>
  );
}
