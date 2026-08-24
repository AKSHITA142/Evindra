"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { getJob, getJobLogs, getExperiments, getReport, getDashboard, listDatasets, getDataset } from "@/services/apiClient";

import { wsClient } from "@/services/websocketClient";
import { useResearchStore } from "@/store/researchStore";
import type { WSEvent } from "@/types/api";


// ── useJob — fetch job status with polling fallback ──────────────────
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Auto-refetch while running; stop when terminal
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false;
      }
      return 10_000; // 10-second fallback polling
    },
  });
}

// ── useJobLogs — fetch historical execution logs ──────────────────────
export function useJobLogs(jobId: string | null) {
  return useQuery({
    queryKey: ["jobLogs", jobId],
    queryFn: () => (jobId ? getJobLogs(jobId) : Promise.resolve([])),
    enabled: !!jobId,
    staleTime: 10_000,
  });
}

// ── useExperiments — fetch experiments list ───────────────────────────
export function useExperiments(jobId: string | null) {
  return useQuery({
    queryKey: ["experiments", jobId],
    queryFn: () => getExperiments(jobId!),
    enabled: !!jobId,
    staleTime: 5_000,
  });
}

// ── useReport — fetch final report ────────────────────────────────────
export function useReport(jobId: string | null) {
  return useQuery({
    queryKey: ["report", jobId],
    queryFn: () => getReport(jobId!),
    enabled: !!jobId,
    staleTime: 60_000,
    retry: 3,
  });
}

// ── useWebSocket — subscribe to live job events ───────────────────────
export function useWebSocket(jobId: string | null) {
  const queryClient = useQueryClient();
  const { setWsConnected, setCurrentStage, setProgressPercent, addLogMessage } =
    useResearchStore();
  const unsubRef = useRef<(() => void) | null>(null);
  const unsubStatusRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!jobId) {
      setWsConnected(false);
      return;
    }

    // Subscribe to live connection status changes
    unsubStatusRef.current = wsClient.onStatusChange((connected) => {
      setWsConnected(connected);
    });

    wsClient.connect(jobId);

    const handler = (event: WSEvent) => {
      const { data } = event;

      // Update stores
      if (data.stage) setCurrentStage(data.stage);
      if (data.progress_percent !== undefined)
        setProgressPercent(data.progress_percent);

      // Log all events to the live log panel
      if (data.message) {
        addLogMessage({
          id: `${event.timestamp}-${Math.random()}`,
          timestamp: event.timestamp,
          level: data.level ?? "info",
          message: data.message,
          stage: data.stage,
        });
      }

      // Invalidate TanStack Query caches based on event type
      switch (event.event) {
        case "job.status_changed":
        case "job.progress":
        case "job.stage_update":
          queryClient.invalidateQueries({ queryKey: ["job", jobId] });
          break;
        case "experiment.completed":
          queryClient.invalidateQueries({ queryKey: ["experiments", jobId] });
          break;
        case "job.completed":
          queryClient.invalidateQueries({ queryKey: ["job", jobId] });
          queryClient.invalidateQueries({ queryKey: ["experiments", jobId] });
          queryClient.invalidateQueries({ queryKey: ["report", jobId] });
          break;
        case "job.failed":
          queryClient.invalidateQueries({ queryKey: ["job", jobId] });
          break;
      }
    };

    unsubRef.current = wsClient.on("all", handler);

    return () => {
      unsubRef.current?.();
      unsubStatusRef.current?.();
    };
  }, [jobId, queryClient, setWsConnected, setCurrentStage, setProgressPercent, addLogMessage]);
}

// ── useDashboard — overview stats ────────────────────────────────────
export function useDashboard() {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: getDashboard,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

// ── useDatasets — full list for the datasets table ───────────────────
export function useDatasets(skip = 0, limit = 50) {
  return useQuery({
    queryKey: ["datasets", skip, limit],
    queryFn: () => listDatasets(skip, limit),
    staleTime: 60_000,
  });
}

// ── useDataset — fetch single dataset details ─────────────────────────
export function useDataset(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => getDataset(datasetId!),
    enabled: !!datasetId,
    staleTime: 60_000,
  });
}


